"""Background run ownership around the existing application composition root."""

from __future__ import annotations

import asyncio
import inspect
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.application import ApplicationStartupError, execute_task
from app.config import AgentLimits
from app.model_types import ModelClient
from app.runtime import AgentRunResult
from app.web.config import WebSettings
from app.web.events import EventBacklog, WebEventSink
from app.web.rate_limit import RateLimitDecision, SlidingWindowRateLimiter
from app.web.sessions import SessionRecord

LOGGER = logging.getLogger("file_agent.web.runs")


class RunManagerError(RuntimeError):
    status_code = 400
    code = "RUN_ERROR"

    def __init__(self, message: str, *, retry_after_seconds: int | None = None) -> None:
        self.safe_message = message
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


class ActiveRunError(RunManagerError):
    status_code = 409
    code = "ACTIVE_RUN"


class RunNotFoundError(RunManagerError):
    status_code = 404
    code = "RUN_NOT_FOUND"


class RunRateLimitError(RunManagerError):
    status_code = 429
    code = "RATE_LIMITED"


class GlobalConcurrencyError(RunManagerError):
    status_code = 503
    code = "GLOBAL_CONCURRENCY"


class RunShuttingDownError(RunManagerError):
    status_code = 503
    code = "SERVICE_SHUTTING_DOWN"


@dataclass
class RunRecord:
    run_id: str
    session_id: str
    task: str
    trace_path: Path
    created_at: datetime
    backlog: EventBacklog
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    status: str = "running"
    answer: str | None = None
    reason: str | None = None
    reason_code: str | None = None
    result: AgentRunResult | None = None
    background_task: asyncio.Task[None] | None = None
    cancel_requested: bool = False

    def public_result(self) -> dict[str, Any]:
        if self.result is None:
            return {
                "run_id": self.run_id,
                "status": self.status,
                "answer": self.answer,
                "reason": self.reason,
                "reason_code": self.reason_code,
                "model_calls": 0,
                "tool_calls": 0,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "available": False,
                    "exact": False,
                    "breakdown_available": False,
                    "total_available": False,
                },
                "elapsed_ms": 0,
                "changed_mutations": 0,
                "failed_mutations": 0,
                "finish_reason": None,
                "trace_available": self.trace_path.is_file(),
            }
        result = self.result
        return {
            "run_id": self.run_id,
            "status": result.status.value,
            "answer": result.answer,
            "reason": result.reason,
            "reason_code": result.reason_code,
            "model_calls": result.model_calls,
            "tool_calls": result.tool_calls,
            "usage": result.usage.model_dump(mode="json"),
            "elapsed_ms": result.elapsed_ms,
            "changed_mutations": len(result.changed_mutations),
            "failed_mutations": len(result.failed_mutations),
            "finish_reason": result.finish_reason,
            "trace_available": self.trace_path.is_file(),
        }


class RunManager:
    """Apply Web admission controls, then call the shared execute_task runtime."""

    def __init__(
        self,
        settings: WebSettings,
        model_client_factory: Callable[[], ModelClient],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._factory = model_client_factory
        self._clock = clock
        self._runs: dict[str, RunRecord] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._active_count = 0
        self._accepting_runs = True
        self._session_limiter = SlidingWindowRateLimiter(
            limit=settings.max_runs_per_session_hour,
            clock=clock,
        )
        self._ip_limiter = SlidingWindowRateLimiter(
            limit=settings.max_runs_per_ip_hour,
            clock=clock,
        )

    @property
    def active_count(self) -> int:
        return self._active_count

    def start(self, session: SessionRecord, *, task: str, client_ip: str) -> RunRecord:
        if not self._accepting_runs:
            raise RunShuttingDownError("The Web service is shutting down.")
        normalized = task.strip()
        if not normalized:
            raise RunManagerError("Task must not be empty.")
        if len(task) > self.settings.max_task_chars:
            raise RunManagerError("Task exceeds the configured character limit.")
        if "\x00" in task:
            raise RunManagerError("Task must not contain NUL characters.")
        if session.active_run_id is not None:
            raise ActiveRunError("This session already has an active run.")

        session_limit = self._session_limiter.check(session.session_id)
        ip_limit = self._ip_limiter.check(client_ip)
        if not session_limit.allowed:
            raise self._rate_error(session_limit)
        if not ip_limit.allowed:
            raise self._rate_error(ip_limit)
        if self._active_count >= self.settings.max_concurrent_runs:
            raise GlobalConcurrencyError(
                "The Web demo is at its global run limit.",
                retry_after_seconds=1,
            )

        self._session_limiter.check(session.session_id, consume=True)
        self._ip_limiter.check(client_ip, consume=True)
        now = self._clock()
        session.recent_run_timestamps.append(now)
        run_id = secrets.token_urlsafe(24)
        run_directory = session.runs_path / run_id
        run_directory.mkdir(parents=False, exist_ok=False)
        record = RunRecord(
            run_id=run_id,
            session_id=session.session_id,
            task=task,
            trace_path=run_directory / "trace.jsonl",
            created_at=datetime.now(UTC),
            backlog=EventBacklog(self.settings.event_backlog_limit),
        )
        self._runs[run_id] = record
        session.active_run_id = run_id
        self._active_count += 1
        background = asyncio.create_task(self._execute(record, session))
        record.background_task = background
        self._tasks.add(background)
        background.add_done_callback(self._tasks.discard)
        return record

    def get(self, run_id: str, session_id: str) -> RunRecord:
        record = self._runs.get(run_id)
        if record is None or record.session_id != session_id:
            raise RunNotFoundError("Run was not found for this session.")
        return record

    def cancel(self, run_id: str, session_id: str) -> RunRecord:
        record = self.get(run_id, session_id)
        if record.status == "running":
            record.cancel_requested = True
            record.cancel_event.set()
        return record

    def forget_session(self, session_id: str) -> None:
        for run_id in [
            item.run_id for item in self._runs.values() if item.session_id == session_id
        ]:
            record = self._runs[run_id]
            if record.status != "running":
                self._runs.pop(run_id, None)
        self._session_limiter.discard(session_id)

    async def shutdown(self) -> None:
        self._accepting_runs = False
        for record in self._runs.values():
            if record.status == "running":
                record.cancel_event.set()
        tasks = tuple(self._tasks)
        if not tasks:
            return
        done, pending = await asyncio.wait(
            tasks,
            timeout=self.settings.shutdown_grace_seconds,
        )
        for task in pending:
            task.cancel()
        if pending:
            cancelled, still_pending = await asyncio.wait(
                pending,
                timeout=min(1.0, self.settings.shutdown_grace_seconds),
            )
            done |= cancelled
            if still_pending:
                LOGGER.warning(
                    "web_shutdown_incomplete category=TASK_CANCEL_TIMEOUT count=%d",
                    len(still_pending),
                )
        if done:
            await asyncio.gather(*done, return_exceptions=True)

    async def _execute(self, record: RunRecord, session: SessionRecord) -> None:
        sink = WebEventSink(record.backlog)
        model: ModelClient | None = None
        final_payload = record.public_result()
        try:
            model = self._factory()
            limits = AgentLimits(
                max_model_turns=self.settings.web_max_model_turns,
                max_tool_calls=self.settings.web_max_tool_calls,
                max_runtime_seconds=self.settings.web_max_runtime_seconds,
                max_total_tokens=self.settings.web_max_total_tokens,
            )
            result = await execute_task(
                workspace=session.workspace_path,
                task=record.task,
                trace_path=record.trace_path,
                limits=limits,
                event_sink=sink,
                cancel_event=record.cancel_event,
                model_client=model,
            )
            record.result = result
            record.status = result.status.value
            record.answer = result.answer
            record.reason = result.reason
            record.reason_code = result.reason_code
            final_payload = record.public_result()
        except ApplicationStartupError as exc:
            record.status = "failed"
            record.reason = exc.safe_message
            record.reason_code = exc.code
            final_payload = record.public_result()
        except asyncio.CancelledError:
            record.status = "cancelled"
            record.reason = "The run was cancelled during server shutdown."
            record.reason_code = "SERVER_SHUTDOWN"
            final_payload = record.public_result()
            raise
        except Exception:  # noqa: BLE001 - background failures stay inside the run record
            record.status = "failed"
            record.reason = "The run failed inside the Web execution boundary."
            record.reason_code = "WEB_RUN_FAILURE"
            final_payload = record.public_result()
        finally:
            await self._close_owned_model(model)
            self._active_count = max(0, self._active_count - 1)
            if session.active_run_id == record.run_id:
                session.active_run_id = None
            await record.backlog.finish(datetime.now(UTC), final_payload)

    @staticmethod
    async def _close_owned_model(model: ModelClient | None) -> None:
        if model is None:
            return
        close = getattr(model, "aclose", None)
        if close is None:
            return
        try:
            value = close()
            if inspect.isawaitable(value):
                await value
        except Exception:  # noqa: BLE001 - cleanup cannot mask the run result
            return

    @staticmethod
    def _rate_error(decision: RateLimitDecision) -> RunRateLimitError:
        return RunRateLimitError(
            "The run rate limit was reached.",
            retry_after_seconds=decision.retry_after_seconds,
        )
