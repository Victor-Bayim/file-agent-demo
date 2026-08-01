from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.model_types import ModelClient, ModelMessage, ModelResponse
from app.runtime import AgentRunStatus
from app.web.config import WebSettings
from app.web.runs import (
    ActiveRunError,
    GlobalConcurrencyError,
    RunManager,
    RunRateLimitError,
    RunShuttingDownError,
)
from app.web.sessions import SessionManager
from tests.fake_model import FakeModelClient
from tests.web.conftest import model_response, tool_call


class BlockingModel(ModelClient):
    def __init__(self, release: asyncio.Event) -> None:
        self.release = release
        self.started = asyncio.Event()

    async def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, object]],
    ) -> ModelResponse:
        del messages, tools
        self.started.set()
        await self.release.wait()
        return model_response("Released.")


def test_background_run_completes_and_clears_active(
    web_settings: WebSettings,
) -> None:
    async def exercise() -> None:
        sessions = SessionManager(web_settings)
        sessions.start()
        session = sessions.create_session(client_ip="one")
        manager = RunManager(
            web_settings,
            lambda: FakeModelClient([model_response("Done.")]),
        )
        record = manager.start(session, task="Respond safely.", client_ip="one")
        assert record.status == "running"
        assert session.active_run_id == record.run_id
        assert record.background_task is not None
        await record.background_task

        payload = record.public_result()
        assert payload["status"] == "completed"
        assert payload["model_calls"] == 1
        assert payload["usage"]["total_tokens"] == 3  # type: ignore[index]
        assert session.active_run_id is None
        assert record.trace_path.is_file()
        sessions.shutdown()

    asyncio.run(exercise())


def test_same_session_and_global_concurrency_are_enforced(
    web_settings: WebSettings,
) -> None:
    async def exercise() -> None:
        release = asyncio.Event()
        settings = web_settings.model_copy(update={"max_concurrent_runs": 1})
        sessions = SessionManager(settings)
        sessions.start()
        first = sessions.create_session(client_ip="one")
        second = sessions.create_session(client_ip="two")
        manager = RunManager(settings, lambda: BlockingModel(release))
        first_run = manager.start(first, task="Wait.", client_ip="one")
        with pytest.raises(ActiveRunError):
            manager.start(first, task="Second.", client_ip="one")
        with pytest.raises(GlobalConcurrencyError):
            manager.start(second, task="Other.", client_ip="two")
        release.set()
        assert first_run.background_task is not None
        await first_run.background_task
        sessions.shutdown()

    asyncio.run(exercise())


def test_session_and_ip_rate_limits_use_no_sleep(web_settings: WebSettings) -> None:
    async def exercise() -> None:
        now = [0.0]
        settings = web_settings.model_copy(
            update={"max_runs_per_session_hour": 1, "max_runs_per_ip_hour": 1}
        )
        sessions = SessionManager(settings, clock=lambda: now[0])
        sessions.start()
        first = sessions.create_session(client_ip="same")
        second = sessions.create_session(client_ip="same")
        manager = RunManager(
            settings,
            lambda: FakeModelClient([model_response("Done.")]),
            clock=lambda: now[0],
        )
        record = manager.start(first, task="One.", client_ip="same")
        assert record.background_task is not None
        await record.background_task
        with pytest.raises(RunRateLimitError) as session_error:
            manager.start(first, task="Again.", client_ip="other")
        with pytest.raises(RunRateLimitError) as ip_error:
            manager.start(second, task="Again.", client_ip="same")
        assert session_error.value.retry_after_seconds == 3600
        assert ip_error.value.retry_after_seconds == 3600
        sessions.shutdown()

    asyncio.run(exercise())


def test_cancel_sets_event_without_rollback_claim(web_settings: WebSettings) -> None:
    async def exercise() -> None:
        sessions = SessionManager(web_settings)
        sessions.start()
        session = sessions.create_session(client_ip="one")
        manager = RunManager(
            web_settings,
            lambda: FakeModelClient([model_response("Should not be reached.")]),
        )
        record = manager.start(session, task="Cancel me.", client_ip="one")
        manager.cancel(record.run_id, session.session_id)
        assert record.background_task is not None
        await record.background_task
        assert record.cancel_requested is True
        assert record.result is not None
        assert record.result.status is AgentRunStatus.CANCELLED
        sessions.shutdown()

    asyncio.run(exercise())


def test_mutation_affects_only_current_session(
    web_settings: WebSettings,
    seed_workspace: Path,
) -> None:
    async def exercise() -> None:
        sessions = SessionManager(web_settings)
        sessions.start()
        first = sessions.create_session(client_ip="one")
        second = sessions.create_session(client_ip="two")
        responses = [
            model_response(
                calls=[
                    tool_call(
                        "write_file",
                        {"path": "generated.txt", "content": "safe output\n"},
                        "write",
                    )
                ]
            ),
            model_response(calls=[tool_call("read_file", {"path": "generated.txt"}, "read")]),
            model_response("Output verified."),
        ]
        manager = RunManager(web_settings, lambda: FakeModelClient(responses))
        record = manager.start(first, task="Create the requested output.", client_ip="one")
        assert record.background_task is not None
        await record.background_task

        assert (first.workspace_path / "generated.txt").is_file()
        assert not (second.workspace_path / "generated.txt").exists()
        assert not (seed_workspace / "generated.txt").exists()
        assert record.public_result()["changed_mutations"] == 1
        sessions.shutdown()

    asyncio.run(exercise())


def test_shutdown_stops_admission_and_does_not_abandon_background_task(
    web_settings: WebSettings,
) -> None:
    async def exercise() -> None:
        release = asyncio.Event()
        settings = web_settings.model_copy(update={"shutdown_grace_seconds": 0.01})
        sessions = SessionManager(settings)
        sessions.start()
        session = sessions.create_session(client_ip="one")
        model = BlockingModel(release)
        manager = RunManager(settings, lambda: model)
        record = manager.start(session, task="Wait for shutdown.", client_ip="one")
        assert record.background_task is not None
        await asyncio.wait_for(model.started.wait(), timeout=1)

        await manager.shutdown()

        assert record.background_task.done()
        assert record.status == "cancelled"
        assert record.reason_code == "SERVER_SHUTDOWN"
        assert record.cancel_event.is_set()
        assert session.active_run_id is None
        with pytest.raises(RunShuttingDownError):
            manager.start(session, task="Too late.", client_ip="one")
        sessions.shutdown()

    asyncio.run(exercise())
