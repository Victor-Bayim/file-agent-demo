"""Bounded, chain-of-thought-free Web event delivery."""

from __future__ import annotations

import asyncio
import copy
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.events import AgentEvent, AgentEventSink, AgentEventType


@dataclass(frozen=True)
class WebEvent:
    event_id: int
    event_type: str
    timestamp: datetime
    data: dict[str, Any]


class EventBacklog:
    """One run's bounded replay buffer and consumer notification primitive."""

    def __init__(self, limit: int) -> None:
        self._events: deque[WebEvent] = deque(maxlen=limit)
        self._next_id = 1
        self._finished = False
        self._condition = asyncio.Condition()

    @property
    def finished(self) -> bool:
        return self._finished

    async def append(
        self,
        event_type: str,
        timestamp: datetime,
        data: dict[str, Any],
    ) -> WebEvent:
        async with self._condition:
            event = WebEvent(
                event_id=self._next_id,
                event_type=event_type,
                timestamp=timestamp,
                data=dict(data),
            )
            self._next_id += 1
            self._events.append(event)
            self._condition.notify_all()
            return event

    async def finish(
        self,
        timestamp: datetime,
        data: dict[str, Any],
    ) -> WebEvent:
        event = await self.append("run_finished", timestamp, data)
        async with self._condition:
            self._finished = True
            self._condition.notify_all()
        return event

    def after(self, event_id: int) -> list[WebEvent]:
        return [event for event in self._events if event.event_id > event_id]

    async def wait_for_change(self, event_id: int, timeout: float) -> None:
        async with self._condition:
            await asyncio.wait_for(
                self._condition.wait_for(
                    lambda: (
                        self._finished or any(event.event_id > event_id for event in self._events)
                    )
                ),
                timeout=timeout,
            )


class WebEventSink(AgentEventSink):
    """Translate existing public Agent events into a safe Web backlog."""

    def __init__(self, backlog: EventBacklog) -> None:
        self.backlog = backlog
        self.run_finished_seen = False

    async def emit(self, event: AgentEvent) -> None:
        try:
            if event.type is AgentEventType.RUN_STARTED:
                data = {
                    "initial_message_count": event.payload.get("initial_message_count", 0),
                }
            elif event.type is AgentEventType.MODEL_COMPLETED:
                data = {
                    key: event.payload.get(key)
                    for key in (
                        "model_calls",
                        "usage",
                        "finish_reason",
                        "raw_finish_reason",
                        "provider_model",
                        "tool_call_count",
                        "has_tool_calls",
                    )
                }
            elif event.type is AgentEventType.TOOL_COMPLETED:
                trace = event.payload.get("trace_event", {})
                if not isinstance(trace, dict):
                    return
                args = copy.deepcopy(trace.get("args", {}))
                if trace.get("tool") == "write_file" and isinstance(args, dict):
                    content = args.get("content")
                    if isinstance(content, dict):
                        content.pop("preview", None)
                data = {
                    "step": trace.get("step"),
                    "tool": trace.get("tool"),
                    "args": args,
                    "result_summary": trace.get("result_summary"),
                    "ok": trace.get("ok"),
                    "error_code": None if trace.get("ok") else "TOOL_FAILED",
                    "duration_ms": trace.get("duration_ms"),
                }
            elif event.type is AgentEventType.RUN_FINISHED:
                self.run_finished_seen = True
                return
            else:
                return
            await self.backlog.append(event.type.value, event.timestamp, data)
        except Exception:  # noqa: BLE001 - observers must never affect Agent execution
            return
