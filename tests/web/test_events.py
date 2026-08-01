from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from app.events import AgentEvent, AgentEventType
from app.web.events import EventBacklog, WebEventSink


def test_backlog_assigns_ids_replays_and_finishes() -> None:
    async def exercise() -> None:
        backlog = EventBacklog(100)
        first = await backlog.append("run_started", datetime.now(UTC), {"value": 1})
        second = await backlog.append("tool_completed", datetime.now(UTC), {"value": 2})
        final = await backlog.finish(datetime.now(UTC), {"status": "completed"})

        assert [first.event_id, second.event_id, final.event_id] == [1, 2, 3]
        assert [event.event_id for event in backlog.after(1)] == [2, 3]
        assert backlog.finished is True

    asyncio.run(exercise())


def test_web_sink_removes_write_preview_and_hidden_fields() -> None:
    async def exercise() -> None:
        backlog = EventBacklog(100)
        sink = WebEventSink(backlog)
        secret_content = "complete secret output"
        await sink.emit(
            AgentEvent(
                run_id="run",
                type=AgentEventType.TOOL_COMPLETED,
                timestamp=datetime.now(UTC),
                payload={
                    "trace_event": {
                        "step": 1,
                        "tool": "write_file",
                        "args": {
                            "path": "output.txt",
                            "content": {
                                "characters": len(secret_content),
                                "sha256": "0" * 64,
                                "preview": secret_content,
                            },
                        },
                        "ok": True,
                        "result_summary": "Wrote output.txt",
                        "duration_ms": 1,
                    }
                },
            )
        )
        event = backlog.after(0)[0]
        serialized = str(event.data)
        assert secret_content not in serialized
        assert "reasoning" not in serialized
        assert "system" not in serialized.casefold()
        assert event.data["args"]["content"]["characters"] == len(secret_content)

    asyncio.run(exercise())


def test_sink_failure_is_contained() -> None:
    class BrokenBacklog:
        async def append(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            raise RuntimeError("consumer failure")

    event = AgentEvent(
        run_id="run",
        type=AgentEventType.RUN_STARTED,
        timestamp=datetime.now(UTC),
        payload={"initial_message_count": 2},
    )
    assert asyncio.run(WebEventSink(BrokenBacklog()).emit(event)) is None  # type: ignore[arg-type]
