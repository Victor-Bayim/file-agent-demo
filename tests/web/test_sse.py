from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from app.web.events import EventBacklog
from app.web.sse import format_sse_event, stream_events


class ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


def test_sse_format_is_single_line_json() -> None:
    async def exercise() -> None:
        backlog = EventBacklog(100)
        event = await backlog.append("tool_completed", datetime.now(UTC), {"text": "a\nb"})
        rendered = format_sse_event(event)
        assert rendered.startswith("id: 1\nevent: tool_completed\ndata: {")
        assert "a\\nb" in rendered
        assert rendered.endswith("\n\n")

    asyncio.run(exercise())


def test_stream_replays_after_last_id_and_ends_after_finish() -> None:
    async def exercise() -> None:
        backlog = EventBacklog(100)
        await backlog.append("run_started", datetime.now(UTC), {})
        await backlog.append("tool_completed", datetime.now(UTC), {"step": 1})
        await backlog.finish(datetime.now(UTC), {"status": "completed"})
        chunks = [
            chunk
            async for chunk in stream_events(
                backlog,
                ConnectedRequest(),  # type: ignore[arg-type]
                last_event_id=1,
                keepalive_seconds=0.01,
            )
        ]
        rendered = "".join(chunks)
        assert "id: 1" not in rendered
        assert "id: 2" in rendered
        assert "id: 3" in rendered
        assert "run_finished" in rendered

    asyncio.run(exercise())


def test_stream_emits_keepalive_without_real_wait() -> None:
    async def exercise() -> None:
        backlog = EventBacklog(100)
        generator = stream_events(
            backlog,
            ConnectedRequest(),  # type: ignore[arg-type]
            keepalive_seconds=0.001,
        )
        assert await anext(generator) == ": keepalive\n\n"
        await generator.aclose()

    asyncio.run(exercise())
