"""SSE formatting and replay for one authenticated run."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from starlette.requests import Request

from app.web.events import EventBacklog, WebEvent


def format_sse_event(event: WebEvent) -> str:
    payload = {
        "timestamp": event.timestamp.isoformat().replace("+00:00", "Z"),
        **event.data,
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.event_id}\nevent: {event.event_type}\ndata: {data}\n\n"


async def stream_events(
    backlog: EventBacklog,
    request: Request,
    *,
    last_event_id: int = 0,
    keepalive_seconds: float = 15.0,
) -> AsyncIterator[str]:
    """Replay, wait, keep alive, and stop after the final event is sent."""
    cursor = max(0, last_event_id)
    while True:
        events = backlog.after(cursor)
        for event in events:
            cursor = event.event_id
            yield format_sse_event(event)
        if backlog.finished and not backlog.after(cursor):
            return
        if await request.is_disconnected():
            return
        try:
            await backlog.wait_for_change(cursor, keepalive_seconds)
        except TimeoutError:
            if await request.is_disconnected():
                return
            yield ": keepalive\n\n"
        except asyncio.CancelledError:
            return
