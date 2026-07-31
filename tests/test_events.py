from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from app.events import AgentEvent, AgentEventType, NullEventSink, RecordingEventSink


def test_agent_event_serializes_public_contract_without_reasoning() -> None:
    event = AgentEvent(
        run_id="run-1",
        type=AgentEventType.MODEL_COMPLETED,
        timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        payload={"model_calls": 1, "has_tool_calls": True},
    )

    payload = event.model_dump(mode="json")

    assert payload["type"] == "model_completed"
    assert payload["timestamp"] == "2026-01-02T00:00:00Z"
    assert "reasoning" not in payload
    assert "chain_of_thought" not in payload


def test_recording_event_sink_retains_independent_copies() -> None:
    sink = RecordingEventSink()
    event = AgentEvent(
        run_id="run-1",
        type=AgentEventType.RUN_STARTED,
        timestamp=datetime.now(UTC),
        payload={"count": 1},
    )

    asyncio.run(sink.emit(event))
    event.payload["count"] = 2

    assert sink.events[0].payload == {"count": 1}


def test_null_event_sink_accepts_events() -> None:
    event = AgentEvent(
        run_id="run-1",
        type=AgentEventType.RUN_FINISHED,
        timestamp=datetime.now(UTC),
        payload={"status": "completed"},
    )

    assert asyncio.run(NullEventSink().emit(event)) is None
