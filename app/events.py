"""Public, chain-of-thought-free lifecycle events for Agent runs."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class AgentEventType(StrEnum):
    RUN_STARTED = "run_started"
    MODEL_COMPLETED = "model_completed"
    TOOL_COMPLETED = "tool_completed"
    RUN_FINISHED = "run_finished"


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    type: AgentEventType
    timestamp: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentEventSink(Protocol):
    async def emit(self, event: AgentEvent) -> None:
        """Consume one public event without receiving hidden model reasoning."""
        ...


class NullEventSink:
    async def emit(self, event: AgentEvent) -> None:
        del event


class RecordingEventSink:
    """In-memory event sink used by deterministic runtime tests."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event.model_copy(deep=True))
