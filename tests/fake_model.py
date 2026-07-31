"""Deterministic provider-neutral model test double."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from app.model_types import ModelClient, ModelMessage, ModelResponse


class FakeModelExhaustedError(RuntimeError):
    """Raised when no configured responses remain."""


@dataclass(frozen=True, slots=True)
class RecordedModelCall:
    messages: tuple[ModelMessage, ...]
    tools: tuple[dict[str, Any], ...]


class FakeModelClient:
    """Return configured responses in order and record every input call."""

    def __init__(
        self,
        responses: Sequence[ModelResponse],
        *,
        failures: Mapping[int, BaseException] | None = None,
    ) -> None:
        self._responses = deque(response.model_copy(deep=True) for response in responses)
        self._failures = dict(failures or {})
        if any(call_number < 1 for call_number in self._failures):
            raise ValueError("failure call numbers must start at 1")
        self.calls: list[RecordedModelCall] = []

    async def complete(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        call_number = len(self.calls) + 1
        self.calls.append(
            RecordedModelCall(
                messages=tuple(message.model_copy(deep=True) for message in messages),
                tools=tuple(deepcopy(tool) for tool in tools),
            )
        )
        if failure := self._failures.get(call_number):
            raise failure
        if not self._responses:
            raise FakeModelExhaustedError("Fake model responses are exhausted")
        return self._responses.popleft().model_copy(deep=True)


def implements_model_client(client: FakeModelClient) -> bool:
    """Expose a narrow runtime protocol assertion for tests."""
    return isinstance(client, ModelClient)
