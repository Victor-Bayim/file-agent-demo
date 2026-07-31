from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from app.agent_loop import run_agent
from app.config import AgentLimits, DeepSeekConfig
from app.deepseek_client import DeepSeekClient, DeepSeekTimeoutError
from app.events import AgentEventType, RecordingEventSink
from app.model_types import ModelMessage, ModelRole
from app.runtime import AgentRunStatus, RunState, ToolExecutionResult, TraceEvent
from app.tools import ToolRegistry, ToolSpec


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


class MemoryTraceWriter:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def write(self, event: TraceEvent) -> None:
        self.events.append(event.model_copy(deep=True))

    def close(self) -> None:
        return


class SequentialCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def sdk_client(outcomes: list[Any]) -> Any:
    completions = SequentialCompletions(outcomes)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        completions=completions,
    )


def completion(
    *,
    content: str | None,
    finish_reason: str,
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 2,
    completion_tokens: int = 1,
    request_id: str = "integration-request",
) -> Any:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        model="integration-model",
    )
    response._request_id = request_id
    return response


def tool_call(arguments: str = '{"value":"hello"}') -> Any:
    return SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="echo", arguments=arguments),
    )


def echo_registry(observed: list[str] | None = None) -> ToolRegistry:
    registry = ToolRegistry()

    def handler(arguments: BaseModel) -> ToolExecutionResult:
        value = EchoArgs.model_validate(arguments).value
        if observed is not None:
            observed.append(value)
        return ToolExecutionResult(
            ok=True,
            data={"value": value},
            trust="trusted_runtime_data",
            result_summary="Echo completed",
        )

    registry.register(
        ToolSpec(
            name="echo",
            description="Return a supplied value.",
            args_model=EchoArgs,
            is_mutating=False,
        ),
        handler,
    )
    return registry


def run_with_sdk(
    tmp_path: Path,
    outcomes: list[Any],
    registry: ToolRegistry | None = None,
) -> tuple[Any, Any, MemoryTraceWriter, RecordingEventSink]:
    sdk = sdk_client(outcomes)
    trace = MemoryTraceWriter()
    sink = RecordingEventSink()
    state = RunState(
        run_id="deepseek-integration",
        workspace_root=tmp_path,
        started_at=datetime.now(UTC),
    )
    result = asyncio.run(
        run_agent(
            model=DeepSeekClient(DeepSeekConfig(), sdk),
            registry=registry or ToolRegistry(),
            state=state,
            messages=[ModelMessage(role=ModelRole.USER, content="Complete the task")],
            trace_writer=trace,
            limits=AgentLimits(),
            event_sink=sink,
        )
    )
    return result, sdk, trace, sink


def test_mock_deepseek_text_completion_reaches_agent_loop(tmp_path: Path) -> None:
    result, _, trace, sink = run_with_sdk(
        tmp_path,
        [completion(content="Final answer", finish_reason="stop")],
    )

    model_event = next(
        event for event in sink.events if event.type is AgentEventType.MODEL_COMPLETED
    )
    assert result.status is AgentRunStatus.COMPLETED
    assert result.answer == "Final answer"
    assert trace.events == []
    assert model_event.payload["raw_finish_reason"] == "stop"
    assert model_event.payload["provider_request_id"] == "integration-request"


def test_mock_deepseek_tool_result_is_returned_and_usage_accumulates(tmp_path: Path) -> None:
    observed: list[str] = []
    first = completion(
        content=None,
        finish_reason="tool_calls",
        tool_calls=[tool_call()],
        prompt_tokens=4,
        completion_tokens=2,
    )
    second = completion(
        content="Tool flow completed",
        finish_reason="stop",
        prompt_tokens=3,
        completion_tokens=1,
    )

    result, sdk, trace, _ = run_with_sdk(
        tmp_path,
        [first, second],
        echo_registry(observed),
    )

    second_request_messages = sdk.completions.requests[1]["messages"]
    assert result.status is AgentRunStatus.COMPLETED
    assert result.model_calls == 2
    assert result.tool_calls == 1
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 3
    assert result.usage.total_tokens == 10
    assert observed == ["hello"]
    assert len(trace.events) == 1
    assert second_request_messages[-1]["role"] == "tool"
    assert isinstance(second_request_messages[-1]["content"], str)
    assert "integration-request" not in str(second_request_messages)
    assert "integration-model" not in str(second_request_messages)


@pytest.mark.parametrize(
    ("raw_finish", "status", "reason_code"),
    [
        ("length", AgentRunStatus.INCOMPLETE, "MODEL_OUTPUT_TRUNCATED"),
        ("content_filter", AgentRunStatus.FAILED, "MODEL_CONTENT_FILTERED"),
        (
            "insufficient_system_resource",
            AgentRunStatus.FAILED,
            "MODEL_PROVIDER_RESOURCE",
        ),
    ],
)
def test_mock_deepseek_specific_finish_reasons_reach_agent_loop(
    raw_finish: str,
    status: AgentRunStatus,
    reason_code: str,
    tmp_path: Path,
) -> None:
    result, _, _, _ = run_with_sdk(
        tmp_path,
        [completion(content=None, finish_reason=raw_finish)],
    )

    assert result.status is status
    assert result.reason_code == reason_code


def test_mock_deepseek_timeout_maps_to_model_timeout(tmp_path: Path) -> None:
    result, _, _, _ = run_with_sdk(
        tmp_path,
        [DeepSeekTimeoutError(retryable=True)],
    )

    assert result.status is AgentRunStatus.FAILED
    assert result.reason_code == "MODEL_TIMEOUT"
    assert result.model_calls == 1
