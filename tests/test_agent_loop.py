from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

import app.agent_loop as agent_loop_module
from app.agent_loop import run_agent
from app.config import AgentLimits
from app.events import AgentEvent, AgentEventType, RecordingEventSink
from app.filesystem_tools import build_filesystem_registry
from app.model_types import (
    ModelFinishReason,
    ModelMessage,
    ModelResponse,
    ModelRole,
    ModelToolCall,
    ModelUsage,
)
from app.runtime import AgentRunResult, AgentRunStatus, RunState, ToolExecutionResult, TraceEvent
from app.sandbox import WorkspaceSandbox
from app.tools import ToolRegistry, ToolSpec
from tests.fake_model import FakeModelClient


class RecordingTraceWriter:
    def __init__(self, *, fail_on: int | None = None) -> None:
        self.events: list[TraceEvent] = []
        self.fail_on = fail_on
        self.closed = False

    def write(self, event: TraceEvent) -> None:
        if self.fail_on == len(self.events) + 1:
            raise OSError("trace destination failure")
        self.events.append(event.model_copy(deep=True))

    def close(self) -> None:
        self.closed = True


def call(name: str, arguments: dict[str, Any], call_id: str) -> ModelToolCall:
    return ModelToolCall.from_arguments(id=call_id, name=name, arguments=arguments)


def raw_call(name: str, arguments_json: str, call_id: str) -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments_json=arguments_json)


def response(
    *,
    content: str | None = None,
    calls: Sequence[ModelToolCall] = (),
    input_tokens: int | None = 2,
    output_tokens: int | None = 1,
    total_tokens: int | None = 3,
    exact: bool = True,
    finish_reason: ModelFinishReason | None = None,
    raw_finish_reason: str | None = None,
    provider_request_id: str | None = None,
    provider_model: str | None = None,
) -> ModelResponse:
    normalized_finish = finish_reason or (
        ModelFinishReason.TOOL_CALLS if calls else ModelFinishReason.STOP
    )
    return ModelResponse(
        message=ModelMessage(
            role=ModelRole.ASSISTANT,
            content=content,
            tool_calls=list(calls),
        ),
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            exact=exact,
        ),
        finish_reason=normalized_finish,
        raw_finish_reason=raw_finish_reason,
        provider_request_id=provider_request_id,
        provider_model=provider_model,
    )


def make_state(root: Path, run_id: str = "agent-test") -> RunState:
    return RunState(
        run_id=run_id,
        workspace_root=root,
        started_at=datetime.now(UTC),
    )


def run(
    *,
    model: Any,
    registry: ToolRegistry,
    state: RunState,
    trace: RecordingTraceWriter | None = None,
    limits: AgentLimits | None = None,
    sink: Any = None,
    cancel_event: asyncio.Event | None = None,
    messages: Sequence[ModelMessage] | None = None,
) -> tuple[AgentRunResult, RecordingTraceWriter]:
    writer = trace or RecordingTraceWriter()
    result = asyncio.run(
        run_agent(
            model=model,
            registry=registry,
            state=state,
            messages=messages or [ModelMessage(role=ModelRole.USER, content="Perform the task")],
            trace_writer=writer,
            limits=limits or AgentLimits(),
            event_sink=sink,
            cancel_event=cancel_event,
        )
    )
    return result, writer


def filesystem_runtime(root: Path) -> tuple[ToolRegistry, RunState]:
    state = make_state(root)
    return build_filesystem_registry(WorkspaceSandbox(root), state, AgentLimits()), state


def tool_payload(message: ModelMessage) -> dict[str, Any]:
    assert message.role is ModelRole.TOOL
    assert message.content is not None
    return json.loads(message.content)


def test_readonly_tool_result_is_filled_back_before_final_completion(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "file.txt").write_text("content", encoding="utf-8")
    registry, state = filesystem_runtime(root)
    model = FakeModelClient(
        [
            response(calls=[call("list_directory", {"path": "."}, "list-1")]),
            response(content="Task complete."),
        ]
    )
    trace = RecordingTraceWriter()
    sink = RecordingEventSink()

    result, _ = run(model=model, registry=registry, state=state, trace=trace, sink=sink)

    assert result.status is AgentRunStatus.COMPLETED
    assert result.answer == "Task complete."
    assert result.model_calls == 2
    assert result.tool_calls == 1
    assert result.usage.total_tokens == 6
    returned = model.calls[1].messages[-1]
    assert returned.role is ModelRole.TOOL
    assert returned.tool_call_id == "list-1"
    assert tool_payload(returned)["ok"] is True
    assert [event.step for event in trace.events] == [1]
    assert [event.type for event in sink.events] == [
        AgentEventType.RUN_STARTED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.TOOL_COMPLETED,
        AgentEventType.MODEL_COMPLETED,
        AgentEventType.RUN_FINISHED,
    ]
    model_event = sink.events[1]
    assert "content" not in model_event.payload
    assert "reasoning" not in model_event.payload
    assert sink.events[-1].payload["changed_mutations"] == 0
    assert len([event for event in sink.events if event.type is AgentEventType.RUN_FINISHED]) == 1
    assert trace.closed is False


def test_multiple_readonly_calls_execute_in_one_model_turn(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b.txt").write_text("beta", encoding="utf-8")
    registry, state = filesystem_runtime(root)
    model = FakeModelClient(
        [
            response(
                calls=[
                    call("read_file", {"path": "a.txt"}, "read-a"),
                    call("read_file", {"path": "b.txt"}, "read-b"),
                ]
            ),
            response(content="Both files read."),
        ]
    )

    result, trace = run(model=model, registry=registry, state=state)

    assert result.status is AgentRunStatus.COMPLETED
    assert result.tool_calls == 2
    assert [event.step for event in trace.events] == [1, 2]
    tool_messages = model.calls[1].messages[-2:]
    assert [message.tool_call_id for message in tool_messages] == ["read-a", "read-b"]
    assert all(tool_payload(message)["ok"] for message in tool_messages)


def test_json_argument_errors_unknown_tool_and_extra_fields_are_recoverable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry, state = filesystem_runtime(root)
    model = FakeModelClient(
        [
            response(calls=[raw_call("list_directory", "{broken", "broken")]),
            response(calls=[raw_call("list_directory", "[]", "array")]),
            response(calls=[raw_call("list_directory", '"text"', "string")]),
            response(calls=[call("list_directory", {"extra": True}, "extra")]),
            response(calls=[call("invented_tool", {}, "unknown")]),
            response(calls=[call("list_directory", {}, "corrected")]),
            response(content="Recovered after structured errors."),
        ]
    )

    result, trace = run(model=model, registry=registry, state=state)
    returned_codes = [
        tool_payload(model.calls[index].messages[-1])["error"]["code"] for index in range(1, 6)
    ]

    assert result.status is AgentRunStatus.COMPLETED
    assert returned_codes == [
        "INVALID_TOOL_CALL_JSON",
        "INVALID_ARGUMENTS",
        "INVALID_ARGUMENTS",
        "INVALID_ARGUMENTS",
        "UNKNOWN_TOOL",
    ]
    assert tool_payload(model.calls[6].messages[-1])["ok"] is True
    assert [event.step for event in trace.events] == [1, 2, 3, 4, 5, 6]
    assert len(trace.events[0].args["raw_arguments_json"]) <= 256


@pytest.mark.parametrize("batch_kind", ["two_mutations", "mixed", "unknown_with_mutation"])
def test_invalid_batches_execute_nothing_then_model_can_correct(
    tmp_path: Path,
    batch_kind: str,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry, state = filesystem_runtime(root)
    if batch_kind == "two_mutations":
        invalid_calls = [
            call("create_directory", {"path": "first"}, "first"),
            call("create_directory", {"path": "second"}, "second"),
        ]
    elif batch_kind == "mixed":
        invalid_calls = [
            call("list_directory", {}, "list"),
            call("create_directory", {"path": "first"}, "first"),
        ]
    else:
        invalid_calls = [
            call("invented_tool", {}, "unknown"),
            call("create_directory", {"path": "first"}, "first"),
        ]
    model = FakeModelClient(
        [
            response(calls=invalid_calls),
            response(calls=[call("create_directory", {"path": "corrected"}, "corrected")]),
            response(content="Corrected batch completed."),
        ]
    )

    result, trace = run(model=model, registry=registry, state=state)
    batch_messages = model.calls[1].messages[-2:]

    assert result.status is AgentRunStatus.COMPLETED
    assert [tool_payload(message)["error"]["code"] for message in batch_messages] == [
        "INVALID_TOOL_BATCH",
        "INVALID_TOOL_BATCH",
    ]
    assert [message.tool_call_id for message in batch_messages] == [
        invalid_calls[0].id,
        invalid_calls[1].id,
    ]
    assert not (root / "first").exists()
    assert not (root / "second").exists()
    assert (root / "corrected").is_dir()
    assert len(state.mutations) == 1
    assert state.mutations[0].changed is True
    assert [event.step for event in trace.events] == [1, 2, 3]


def test_filesystem_errors_are_returned_and_model_can_recover(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "source.txt").write_text("source", encoding="utf-8")
    (root / "target.txt").write_text("target", encoding="utf-8")
    registry, state = filesystem_runtime(root)
    model = FakeModelClient(
        [
            response(calls=[call("read_file", {"path": "missing.txt"}, "missing")]),
            response(
                calls=[
                    call(
                        "move_file",
                        {"source": "source.txt", "destination": "moved.txt"},
                        "unobserved",
                    )
                ]
            ),
            response(
                calls=[
                    call(
                        "write_file",
                        {"path": "target.txt", "content": "new"},
                        "existing",
                    )
                ]
            ),
            response(calls=[call("read_file", {"path": "source.txt"}, "observe")]),
            response(
                calls=[
                    call(
                        "move_file",
                        {"source": "source.txt", "destination": "moved.txt"},
                        "move",
                    )
                ]
            ),
            response(content="Recovered."),
        ]
    )

    result, _ = run(model=model, registry=registry, state=state)

    assert result.status is AgentRunStatus.COMPLETED
    assert [record.status for record in state.mutations] == ["failed", "failed", "succeeded"]
    assert [record.error_code for record in state.mutations[:2]] == [
        "SOURCE_NOT_OBSERVED",
        "TARGET_ALREADY_EXISTS",
    ]
    assert (root / "moved.txt").read_text(encoding="utf-8") == "source"


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: int = Field(description="First value.")
    b: int = Field(description="Second value.")


def echo_registry(counter: list[dict[str, int]], *, large: bool = False) -> ToolRegistry:
    registry = ToolRegistry()

    def handler(arguments: BaseModel) -> ToolExecutionResult:
        parsed = EchoArgs.model_validate(arguments)
        counter.append(parsed.model_dump())
        return ToolExecutionResult(
            ok=True,
            data={"value": "x" * 250 if large else parsed.model_dump()},
            trust="trusted_runtime_data",
            result_summary=f"Echoed {parsed.a} and {parsed.b}",
        )

    registry.register(
        ToolSpec(
            name="echo",
            description="Echo two deterministic integer values.",
            args_model=EchoArgs,
            is_mutating=False,
        ),
        handler,
    )
    return registry


def test_repeated_calls_use_canonical_json_and_skip_triggering_call(tmp_path: Path) -> None:
    counter: list[dict[str, int]] = []
    registry = echo_registry(counter)
    state = make_state(tmp_path)
    model = FakeModelClient(
        [
            response(calls=[raw_call("echo", '{"a":1,"b":2}', "one")]),
            response(calls=[raw_call("echo", '{"b":2,"a":1}', "two")]),
        ]
    )

    result, trace = run(
        model=model,
        registry=registry,
        state=state,
        limits=AgentLimits(max_identical_calls=2),
    )

    assert result.status is AgentRunStatus.INCOMPLETE
    assert result.reason_code == "REPEATED_TOOL_CALL_LIMIT"
    assert counter == [{"a": 1, "b": 2}]
    assert result.tool_calls == 2
    assert len(trace.events) == 2
    assert "repeated-call limit" in trace.events[-1].result_summary


def test_changed_arguments_reset_consecutive_repeat_detection(tmp_path: Path) -> None:
    counter: list[dict[str, int]] = []
    registry = echo_registry(counter)
    state = make_state(tmp_path)
    model = FakeModelClient(
        [
            response(calls=[call("echo", {"a": 1, "b": 1}, "one")]),
            response(calls=[call("echo", {"a": 2, "b": 1}, "two")]),
            response(calls=[call("echo", {"a": 1, "b": 1}, "three")]),
            response(content="Done."),
        ]
    )

    result, _ = run(
        model=model,
        registry=registry,
        state=state,
        limits=AgentLimits(max_identical_calls=2),
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert len(counter) == 3


def test_model_turn_and_whole_batch_tool_budgets(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "b.txt").write_text("b", encoding="utf-8")
    registry, state = filesystem_runtime(root)
    turns_model = FakeModelClient([response(calls=[call("list_directory", {}, "list")])])

    turns, turns_trace = run(
        model=turns_model,
        registry=registry,
        state=state,
        limits=AgentLimits(max_model_turns=1),
    )

    assert turns.reason_code == "MAX_MODEL_TURNS"
    assert turns.model_calls == 1
    assert turns.tool_calls == 1
    assert len(turns_trace.events) == 1

    registry, state = filesystem_runtime(root)
    tools_model = FakeModelClient(
        [
            response(
                calls=[
                    call("read_file", {"path": "a.txt"}, "a"),
                    call("read_file", {"path": "b.txt"}, "b"),
                ]
            )
        ]
    )
    tools, tools_trace = run(
        model=tools_model,
        registry=registry,
        state=state,
        limits=AgentLimits(max_tool_calls=1, max_identical_calls=1),
    )

    assert tools.reason_code == "MAX_TOOL_CALLS"
    assert tools.tool_calls == 0
    assert tools_trace.events == []
    assert state.observed_files == {}


def test_runtime_and_total_token_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolRegistry()
    state = make_state(tmp_path)
    times = iter([0.0, 2.0, 2.0])
    monkeypatch.setattr(agent_loop_module, "monotonic", lambda: next(times))

    runtime, _ = run(
        model=FakeModelClient([]),
        registry=registry,
        state=state,
        limits=AgentLimits(max_runtime_seconds=1),
    )

    assert runtime.reason_code == "MAX_RUNTIME"
    assert runtime.model_calls == 0

    monkeypatch.undo()
    state = make_state(tmp_path)
    tokens, _ = run(
        model=FakeModelClient(
            [
                response(
                    content="Would otherwise finish",
                    input_tokens=4,
                    output_tokens=2,
                    total_tokens=6,
                )
            ]
        ),
        registry=ToolRegistry(),
        state=state,
        limits=AgentLimits(max_total_tokens=5),
    )
    assert tokens.reason_code == "MAX_TOTAL_TOKENS"
    assert tokens.usage.total_tokens == 6


def test_unavailable_usage_does_not_trigger_token_budget(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    model = FakeModelClient(
        [
            response(
                content="Completed without usage data.",
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                exact=False,
            )
        ]
    )

    result, _ = run(
        model=model,
        registry=ToolRegistry(),
        state=state,
        limits=AgentLimits(max_total_tokens=1),
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert result.usage.available is False
    assert result.usage.total_available is False


class SlowModel:
    async def complete(self, messages: Any, tools: Any) -> ModelResponse:
        del messages, tools
        await asyncio.sleep(1)
        return response(content="too late")


def test_model_timeout_and_pre_call_cancellation(tmp_path: Path) -> None:
    timeout_state = make_state(tmp_path)
    timed_out, _ = run(
        model=SlowModel(),
        registry=ToolRegistry(),
        state=timeout_state,
        limits=AgentLimits(max_runtime_seconds=0.01),
    )
    assert timed_out.status is AgentRunStatus.FAILED
    assert timed_out.reason_code == "MODEL_TIMEOUT"
    assert timed_out.model_calls == 1

    cancel_event = asyncio.Event()
    cancel_event.set()
    cancelled, _ = run(
        model=FakeModelClient([]),
        registry=ToolRegistry(),
        state=make_state(tmp_path),
        cancel_event=cancel_event,
    )
    assert cancelled.status is AgentRunStatus.CANCELLED
    assert cancelled.reason_code == "CANCELLED"
    assert cancelled.model_calls == 0


def test_cancellation_is_checked_between_tool_calls(tmp_path: Path) -> None:
    cancel_event = asyncio.Event()
    counter: list[dict[str, int]] = []
    registry = echo_registry(counter)
    original_handler = registry._handlers["echo"]  # noqa: SLF001 - deliberate test hook

    def cancelling_handler(arguments: BaseModel) -> ToolExecutionResult:
        assert original_handler is not None
        result = original_handler(arguments)
        cancel_event.set()
        return result

    registry._handlers["echo"] = cancelling_handler  # noqa: SLF001 - deliberate test hook
    model = FakeModelClient(
        [
            response(
                calls=[
                    call("echo", {"a": 1, "b": 1}, "first"),
                    call("echo", {"a": 2, "b": 2}, "second"),
                ]
            )
        ]
    )

    result, trace = run(
        model=model,
        registry=registry,
        state=make_state(tmp_path),
        cancel_event=cancel_event,
    )

    assert result.status is AgentRunStatus.CANCELLED
    assert result.tool_calls == 1
    assert len(trace.events) == 1
    assert len(counter) == 1


@pytest.mark.parametrize("content", [None, "", "   \n"])
def test_empty_model_response_fails(content: str | None, tmp_path: Path) -> None:
    result, _ = run(
        model=FakeModelClient([response(content=content)]),
        registry=ToolRegistry(),
        state=make_state(tmp_path),
    )

    assert result.status is AgentRunStatus.FAILED
    assert result.reason_code == "EMPTY_MODEL_RESPONSE"


@pytest.mark.parametrize(
    ("finish_reason", "content", "status", "reason_code"),
    [
        (
            ModelFinishReason.LENGTH,
            "partial answer",
            AgentRunStatus.INCOMPLETE,
            "MODEL_OUTPUT_TRUNCATED",
        ),
        (
            ModelFinishReason.CONTENT_FILTER,
            "filtered placeholder",
            AgentRunStatus.FAILED,
            "MODEL_CONTENT_FILTERED",
        ),
        (
            ModelFinishReason.PROVIDER_RESOURCE,
            None,
            AgentRunStatus.FAILED,
            "MODEL_PROVIDER_RESOURCE",
        ),
    ],
)
def test_specific_finish_reasons_take_precedence_over_content_or_empty_response(
    finish_reason: ModelFinishReason,
    content: str | None,
    status: AgentRunStatus,
    reason_code: str,
    tmp_path: Path,
) -> None:
    result, _ = run(
        model=FakeModelClient(
            [response(content=content, finish_reason=finish_reason, raw_finish_reason="raw")]
        ),
        registry=ToolRegistry(),
        state=make_state(tmp_path),
    )

    assert result.status is status
    assert result.reason_code == reason_code
    assert result.answer is None


def test_unknown_finish_reason_with_content_completes_and_records_metadata(tmp_path: Path) -> None:
    sink = RecordingEventSink()
    result, _ = run(
        model=FakeModelClient(
            [
                response(
                    content="Provider supplied a usable final answer.",
                    finish_reason=ModelFinishReason.UNKNOWN,
                    raw_finish_reason="future_reason",
                    provider_request_id="safe-request-id",
                    provider_model="provider-model",
                )
            ]
        ),
        registry=ToolRegistry(),
        state=make_state(tmp_path),
        sink=sink,
    )

    model_event = next(
        event for event in sink.events if event.type is AgentEventType.MODEL_COMPLETED
    )
    assert result.status is AgentRunStatus.COMPLETED
    assert model_event.payload["finish_reason"] == "unknown"
    assert model_event.payload["raw_finish_reason"] == "future_reason"
    assert model_event.payload["provider_request_id"] == "safe-request-id"
    assert model_event.payload["provider_model"] == "provider-model"
    assert "content" not in model_event.payload


def test_unknown_finish_reason_without_content_remains_empty_response(tmp_path: Path) -> None:
    result, _ = run(
        model=FakeModelClient(
            [response(finish_reason=ModelFinishReason.UNKNOWN, raw_finish_reason="future_reason")]
        ),
        registry=ToolRegistry(),
        state=make_state(tmp_path),
    )

    assert result.reason_code == "EMPTY_MODEL_RESPONSE"


def test_tool_calls_take_precedence_over_conflicting_finish_reason(tmp_path: Path) -> None:
    counter: list[dict[str, int]] = []
    sink = RecordingEventSink()
    model = FakeModelClient(
        [
            response(
                calls=[call("echo", {"a": 1, "b": 2}, "call-1")],
                finish_reason=ModelFinishReason.CONTENT_FILTER,
                raw_finish_reason="content_filter",
            ),
            response(content="Tool result accepted."),
        ]
    )

    result, _ = run(
        model=model,
        registry=echo_registry(counter),
        state=make_state(tmp_path),
        sink=sink,
    )

    model_events = [event for event in sink.events if event.type is AgentEventType.MODEL_COMPLETED]
    assert result.status is AgentRunStatus.COMPLETED
    assert counter == [{"a": 1, "b": 2}]
    assert model_events[0].payload["finish_reason"] == "content_filter"


def test_model_error_reason_does_not_leak_exception_message(tmp_path: Path) -> None:
    sensitive = "api_key=secret-value Authorization=Bearer-secret C:/private/request.json"
    model = FakeModelClient([], failures={1: RuntimeError(sensitive)})

    result, _ = run(model=model, registry=ToolRegistry(), state=make_state(tmp_path))

    assert result.reason_code == "MODEL_ERROR"
    assert "RuntimeError" in (result.reason or "")
    assert sensitive not in (result.reason or "")
    assert "secret-value" not in (result.reason or "")


def test_trace_failure_stops_after_preserving_completed_mutation(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry, state = filesystem_runtime(root)
    model = FakeModelClient(
        [response(calls=[call("write_file", {"path": "created.txt", "content": "done"}, "w")])]
    )
    trace = RecordingTraceWriter(fail_on=1)

    result, _ = run(model=model, registry=registry, state=state, trace=trace)

    assert result.status is AgentRunStatus.FAILED
    assert result.reason_code == "TRACE_WRITE_FAILED"
    assert (root / "created.txt").read_text(encoding="utf-8") == "done"
    assert len(result.changed_mutations) == 1
    assert result.changed_mutations[0].operation == "write_file"
    assert result.tool_calls == 1
    assert len(model.calls) == 1


class ExplodingEventSink:
    async def emit(self, event: AgentEvent) -> None:
        raise RuntimeError(f"consumer failed for {event.type}")


def test_event_sink_failure_does_not_affect_agent(tmp_path: Path) -> None:
    result, _ = run(
        model=FakeModelClient([response(content="Completed despite consumer failure.")]),
        registry=ToolRegistry(),
        state=make_state(tmp_path),
        sink=ExplodingEventSink(),
    )

    assert result.status is AgentRunStatus.COMPLETED


def test_single_tool_result_and_tool_history_are_bounded(tmp_path: Path) -> None:
    counter: list[dict[str, int]] = []
    registry = echo_registry(counter, large=True)
    state = make_state(tmp_path)
    initial_messages = [ModelMessage(role=ModelRole.USER, content="Original task")]
    model = FakeModelClient(
        [
            response(calls=[call("echo", {"a": 1, "b": 1}, "first")]),
            response(calls=[call("echo", {"a": 2, "b": 2}, "second")]),
            response(content="Context remained valid."),
        ]
    )

    result, _ = run(
        model=model,
        registry=registry,
        state=state,
        messages=initial_messages,
        limits=AgentLimits(max_tool_result_chars=230, max_tool_history_chars=300),
    )

    first_tool = model.calls[1].messages[-1]
    third_call_tools = [
        message for message in model.calls[2].messages if message.role is ModelRole.TOOL
    ]
    assert result.status is AgentRunStatus.COMPLETED
    assert len(first_tool.content or "") <= 230
    assert json.loads(first_tool.content or "null")["result_truncated_for_model"] is True
    assert json.loads(third_call_tools[0].content or "null")["tool_result_compacted"] is True
    assert json.loads(third_call_tools[-1].content or "null")["result_truncated_for_model"] is True
    assert sum(len(message.content or "") for message in third_call_tools) <= 300
    assert initial_messages == [ModelMessage(role=ModelRole.USER, content="Original task")]
    assistant_ids = {
        tool_call.id
        for message in model.calls[2].messages
        if message.role is ModelRole.ASSISTANT
        for tool_call in message.tool_calls
    }
    assert {message.tool_call_id for message in third_call_tools} <= assistant_ids


def test_trace_arguments_are_sanitized_inside_agent_loop(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry, state = filesystem_runtime(root)
    content = "sensitive body " * 200
    secret = "Bearer full-secret-value"
    model = FakeModelClient(
        [
            response(
                calls=[
                    call(
                        "write_file",
                        {
                            "path": "output.txt",
                            "content": content,
                            "nested": {"Authorization": secret},
                        },
                        "write",
                    )
                ]
            ),
            response(content="Handled validation error."),
        ]
    )

    result, trace = run(model=model, registry=registry, state=state)
    serialized_args = json.dumps(trace.events[0].args, ensure_ascii=False)

    assert result.status is AgentRunStatus.COMPLETED
    assert content not in serialized_args
    assert secret not in serialized_args
    assert trace.events[0].args["content"]["characters"] == len(content)
    assert trace.events[0].args["nested"]["Authorization"] == "[REDACTED]"


def test_complete_fake_index_flow_is_driven_only_by_tool_calls(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    (root / "documents").mkdir(parents=True)
    (root / "documents" / "a.txt").write_text("exact phrase in alpha\n", encoding="utf-8")
    (root / "documents" / "b.txt").write_text("beta has exact phrase too\n", encoding="utf-8")
    registry, state = filesystem_runtime(root)
    index_content = "documents/a.txt\ndocuments/b.txt\n"
    model = FakeModelClient(
        [
            response(calls=[call("list_directory", {"path": ".", "recursive": True}, "list")]),
            response(calls=[call("search_text", {"query": "exact phrase"}, "search")]),
            response(
                calls=[
                    call("read_file", {"path": "documents/a.txt"}, "read-a"),
                    call("read_file", {"path": "documents/b.txt"}, "read-b"),
                ]
            ),
            response(
                calls=[
                    call(
                        "write_file",
                        {"path": "index.txt", "content": index_content},
                        "write-index",
                    )
                ]
            ),
            response(calls=[call("read_file", {"path": "index.txt"}, "verify-index")]),
            response(content="Index created and verified."),
        ]
    )
    sink = RecordingEventSink()

    result, trace = run(model=model, registry=registry, state=state, sink=sink)

    assert result.status is AgentRunStatus.COMPLETED
    assert result.answer == "Index created and verified."
    assert (root / "index.txt").read_text(encoding="utf-8") == index_content
    assert [event.tool for event in trace.events] == [
        "list_directory",
        "search_text",
        "read_file",
        "read_file",
        "write_file",
        "read_file",
    ]
    assert len(result.changed_mutations) == 1
    assert result.changed_mutations[0].operation == "write_file"
    assert sink.events[-1].payload["changed_mutations"] == 1


def test_complete_fake_move_flow_uses_the_same_agent_loop(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    for number in range(1, 4):
        (root / f"source-{number}.txt").write_text(f"content {number}\n", encoding="utf-8")
    registry, state = filesystem_runtime(root)
    read_calls = [
        call("read_file", {"path": f"source-{number}.txt"}, f"read-{number}")
        for number in range(1, 4)
    ]
    model = FakeModelClient(
        [
            response(calls=[call("list_directory", {}, "list")]),
            response(calls=read_calls),
            response(calls=[call("create_directory", {"path": "archive"}, "create")]),
            *[
                response(
                    calls=[
                        call(
                            "move_file",
                            {
                                "source": f"source-{number}.txt",
                                "destination": f"archive/source-{number}.txt",
                            },
                            f"move-{number}",
                        )
                    ]
                )
                for number in range(1, 4)
            ],
            response(
                calls=[
                    call(
                        "write_file",
                        {
                            "path": "manifest.txt",
                            "content": (
                                "archive/source-1.txt\narchive/source-2.txt\narchive/source-3.txt\n"
                            ),
                        },
                        "manifest",
                    )
                ]
            ),
            response(content="Files moved and manifest created."),
        ]
    )

    result, trace = run(model=model, registry=registry, state=state)

    assert result.status is AgentRunStatus.COMPLETED
    assert [event.tool for event in trace.events] == [
        "list_directory",
        "read_file",
        "read_file",
        "read_file",
        "create_directory",
        "move_file",
        "move_file",
        "move_file",
        "write_file",
    ]
    assert all((root / "archive" / f"source-{number}.txt").is_file() for number in range(1, 4))
    assert (root / "manifest.txt").is_file()
    assert len(result.changed_mutations) == 5
    assert all(mutation.changed for mutation in result.mutations)
