"""Small deterministic helpers kept outside the core Agent control loop."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from app.events import AgentEvent, AgentEventSink, AgentEventType
from app.message_utils import serialize_tool_result_for_model, trace_arguments_from_json
from app.model_types import ModelFinishReason, ModelMessage, ModelResponse, ModelRole, ModelToolCall
from app.runtime import (
    AgentRunStatus,
    AgentTerminationCode,
    RunState,
    ToolError,
    ToolExecutionResult,
    TraceEvent,
    UsageStats,
)
from app.tools import ToolErrorCode, ToolRegistry, UnknownToolError
from app.trace import TraceWriter


def valid_tool_batch(calls: Sequence[ModelToolCall], registry: ToolRegistry) -> bool:
    mutating = 0
    unknown = 0
    for call in calls:
        try:
            mutating += int(registry.get(call.name).is_mutating)
        except UnknownToolError:
            unknown += 1
    if mutating == 0:
        return True
    return len(calls) == 1 and mutating == 1 and unknown == 0


async def emit_run_started(
    sink: AgentEventSink,
    state: RunState,
    initial_message_count: int,
) -> None:
    await emit_safely(
        sink,
        AgentEvent(
            run_id=state.run_id,
            type=AgentEventType.RUN_STARTED,
            timestamp=datetime.now(UTC),
            payload={"initial_message_count": initial_message_count},
        ),
    )


def terminal_from_model_response(
    response: ModelResponse,
) -> tuple[AgentRunStatus, AgentTerminationCode | None, str | None, str | None]:
    """Apply deterministic no-tool finish semantics without provider-specific logic."""
    reason = response.finish_reason
    if reason is ModelFinishReason.LENGTH:
        return (
            AgentRunStatus.INCOMPLETE,
            AgentTerminationCode.MODEL_OUTPUT_TRUNCATED,
            "The model reached its output limit before completing the response.",
            None,
        )
    if reason is ModelFinishReason.CONTENT_FILTER:
        return (
            AgentRunStatus.FAILED,
            AgentTerminationCode.MODEL_CONTENT_FILTERED,
            "The model response was blocked by the provider content filter.",
            None,
        )
    if reason is ModelFinishReason.PROVIDER_RESOURCE:
        return (
            AgentRunStatus.FAILED,
            AgentTerminationCode.MODEL_PROVIDER_RESOURCE,
            "The provider could not complete the response because resources were unavailable.",
            None,
        )
    if response.message.content and response.message.content.strip():
        return AgentRunStatus.COMPLETED, None, None, response.message.content
    return (
        AgentRunStatus.FAILED,
        AgentTerminationCode.EMPTY_MODEL_RESPONSE,
        "The model returned neither tool calls nor a final answer.",
        None,
    )


async def emit_model_completed(
    sink: AgentEventSink,
    state: RunState,
    response: ModelResponse,
    usage: UsageStats,
) -> None:
    """Emit only public provider metadata and deterministic usage details."""
    calls = response.message.tool_calls
    await emit_safely(
        sink,
        AgentEvent(
            run_id=state.run_id,
            type=AgentEventType.MODEL_COMPLETED,
            timestamp=datetime.now(UTC),
            payload={
                "model_calls": state.model_calls,
                "usage": usage.model_dump(mode="json"),
                "has_tool_calls": bool(calls),
                "tool_call_count": len(calls),
                "finish_reason": response.finish_reason.value,
                "raw_finish_reason": response.raw_finish_reason,
                "provider_request_id": response.provider_request_id,
                "provider_model": response.provider_model,
            },
        ),
    )


async def record_tool_call(
    *,
    call: ModelToolCall,
    result: ToolExecutionResult,
    duration_ms: float,
    state: RunState,
    trace_writer: TraceWriter,
    sink: AgentEventSink,
    max_result_chars: int,
) -> ModelMessage | None:
    event = TraceEvent(
        run_id=state.run_id,
        step=state.tool_calls,
        timestamp=datetime.now(UTC),
        tool=call.name,
        args=trace_arguments_from_json(call.name, call.arguments_json),
        ok=result.ok,
        result_summary=result.result_summary,
        duration_ms=duration_ms,
    )
    try:
        trace_writer.write(event)
    except Exception:  # noqa: BLE001 - trace implementations are isolated here
        return None
    await emit_safely(
        sink,
        AgentEvent(
            run_id=state.run_id,
            type=AgentEventType.TOOL_COMPLETED,
            timestamp=datetime.now(UTC),
            payload={"trace_event": event.model_dump(mode="json")},
        ),
    )
    return ModelMessage(
        role=ModelRole.TOOL,
        tool_call_id=call.id,
        content=serialize_tool_result_for_model(result, max_result_chars),
    )


async def emit_safely(sink: AgentEventSink, event: AgentEvent) -> None:
    try:
        await sink.emit(event)
    except Exception:  # noqa: BLE001 - observers cannot affect the Agent run
        return


def runtime_tool_error(
    code: ToolErrorCode,
    message: str,
    summary: str,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        ok=False,
        error=ToolError(code=code.value, message=message),
        trust="trusted_runtime_data",
        result_summary=summary,
    )
