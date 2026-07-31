"""A small, provider-neutral Agent loop owned entirely by this project."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from time import monotonic

from app.agent_loop_support import (
    emit_model_completed,
    emit_run_started,
    emit_safely,
    record_tool_call,
    runtime_tool_error,
    terminal_from_model_response,
    valid_tool_batch,
)
from app.config import AgentLimits
from app.events import AgentEvent, AgentEventSink, AgentEventType, NullEventSink
from app.message_utils import (
    canonical_tool_fingerprint,
    compact_tool_history,
    parse_tool_arguments,
)
from app.model_types import ModelClient, ModelMessage
from app.runtime import (
    AgentRunResult,
    AgentRunStatus,
    AgentTerminationCode,
    RunState,
)
from app.tools import ToolErrorCode, ToolRegistry
from app.trace import TraceWriter


async def run_agent(
    *,
    model: ModelClient,
    registry: ToolRegistry,
    state: RunState,
    messages: Sequence[ModelMessage],
    trace_writer: TraceWriter,
    limits: AgentLimits,
    event_sink: AgentEventSink | None = None,
    cancel_event: asyncio.Event | None = None,
) -> AgentRunResult:
    """Run the handwritten model/tool cycle; the caller owns ``trace_writer.close``."""
    sink = event_sink or NullEventSink()
    history = [message.model_copy(deep=True) for message in messages]
    started = monotonic()
    last_fingerprint: str | None = None
    identical_calls = 0
    last_response_metadata: tuple[str | None, str | None, str | None] = (None, None, None)
    await emit_run_started(sink, state, len(history))

    async def finish(
        status: AgentRunStatus,
        code: AgentTerminationCode | None,
        reason: str | None,
        answer: str | None = None,
    ) -> AgentRunResult:
        result = AgentRunResult(
            run_id=state.run_id,
            status=status,
            answer=answer,
            reason=reason,
            reason_code=code.value if code else None,
            usage=state.usage.model_copy(deep=True),
            model_calls=state.model_calls,
            tool_calls=state.tool_calls,
            finish_reason=last_response_metadata[0],
            raw_finish_reason=last_response_metadata[1],
            provider_model=last_response_metadata[2],
            elapsed_ms=(monotonic() - started) * 1000,
            mutations=[item.model_copy(deep=True) for item in state.mutations],
        )
        await emit_safely(
            sink,
            AgentEvent(
                run_id=state.run_id,
                type=AgentEventType.RUN_FINISHED,
                timestamp=datetime.now(UTC),
                payload={
                    "status": result.status.value,
                    "reason_code": result.reason_code,
                    "changed_mutations": len(result.changed_mutations),
                    "failed_mutations": len(result.failed_mutations),
                },
            ),
        )
        return result

    while True:
        if cancel_event is not None and cancel_event.is_set():
            return await finish(
                AgentRunStatus.CANCELLED,
                AgentTerminationCode.CANCELLED,
                "The run was cancelled before the next model call.",
            )
        elapsed = monotonic() - started
        if elapsed >= limits.max_runtime_seconds:
            return await finish(
                AgentRunStatus.INCOMPLETE,
                AgentTerminationCode.MAX_RUNTIME,
                "The maximum runtime was reached.",
            )
        if state.model_calls >= limits.max_model_turns:
            return await finish(
                AgentRunStatus.INCOMPLETE,
                AgentTerminationCode.MAX_MODEL_TURNS,
                "The maximum number of model turns was reached.",
            )

        request_messages = compact_tool_history(history, limits.max_tool_history_chars)
        state.increment_model_calls()
        remaining_seconds = limits.max_runtime_seconds - (monotonic() - started)
        try:
            response = await asyncio.wait_for(
                model.complete(request_messages, registry.model_schemas()),
                timeout=max(remaining_seconds, 0.000_001),
            )
        except TimeoutError:
            return await finish(
                AgentRunStatus.FAILED,
                AgentTerminationCode.MODEL_TIMEOUT,
                "The model call timed out.",
            )
        except Exception as exc:  # noqa: BLE001 - provider exceptions stop at this boundary
            return await finish(
                AgentRunStatus.FAILED,
                AgentTerminationCode.MODEL_ERROR,
                f"The model call failed safely ({type(exc).__name__}).",
            )

        response_usage = response.usage.to_usage_stats()
        last_response_metadata = (
            response.finish_reason.value,
            response.raw_finish_reason,
            response.provider_model,
        )
        state.usage.add(response_usage)
        assistant_message = response.message.model_copy(deep=True)
        history.append(assistant_message)
        await emit_model_completed(sink, state, response, response_usage)
        if (
            limits.max_total_tokens is not None
            and state.usage.total_available
            and state.usage.total_tokens > limits.max_total_tokens
        ):
            return await finish(
                AgentRunStatus.INCOMPLETE,
                AgentTerminationCode.MAX_TOTAL_TOKENS,
                "The total token budget was exceeded.",
            )

        calls = assistant_message.tool_calls
        if not calls:
            return await finish(*terminal_from_model_response(response))
        if len(calls) > limits.max_tool_calls - state.tool_calls:
            return await finish(
                AgentRunStatus.INCOMPLETE,
                AgentTerminationCode.MAX_TOOL_CALLS,
                "The requested tool batch exceeds the remaining tool-call budget.",
            )

        batch_valid = valid_tool_batch(calls, registry)
        for call in calls:
            if cancel_event is not None and cancel_event.is_set():
                return await finish(
                    AgentRunStatus.CANCELLED,
                    AgentTerminationCode.CANCELLED,
                    "The run was cancelled before the next tool call.",
                )
            if monotonic() - started >= limits.max_runtime_seconds:
                return await finish(
                    AgentRunStatus.INCOMPLETE,
                    AgentTerminationCode.MAX_RUNTIME,
                    "The maximum runtime was reached before the next tool call.",
                )

            state.increment_tool_calls()
            tool_started = monotonic()
            fingerprint = canonical_tool_fingerprint(call.name, call.arguments_json)
            if fingerprint == last_fingerprint:
                identical_calls += 1
            else:
                last_fingerprint = fingerprint
                identical_calls = 1

            if not batch_valid:
                result = runtime_tool_error(
                    ToolErrorCode.INVALID_TOOL_BATCH,
                    "Tool batch mixes mutation calls or contains multiple mutations",
                    "Tool call rejected: invalid tool batch",
                )
            elif identical_calls >= limits.max_identical_calls:
                result = runtime_tool_error(
                    ToolErrorCode.REPEATED_TOOL_CALL_LIMIT,
                    "The consecutive identical tool-call limit was reached",
                    "Tool call rejected: repeated-call limit reached",
                )
            else:
                arguments, parse_error = parse_tool_arguments(call.arguments_json)
                result = parse_error or registry.execute(call.name, arguments)

            recorded = await record_tool_call(
                call=call,
                result=result,
                duration_ms=(monotonic() - tool_started) * 1000,
                state=state,
                trace_writer=trace_writer,
                sink=sink,
                max_result_chars=limits.max_tool_result_chars,
            )
            if recorded is None:
                return await finish(
                    AgentRunStatus.FAILED,
                    AgentTerminationCode.TRACE_WRITE_FAILED,
                    "Writing the tool trace failed.",
                )
            history.append(recorded)
            if batch_valid and identical_calls >= limits.max_identical_calls:
                return await finish(
                    AgentRunStatus.INCOMPLETE,
                    AgentTerminationCode.REPEATED_TOOL_CALL_LIMIT,
                    "The consecutive identical tool-call limit was reached.",
                )
