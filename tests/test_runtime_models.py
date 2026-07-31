from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.runtime import (
    AgentRunResult,
    AgentRunStatus,
    MutationRecord,
    ObservedFile,
    RunState,
    ToolError,
    ToolExecutionResult,
    TraceEvent,
    UsageStats,
)


def test_usage_stats_accumulate_and_exact_uses_logical_and() -> None:
    usage = UsageStats(input_tokens=10, output_tokens=4, total_tokens=14, exact=True)
    other = UsageStats(input_tokens=3, output_tokens=2, total_tokens=5, exact=False)

    usage.add(other)

    assert usage == UsageStats(
        input_tokens=13,
        output_tokens=6,
        total_tokens=19,
        exact=False,
    )


def test_usage_stats_preserve_missing_breakdown_during_addition() -> None:
    usage = UsageStats(input_tokens=2, output_tokens=1, total_tokens=3)
    total_only = UsageStats(
        total_tokens=8,
        exact=True,
        available=True,
        breakdown_available=False,
    )

    usage.add(total_only)

    assert usage.total_tokens == 11
    assert usage.input_tokens == 2
    assert usage.output_tokens == 1
    assert usage.breakdown_available is False
    assert usage.exact is True


def test_usage_stats_preserve_missing_total_for_budget_decisions() -> None:
    usage = UsageStats(
        exact=False,
        available=False,
        breakdown_available=False,
        total_available=True,
    )
    unavailable = UsageStats(
        exact=False,
        available=False,
        breakdown_available=False,
        total_available=False,
    )
    known = UsageStats(input_tokens=2, output_tokens=1, total_tokens=3)

    usage.add(unavailable)
    usage.add(known)

    assert usage.available is True
    assert usage.total_tokens == 3
    assert usage.total_available is False
    assert usage.exact is False


def test_usage_stats_reject_inconsistent_complete_total() -> None:
    with pytest.raises(ValidationError, match="total_tokens must equal"):
        UsageStats(input_tokens=1, output_tokens=2, total_tokens=4)


def test_tool_error_details_are_not_shared() -> None:
    first = ToolError(code="FIRST", message="first")
    second = ToolError(code="SECOND", message="second")

    first.details["changed"] = True

    assert second.details == {}


def test_tool_execution_result_enforces_state_consistency() -> None:
    success = ToolExecutionResult(
        ok=True,
        data={"count": 1},
        trust="trusted_runtime_data",
        result_summary="One entry found",
    )
    assert success.error is None

    with pytest.raises(ValidationError, match="must not contain an error"):
        ToolExecutionResult(
            ok=True,
            error=ToolError(code="BAD", message="bad"),
            trust="trusted_runtime_data",
            result_summary="Invalid success",
        )
    with pytest.raises(ValidationError, match="must contain an error"):
        ToolExecutionResult(
            ok=False,
            trust="trusted_runtime_data",
            result_summary="Invalid failure",
        )
    with pytest.raises(ValidationError, match="must not be empty"):
        ToolExecutionResult(
            ok=True,
            trust="trusted_runtime_data",
            result_summary="   ",
        )


def test_trace_event_serializes_timestamp_as_iso_8601() -> None:
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    event = TraceEvent(
        run_id="run-1",
        step=1,
        timestamp=timestamp,
        tool="sample_tool",
        args={"path": "example.txt"},
        ok=True,
        result_summary="Completed",
        duration_ms=1.25,
    )

    payload = event.model_dump(mode="json")

    assert payload["timestamp"] == "2026-01-02T03:04:05Z"
    assert "reasoning" not in payload


@pytest.mark.parametrize(
    "payload",
    [
        {"status": AgentRunStatus.COMPLETED, "answer": None},
        {"status": AgentRunStatus.INCOMPLETE, "reason": None},
        {"status": AgentRunStatus.FAILED, "reason": "   "},
    ],
)
def test_agent_run_result_validates_status_payload(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AgentRunResult(run_id="run-1", **payload)


def test_agent_run_result_accepts_valid_completed_result() -> None:
    result = AgentRunResult(
        run_id="run-1",
        status=AgentRunStatus.COMPLETED,
        answer="Done",
        model_calls=2,
        tool_calls=3,
        elapsed_ms=25.0,
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert result.mutations == []


def test_noncompleted_agent_result_requires_reason_code() -> None:
    with pytest.raises(ValidationError, match="reason and reason_code"):
        AgentRunResult(
            run_id="run-1",
            status=AgentRunStatus.CANCELLED,
            reason="Cancelled by user",
        )


def test_agent_result_exposes_changed_and_failed_mutations() -> None:
    changed = MutationRecord(
        step=1,
        operation="write_file",
        destination="new.txt",
        status="succeeded",
        changed=True,
    )
    failed = MutationRecord(
        step=2,
        operation="move_file",
        source="new.txt",
        destination="other.txt",
        status="failed",
        changed=False,
        error_code="TARGET_ALREADY_EXISTS",
    )
    result = AgentRunResult(
        run_id="run-1",
        status=AgentRunStatus.INCOMPLETE,
        reason="Limit reached",
        reason_code="MAX_TOOL_CALLS",
        mutations=[changed, failed],
    )

    assert result.changed_mutations == [changed]
    assert result.failed_mutations == [failed]


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "/absolute.txt",
        "../escape.txt",
        "dir\\file.txt",
        "a//b.txt",
        "C:/drive.txt",
    ],
)
def test_observed_file_rejects_non_posix_relative_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        ObservedFile(path=path, sha256="a" * 64, observed_at_step=1)


@pytest.mark.parametrize("sha256", ["a" * 63, "g" * 64, "not-a-hash"])
def test_observed_file_rejects_invalid_sha256(sha256: str) -> None:
    with pytest.raises(ValidationError, match="64 hexadecimal"):
        ObservedFile(path="dir/file.txt", sha256=sha256, observed_at_step=1)


def test_run_state_tracks_observations_counts_and_mutations() -> None:
    state = RunState(
        run_id="run-1",
        workspace_root=Path("workspace-copy"),
        started_at=datetime.now(UTC),
    )
    observation = state.observe_file("dir/file.txt", "A" * 64, observed_at_step=2)
    mutation = MutationRecord(
        step=3,
        operation="write",
        destination="output.txt",
        status="succeeded",
        changed=True,
        before_sha256=None,
        after_sha256="b" * 64,
        error_code=None,
    )

    state.record_mutation(mutation)

    assert observation.sha256 == "a" * 64
    assert state.get_observation("dir/file.txt") == observation
    assert state.get_observation("other.txt") is None
    assert state.increment_model_calls() == 1
    assert state.increment_tool_calls() == 1
    assert state.mutations == [mutation]
