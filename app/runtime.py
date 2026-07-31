"""Provider-independent runtime state and result contracts."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_sha256(value: str) -> str:
    if not SHA256_PATTERN.fullmatch(value):
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def _normalize_posix_relative_path(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError("path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or value != path.as_posix():
        raise ValueError("path must be a normalized POSIX relative path")
    if any(part in {"", ".", ".."} or ":" in part for part in path.parts):
        raise ValueError("path must not contain traversal, empty, or drive components")
    return value


class RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UsageStats(RuntimeModel):
    """Accumulated token usage with explicit availability semantics.

    ``available=False`` means no usage numbers were supplied.  When a provider
    reports only a total (or another partial breakdown), ``breakdown_available``
    is false: known component fields remain usable, but their zero values must
    not be interpreted as provider-reported exact zeros. ``total_available``
    means the aggregate total covers every accumulated model call, which is the
    only form safe for enforcing a total-token budget. The total equality
    invariant is enforced only for exact, complete breakdowns.
    """

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    exact: bool = True
    available: bool = True
    breakdown_available: bool = True
    total_available: bool = True

    @model_validator(mode="after")
    def validate_usage(self) -> Self:
        self.validate_total()
        if not self.available:
            if any((self.input_tokens, self.output_tokens, self.total_tokens)):
                raise ValueError("unavailable usage cannot contain token values")
            if self.exact:
                raise ValueError("unavailable usage cannot be marked exact")
            if self.breakdown_available:
                raise ValueError("unavailable usage cannot have a breakdown")
        return self

    def validate_total(self) -> None:
        """Validate the total when input/output breakdown values are complete."""
        if self.breakdown_available and self.exact:
            expected = self.input_tokens + self.output_tokens
            if self.total_tokens != expected:
                raise ValueError("total_tokens must equal input_tokens + output_tokens")

    def add(self, other: UsageStats) -> None:
        """Accumulate usage, preserving uncertainty and incomplete breakdowns."""
        updated = UsageStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            exact=self.exact and other.exact,
            available=self.available or other.available,
            breakdown_available=self.breakdown_available and other.breakdown_available,
            total_available=self.total_available and other.total_available,
        )
        if not self.available and self.total_available:
            updated = other.model_copy(deep=True)
        for field_name in type(self).model_fields:
            object.__setattr__(self, field_name, getattr(updated, field_name))


def _usage_accumulator() -> UsageStats:
    """Return an unavailable but neutral accumulator before the first model call."""
    return UsageStats(
        exact=False,
        available=False,
        breakdown_available=False,
        total_available=True,
    )


class ToolError(RuntimeModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("code", "message")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)


class ToolExecutionResult(RuntimeModel):
    ok: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    trust: Literal["trusted_runtime_data", "untrusted_workspace_data"]
    result_summary: str

    @field_validator("result_summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _non_empty(value, "result_summary")

    @model_validator(mode="after")
    def validate_result_state(self) -> Self:
        if self.ok and self.error is not None:
            raise ValueError("successful tool results must not contain an error")
        if not self.ok and self.error is None:
            raise ValueError("failed tool results must contain an error")
        return self


class TraceEvent(RuntimeModel):
    run_id: str
    step: int = Field(ge=1)
    timestamp: datetime
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    result_summary: str
    duration_ms: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("run_id", "tool", "result_summary")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)


class MutationRecord(RuntimeModel):
    step: int = Field(ge=1)
    operation: str
    source: str | None = None
    destination: str
    status: Literal["succeeded", "failed"]
    changed: bool
    before_sha256: str | None = None
    after_sha256: str | None = None
    error_code: str | None = None

    @field_validator("operation", "destination")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("before_sha256", "after_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return None if value is None else _normalize_sha256(value)

    @model_validator(mode="after")
    def validate_error_state(self) -> Self:
        if self.status == "failed" and not (self.error_code or "").strip():
            raise ValueError("failed mutations must contain an error_code")
        if self.status == "failed" and self.changed:
            raise ValueError("failed mutations cannot be marked changed")
        if self.status == "succeeded" and self.error_code is not None:
            raise ValueError("succeeded mutations must not contain an error_code")
        return self


class AgentRunStatus(StrEnum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentTerminationCode(StrEnum):
    MAX_MODEL_TURNS = "MAX_MODEL_TURNS"
    MAX_TOOL_CALLS = "MAX_TOOL_CALLS"
    MAX_RUNTIME = "MAX_RUNTIME"
    MAX_TOTAL_TOKENS = "MAX_TOTAL_TOKENS"
    REPEATED_TOOL_CALL_LIMIT = "REPEATED_TOOL_CALL_LIMIT"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_ERROR = "MODEL_ERROR"
    MODEL_OUTPUT_TRUNCATED = "MODEL_OUTPUT_TRUNCATED"
    MODEL_CONTENT_FILTERED = "MODEL_CONTENT_FILTERED"
    MODEL_PROVIDER_RESOURCE = "MODEL_PROVIDER_RESOURCE"
    EMPTY_MODEL_RESPONSE = "EMPTY_MODEL_RESPONSE"
    TRACE_WRITE_FAILED = "TRACE_WRITE_FAILED"
    CANCELLED = "CANCELLED"


class AgentRunResult(RuntimeModel):
    run_id: str
    status: AgentRunStatus
    trace_path: Path | None = None
    answer: str | None = None
    reason: str | None = None
    reason_code: str | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    finish_reason: str | None = None
    raw_finish_reason: str | None = None
    provider_model: str | None = None
    elapsed_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    mutations: list[MutationRecord] = Field(default_factory=list)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _non_empty(value, "run_id")

    @model_validator(mode="after")
    def validate_status_payload(self) -> Self:
        if self.status is AgentRunStatus.COMPLETED:
            if not (self.answer or "").strip():
                raise ValueError("completed results must contain an answer")
            if self.reason is not None or self.reason_code is not None:
                raise ValueError("completed results cannot contain a termination reason")
        elif not (self.reason or "").strip() or not (self.reason_code or "").strip():
            raise ValueError(f"{self.status.value} results must contain reason and reason_code")
        return self

    @property
    def changed_mutations(self) -> list[MutationRecord]:
        return [mutation for mutation in self.mutations if mutation.changed]

    @property
    def failed_mutations(self) -> list[MutationRecord]:
        return [mutation for mutation in self.mutations if mutation.status == "failed"]


class ObservedFile(RuntimeModel):
    path: str
    sha256: str
    observed_at_step: int = Field(gt=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _normalize_posix_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _normalize_sha256(value)


class RunState(RuntimeModel):
    """In-memory bookkeeping only; methods never access the filesystem."""

    run_id: str
    workspace_root: Path
    usage: UsageStats = Field(default_factory=_usage_accumulator)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    started_at: datetime
    observed_files: dict[str, ObservedFile] = Field(default_factory=dict)
    mutations: list[MutationRecord] = Field(default_factory=list)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _non_empty(value, "run_id")

    @model_validator(mode="after")
    def validate_observation_keys(self) -> Self:
        for path, observation in self.observed_files.items():
            normalized = _normalize_posix_relative_path(path)
            if normalized != observation.path:
                raise ValueError("observed_files keys must match observation paths")
        return self

    def observe_file(
        self,
        path: str,
        sha256: str,
        observed_at_step: int,
    ) -> ObservedFile:
        observation = ObservedFile(
            path=path,
            sha256=sha256,
            observed_at_step=observed_at_step,
        )
        self.observed_files[observation.path] = observation
        return observation

    def get_observation(self, path: str) -> ObservedFile | None:
        normalized = _normalize_posix_relative_path(path)
        return self.observed_files.get(normalized)

    def remove_observation(self, path: str) -> ObservedFile | None:
        """Remove an observation after a file is moved or otherwise invalidated."""
        normalized = _normalize_posix_relative_path(path)
        return self.observed_files.pop(normalized, None)

    def record_mutation(self, mutation: MutationRecord) -> None:
        self.mutations.append(mutation)

    def increment_model_calls(self) -> int:
        self.model_calls += 1
        return self.model_calls

    def increment_tool_calls(self) -> int:
        self.tool_calls += 1
        return self.tool_calls
