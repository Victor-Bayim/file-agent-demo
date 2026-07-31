"""Normalized model messages and a provider-neutral client protocol."""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.runtime import UsageStats


class ModelContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ModelFinishReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    PROVIDER_RESOURCE = "provider_resource"
    UNKNOWN = "unknown"


class ModelToolCall(ModelContract):
    id: str
    name: str
    arguments_json: str

    @field_validator("id", "name")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @classmethod
    def from_arguments(
        cls,
        *,
        id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> ModelToolCall:
        """Build a call with stable, strict JSON serialization for test adapters."""
        arguments_json = json.dumps(
            arguments,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return cls(id=id, name=name, arguments_json=arguments_json)


class ModelMessage(ModelContract):
    role: ModelRole
    content: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    @field_validator("tool_call_id")
    @classmethod
    def validate_optional_call_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool_call_id must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_role_contract(self) -> Self:
        if self.role is ModelRole.TOOL and self.tool_call_id is None:
            raise ValueError("tool messages must contain tool_call_id")
        if self.role is not ModelRole.TOOL and self.tool_call_id is not None:
            raise ValueError("only tool messages may contain tool_call_id")
        if self.role is not ModelRole.ASSISTANT and self.tool_calls:
            raise ValueError("only assistant messages may contain tool_calls")
        return self


class ModelUsage(ModelContract):
    """Usage exactly as returned by a provider; missing values remain ``None``."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    exact: bool

    @model_validator(mode="after")
    def validate_complete_usage(self) -> Self:
        if self.exact and None not in (self.input_tokens, self.output_tokens, self.total_tokens):
            assert self.input_tokens is not None
            assert self.output_tokens is not None
            assert self.total_tokens is not None
            if self.total_tokens != self.input_tokens + self.output_tokens:
                raise ValueError("total_tokens must equal input_tokens + output_tokens")
        return self

    def to_usage_stats(self) -> UsageStats:
        """Convert without presenting missing provider values as exact zeros.

        A complete input/output pair permits a derived total when the provider
        omits it.  Partial components without a total are retained as an
        inexact lower bound with ``breakdown_available=False``.
        """
        supplied = (
            self.input_tokens is not None,
            self.output_tokens is not None,
            self.total_tokens is not None,
        )
        if not any(supplied):
            return UsageStats(
                exact=False,
                available=False,
                breakdown_available=False,
                total_available=False,
            )

        complete_breakdown = self.input_tokens is not None and self.output_tokens is not None
        known_input = self.input_tokens or 0
        known_output = self.output_tokens or 0
        total = self.total_tokens if self.total_tokens is not None else known_input + known_output
        total_available = self.total_tokens is not None or complete_breakdown

        return UsageStats(
            input_tokens=known_input,
            output_tokens=known_output,
            total_tokens=total,
            exact=self.exact if self.total_tokens is not None or complete_breakdown else False,
            available=True,
            breakdown_available=complete_breakdown,
            total_available=total_available,
        )


class ModelResponse(ModelContract):
    message: ModelMessage
    usage: ModelUsage
    finish_reason: ModelFinishReason
    raw_finish_reason: str | None = None
    provider_request_id: str | None = None
    provider_model: str | None = None


@runtime_checkable
class ModelClient(Protocol):
    async def complete(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        """Return one normalized completion response."""
        ...
