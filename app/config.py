"""Runtime configuration contracts with explicit environment loading."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)


class ConfigurationError(ValueError):
    """Raised when environment-backed runtime configuration is invalid."""


class AgentLimits(BaseModel):
    """Deterministic resource ceilings enforced by the handwritten Agent loop."""

    model_config = ConfigDict(extra="forbid")

    max_model_turns: int = Field(default=20, gt=0)
    max_tool_calls: int = Field(default=50, gt=0)
    max_runtime_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    max_identical_calls: int = Field(default=3, gt=0)
    max_write_bytes: int = Field(default=100_000, gt=0)
    max_tool_result_chars: int = Field(default=16_000, gt=0)
    max_total_tokens: int | None = Field(default=100_000, gt=0)
    max_tool_history_chars: int = Field(default=60_000, gt=0)

    @model_validator(mode="after")
    def validate_call_limits(self) -> Self:
        if self.max_identical_calls > self.max_tool_calls:
            raise ValueError("max_identical_calls must not exceed max_tool_calls")
        return self


class DeepSeekConfig(BaseModel):
    """DeepSeek transport settings without network or filesystem side effects."""

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    thinking: Literal["disabled"] = "disabled"
    temperature: float = Field(default=0.1, ge=0, le=2, allow_inf_nan=False)
    max_output_tokens: int = Field(default=4096, gt=0, le=65_536)
    timeout_seconds: float = Field(default=45.0, gt=0, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0, le=5)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("api_key must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise ValueError("base_url must be an HTTPS URL")
        return normalized

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model must not be empty")
        if normalized in {"deepseek-chat", "deepseek-reasoner"}:
            raise ValueError("model uses a deprecated identifier")
        return normalized

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> DeepSeekConfig:
        """Load only documented DeepSeek variables; a missing key remains explicit."""
        source = os.environ if environ is None else environ
        variables = {
            "DEEPSEEK_API_KEY": "api_key",
            "DEEPSEEK_BASE_URL": "base_url",
            "DEEPSEEK_MODEL": "model",
            "DEEPSEEK_THINKING": "thinking",
            "DEEPSEEK_TEMPERATURE": "temperature",
            "DEEPSEEK_MAX_OUTPUT_TOKENS": "max_output_tokens",
            "DEEPSEEK_TIMEOUT_SECONDS": "timeout_seconds",
            "DEEPSEEK_MAX_RETRIES": "max_retries",
        }
        payload = {
            field_name: source[environment_name]
            for environment_name, field_name in variables.items()
            if environment_name in source
        }
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors(include_input=False)
            )
            raise ConfigurationError(f"Invalid DEEPSEEK configuration: {details}") from None


class RuntimeConfig(BaseModel):
    """Process configuration without filesystem or secret side effects."""

    model_config = ConfigDict(extra="forbid")

    runs_dir: Path = Path("runs")
    default_trace_filename: str = "trace.jsonl"
    limits: AgentLimits = Field(default_factory=AgentLimits)
    log_level: str = "INFO"

    @field_validator("default_trace_filename")
    @classmethod
    def validate_trace_filename(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("default_trace_filename must not be empty")
        if "/" in normalized or "\\" in normalized or Path(normalized).name != normalized:
            raise ValueError("default_trace_filename must be a filename, not a path")
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"log_level must be one of: {choices}")
        return normalized

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeConfig:
        """Load only documented, non-secret settings from an environment mapping."""
        source = os.environ if environ is None else environ
        payload: dict[str, object] = {}
        limit_values: dict[str, str] = {}

        direct_variables = {
            "FILE_AGENT_RUNS_DIR": "runs_dir",
            "FILE_AGENT_LOG_LEVEL": "log_level",
        }
        limit_variables = {
            "FILE_AGENT_MAX_MODEL_TURNS": "max_model_turns",
            "FILE_AGENT_MAX_TOOL_CALLS": "max_tool_calls",
            "FILE_AGENT_MAX_RUNTIME_SECONDS": "max_runtime_seconds",
            "FILE_AGENT_MAX_IDENTICAL_CALLS": "max_identical_calls",
            "FILE_AGENT_MAX_WRITE_BYTES": "max_write_bytes",
            "FILE_AGENT_MAX_TOOL_RESULT_CHARS": "max_tool_result_chars",
            "FILE_AGENT_MAX_TOTAL_TOKENS": "max_total_tokens",
            "FILE_AGENT_MAX_TOOL_HISTORY_CHARS": "max_tool_history_chars",
        }

        for environment_name, field_name in direct_variables.items():
            if environment_name in source:
                payload[field_name] = source[environment_name]
        for environment_name, field_name in limit_variables.items():
            if environment_name in source:
                raw_value = source[environment_name]
                disables_token_budget = (
                    environment_name == "FILE_AGENT_MAX_TOTAL_TOKENS"
                    and raw_value.strip().lower()
                    in {
                        "disabled",
                        "none",
                        "null",
                        "off",
                        "unlimited",
                    }
                )
                if disables_token_budget:
                    payload.setdefault("limits", {})
                    assert isinstance(payload["limits"], dict)
                    payload["limits"][field_name] = None
                else:
                    limit_values[field_name] = raw_value
        if limit_values:
            existing_limits = payload.setdefault("limits", {})
            assert isinstance(existing_limits, dict)
            existing_limits.update(limit_values)

        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid FILE_AGENT configuration: {exc}") from exc
