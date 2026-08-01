"""Side-effect-free configuration contracts for the local Web demo."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from app.sandbox import is_link_or_reparse_point


class WebConfigurationError(ValueError):
    """A safe Web configuration failure without secret input values."""


def _overlaps(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


class WebSettings(BaseModel):
    """Validated local Web settings with explicit resource ceilings."""

    model_config = ConfigDict(extra="forbid")

    seed_workspace: Path = Path("workspace")
    session_root: Path = Path("runtime/web-sessions")
    web_runs_root: Path = Path("runs/web")
    access_code: SecretStr
    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0, le=65535)
    cookie_secure: bool = False
    session_ttl_seconds: int = Field(default=7200, gt=0, le=604_800)
    max_sessions: int = Field(default=100, gt=0, le=10_000)
    max_task_chars: int = Field(default=10_000, gt=0, le=100_000)
    max_concurrent_runs: int = Field(default=2, gt=0, le=32)
    max_runs_per_session_hour: int = Field(default=5, gt=0, le=10_000)
    max_runs_per_ip_hour: int = Field(default=10, gt=0, le=100_000)
    web_max_model_turns: int = Field(default=24, gt=0, le=100)
    web_max_tool_calls: int = Field(default=60, gt=0, le=500)
    web_max_runtime_seconds: int = Field(default=240, gt=0, le=3600)
    web_max_total_tokens: int = Field(default=80_000, gt=0, le=1_000_000)
    event_backlog_limit: int = Field(default=256, ge=100, le=2000)
    sse_keepalive_seconds: float = Field(default=15.0, gt=0, le=60)

    @field_validator("access_code")
    @classmethod
    def validate_access_code(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("access_code must not be empty")
        return value

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError("host must be a non-empty address")
        return normalized

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        seed = self.seed_workspace.absolute()
        if not seed.exists() or not seed.is_dir():
            raise ValueError("seed_workspace must be an existing directory")
        if is_link_or_reparse_point(seed):
            raise ValueError("seed_workspace must not be a symbolic link")
        seed = seed.resolve(strict=True)

        raw_session_root = self.session_root.absolute()
        raw_runs_root = self.web_runs_root.absolute()
        for path, label in (
            (raw_session_root, "session_root"),
            (raw_runs_root, "web_runs_root"),
        ):
            if path.exists() and is_link_or_reparse_point(path):
                raise ValueError(f"{label} must not be a symbolic link")
        session_root = raw_session_root.resolve(strict=False)
        runs_root = raw_runs_root.resolve(strict=False)
        for path, label in (
            (session_root, "session_root"),
            (runs_root, "web_runs_root"),
        ):
            if _overlaps(path, seed):
                raise ValueError(f"{label} must not overlap seed_workspace")
        if _overlaps(session_root, runs_root):
            raise ValueError("session_root and web_runs_root must not overlap")

        self.seed_workspace = seed
        self.session_root = session_root
        self.web_runs_root = runs_root
        return self

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> WebSettings:
        """Read only documented Web variables from a supplied mapping."""
        source = os.environ if environ is None else environ
        variables = {
            "FILE_AGENT_WEB_SEED_WORKSPACE": "seed_workspace",
            "FILE_AGENT_WEB_SESSION_ROOT": "session_root",
            "FILE_AGENT_WEB_RUNS_ROOT": "web_runs_root",
            "FILE_AGENT_WEB_ACCESS_CODE": "access_code",
            "FILE_AGENT_WEB_HOST": "host",
            "FILE_AGENT_WEB_PORT": "port",
            "FILE_AGENT_WEB_COOKIE_SECURE": "cookie_secure",
            "FILE_AGENT_WEB_SESSION_TTL_SECONDS": "session_ttl_seconds",
            "FILE_AGENT_WEB_MAX_SESSIONS": "max_sessions",
            "FILE_AGENT_WEB_MAX_TASK_CHARS": "max_task_chars",
            "FILE_AGENT_WEB_MAX_CONCURRENT_RUNS": "max_concurrent_runs",
            "FILE_AGENT_WEB_MAX_RUNS_PER_SESSION_HOUR": "max_runs_per_session_hour",
            "FILE_AGENT_WEB_MAX_RUNS_PER_IP_HOUR": "max_runs_per_ip_hour",
            "FILE_AGENT_WEB_MAX_MODEL_TURNS": "web_max_model_turns",
            "FILE_AGENT_WEB_MAX_TOOL_CALLS": "web_max_tool_calls",
            "FILE_AGENT_WEB_MAX_RUNTIME_SECONDS": "web_max_runtime_seconds",
            "FILE_AGENT_WEB_MAX_TOTAL_TOKENS": "web_max_total_tokens",
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
            raise WebConfigurationError(f"Invalid Web configuration: {details}") from None
