"""Provider-neutral tool contracts and deterministic execution binding."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from app.runtime import RunState, ToolError, ToolExecutionResult

TOOL_NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")


class ToolErrorCode(StrEnum):
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    INVALID_PATH = "INVALID_PATH"
    PATH_OUTSIDE_WORKSPACE = "PATH_OUTSIDE_WORKSPACE"
    PATH_NOT_FOUND = "PATH_NOT_FOUND"
    NOT_A_FILE = "NOT_A_FILE"
    NOT_A_DIRECTORY = "NOT_A_DIRECTORY"
    SYMLINK_NOT_ALLOWED = "SYMLINK_NOT_ALLOWED"
    BINARY_FILE_NOT_SUPPORTED = "BINARY_FILE_NOT_SUPPORTED"
    READ_LIMIT_EXCEEDED = "READ_LIMIT_EXCEEDED"
    WRITE_TOO_LARGE = "WRITE_TOO_LARGE"
    TARGET_ALREADY_EXISTS = "TARGET_ALREADY_EXISTS"
    PARENT_NOT_FOUND = "PARENT_NOT_FOUND"
    SOURCE_NOT_OBSERVED = "SOURCE_NOT_OBSERVED"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    SAME_SOURCE_AND_DESTINATION = "SAME_SOURCE_AND_DESTINATION"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    INTERNAL_TOOL_ERROR = "INTERNAL_TOOL_ERROR"
    INVALID_TOOL_BATCH = "INVALID_TOOL_BATCH"
    INVALID_TOOL_CALL_JSON = "INVALID_TOOL_CALL_JSON"
    REPEATED_TOOL_CALL_LIMIT = "REPEATED_TOOL_CALL_LIMIT"


class ToolRegistryError(RuntimeError):
    """Base class for deterministic registry failures."""


class DuplicateToolError(ToolRegistryError):
    """Raised when a tool name is registered more than once."""


class UnknownToolError(ToolRegistryError):
    """Raised when a requested tool has not been registered."""


class ToolHandlerError(RuntimeError):
    """Expected handler failure safe to return as structured data."""

    def __init__(
        self,
        code: ToolErrorCode | str,
        message: str,
        *,
        result_summary: str | None = None,
        details: dict[str, Any] | None = None,
        trust: str = "untrusted_workspace_data",
    ) -> None:
        super().__init__(message)
        self.code = code.value if isinstance(code, ToolErrorCode) else code
        self.safe_message = message
        self.result_summary = result_summary or message
        self.details = {} if details is None else details
        self.trust = trust


class ToolHandler(Protocol):
    def __call__(self, arguments: BaseModel) -> ToolExecutionResult:
        """Execute already-validated arguments."""
        ...


class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    name: str
    description: str
    args_model: type[BaseModel]
    is_mutating: bool

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not TOOL_NAME_PATTERN.fullmatch(value):
            raise ValueError(
                "tool name may contain only lowercase letters, digits, and underscores"
            )
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool description must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_args_model(self) -> Self:
        if not isinstance(self.args_model, type) or not issubclass(self.args_model, BaseModel):
            raise ValueError("args_model must be a Pydantic BaseModel subclass")
        return self

    def json_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    def model_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }


class ToolRegistry:
    """Insertion-ordered specs plus a small, exception-safe execution boundary."""

    def __init__(
        self,
        *,
        state: RunState | None = None,
        expose_internal_errors: bool = False,
    ) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler | None] = {}
        self._state = state
        self._expose_internal_errors = expose_internal_errors

    def register(self, spec: ToolSpec, handler: ToolHandler | None = None) -> None:
        if spec.name in self._specs:
            raise DuplicateToolError(f"Tool is already registered: {spec.name}")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise UnknownToolError(f"Unknown tool: {name}") from exc

    def list_specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def model_schemas(self) -> list[dict[str, Any]]:
        return [spec.model_schema() for spec in self._specs.values()]

    def execute(self, name: str, arguments: Mapping[str, Any] | object) -> ToolExecutionResult:
        """Validate arguments, execute one handler, and contain every exception."""
        try:
            spec = self.get(name)
        except UnknownToolError:
            return _runtime_failure(
                ToolErrorCode.UNKNOWN_TOOL,
                f"Unknown tool: {name}",
                f"Unknown tool rejected: {name}",
            )

        try:
            validated = spec.args_model.model_validate(arguments)
        except ValidationError as exc:
            errors = [
                {
                    "type": error["type"],
                    "loc": list(error["loc"]),
                    "msg": error["msg"],
                }
                for error in exc.errors()
            ]
            return _runtime_failure(
                ToolErrorCode.INVALID_ARGUMENTS,
                "Tool arguments failed validation",
                f"Invalid arguments for {name}",
                details={"errors": errors},
            )

        handler = self._handlers[name]
        if handler is None:
            return _runtime_failure(
                ToolErrorCode.INTERNAL_TOOL_ERROR,
                "Tool handler is not configured",
                f"Tool execution failed: {name}",
            )
        try:
            return handler(validated)
        except ToolHandlerError as exc:
            return ToolExecutionResult(
                ok=False,
                error=ToolError(
                    code=exc.code,
                    message=exc.safe_message,
                    details=exc.details,
                ),
                trust=exc.trust,
                result_summary=exc.result_summary,
            )
        except Exception as exc:  # noqa: BLE001 - this is the tool isolation boundary
            details: dict[str, Any] = {"exception_type": type(exc).__name__}
            if self._expose_internal_errors:
                details["exception"] = repr(exc)
            return _runtime_failure(
                ToolErrorCode.INTERNAL_TOOL_ERROR,
                "Tool execution failed unexpectedly",
                f"Tool execution failed: {name}",
                details=details,
            )


def _runtime_failure(
    code: ToolErrorCode,
    message: str,
    summary: str,
    *,
    details: dict[str, Any] | None = None,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        ok=False,
        error=ToolError(code=code.value, message=message, details=details or {}),
        trust="trusted_runtime_data",
        result_summary=summary,
    )
