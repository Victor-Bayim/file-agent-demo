"""Strict tool JSON handling, context compaction, and trace-safe arguments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.model_types import ModelMessage, ModelRole
from app.runtime import ToolError, ToolExecutionResult
from app.tools import ToolErrorCode

SENSITIVE_FIELD_NAMES = {
    "apikey",
    "authorization",
    "token",
    "password",
    "secret",
    "accesscode",
}
MAX_TRACE_STRING_CHARS = 256
TRACE_PREVIEW_CHARS = 96
COMPACTED_TOOL_MARKER = {"tool_result_compacted": True}


def parse_tool_arguments(
    arguments_json: str,
) -> tuple[dict[str, Any] | None, ToolExecutionResult | None]:
    """Parse strict JSON without allowing non-object or non-standard values."""
    try:
        parsed = json.loads(arguments_json, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, _argument_error(
            ToolErrorCode.INVALID_TOOL_CALL_JSON,
            "Tool arguments are not valid strict JSON",
            "Tool call rejected: invalid JSON",
        )
    if not isinstance(parsed, dict):
        return None, _argument_error(
            ToolErrorCode.INVALID_ARGUMENTS,
            "Tool arguments must decode to a JSON object",
            "Tool call rejected: arguments are not an object",
        )
    return parsed, None


def serialize_tool_result_for_model(result: ToolExecutionResult, max_chars: int) -> str:
    """Serialize a complete result or a valid bounded JSON summary wrapper."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    payload: dict[str, Any] = {
        "ok": result.ok,
        "trust": result.trust,
        "result_summary": result.result_summary,
    }
    if result.ok:
        payload["data"] = result.data
    else:
        payload["error"] = result.error.model_dump(mode="json") if result.error else None
    serialized = _stable_json(payload)
    if len(serialized) <= max_chars:
        return serialized

    wrapper = {
        "ok": result.ok,
        "trust": result.trust,
        "result_summary": result.result_summary,
        "result_truncated_for_model": True,
        "instruction": "Narrow the query or request a smaller range.",
    }
    for summary_limit in (256, 128, 64, 32, 16, 1):
        wrapper["result_summary"] = _truncate(result.result_summary, summary_limit)
        serialized = _stable_json(wrapper)
        if len(serialized) <= max_chars:
            return serialized
    raise ValueError("max_chars is too small for the required structured result wrapper")


def compact_tool_history(
    messages: Sequence[ModelMessage],
    max_tool_history_chars: int,
) -> list[ModelMessage]:
    """Copy history and compact oldest tool content while preserving every role."""
    if max_tool_history_chars <= 0:
        raise ValueError("max_tool_history_chars must be positive")
    copied = [message.model_copy(deep=True) for message in messages]
    tool_indices = [index for index, message in enumerate(copied) if message.role is ModelRole.TOOL]
    full_contents = {index: copied[index].content or "null" for index in tool_indices}
    if sum(len(content) for content in full_contents.values()) <= max_tool_history_chars:
        return copied

    compacted = {index: _compact_tool_content(content) for index, content in full_contents.items()}
    total = sum(len(content) for content in compacted.values())
    for index in tool_indices:
        if total <= max_tool_history_chars:
            break
        minimal = _stable_json(COMPACTED_TOOL_MARKER)
        total += len(minimal) - len(compacted[index])
        compacted[index] = minimal
    for index in tool_indices:
        if total <= max_tool_history_chars:
            break
        total += 2 - len(compacted[index])
        compacted[index] = "{}"
    if total > max_tool_history_chars:
        raise ValueError("tool history budget is too small to preserve tool message pairing")

    for index in reversed(tool_indices):
        candidate_total = total - len(compacted[index]) + len(full_contents[index])
        if candidate_total > max_tool_history_chars:
            break
        compacted[index] = full_contents[index]
        total = candidate_total
    for index, content in compacted.items():
        copied[index].content = content
    return copied


def sanitize_trace_args(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-safe arguments with recursive secret and content reduction."""
    sanitized: dict[str, Any] = {}
    for key, value in arguments.items():
        key_text = str(key)
        normalized_key = re.sub(r"[-_]", "", key_text).casefold()
        if normalized_key in SENSITIVE_FIELD_NAMES:
            sanitized[key_text] = "[REDACTED]"
        elif tool_name == "write_file" and normalized_key == "content" and isinstance(value, str):
            encoded = value.encode("utf-8")
            sanitized[key_text] = {
                "characters": len(value),
                "utf8_bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "preview": _truncate(value, TRACE_PREVIEW_CHARS),
            }
        else:
            sanitized[key_text] = _sanitize_value(value)
    return sanitized


def canonical_tool_fingerprint(tool_name: str, arguments_json: str) -> str:
    """Canonicalize valid JSON key order; retain invalid raw argument semantics."""
    try:
        parsed = json.loads(arguments_json, parse_constant=_reject_json_constant)
        normalized = _stable_json(parsed)
    except (json.JSONDecodeError, ValueError, TypeError):
        normalized = arguments_json
    return f"{tool_name}\x00{normalized}"


def trace_arguments_from_json(tool_name: str, arguments_json: str) -> dict[str, Any]:
    """Sanitize parsed object arguments or retain only a bounded raw preview."""
    arguments, error = parse_tool_arguments(arguments_json)
    if error is not None or arguments is None:
        return {
            "raw_arguments_json": _truncate(arguments_json, MAX_TRACE_STRING_CHARS),
            "arguments_json_valid": False,
        }
    return sanitize_trace_args(tool_name, arguments)


def _argument_error(code: ToolErrorCode, message: str, summary: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        ok=False,
        error=ToolError(code=code.value, message=message),
        trust="trusted_runtime_data",
        result_summary=summary,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _compact_tool_content(content: str) -> str:
    summary: dict[str, Any] = dict(COMPACTED_TOOL_MARKER)
    try:
        payload = json.loads(content, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError, TypeError):
        return _stable_json(summary)
    if isinstance(payload, dict):
        if isinstance(payload.get("trust"), str):
            summary["trust"] = payload["trust"]
        if isinstance(payload.get("result_summary"), str):
            summary["result_summary"] = _truncate(payload["result_summary"], 160)
    return _stable_json(summary)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return sanitize_trace_args("", value)
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, str):
        return _truncate(value, MAX_TRACE_STRING_CHARS)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return _truncate(str(value), MAX_TRACE_STRING_CHARS)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = f"…[truncated; chars={len(value)}]"
    if len(marker) >= limit:
        return marker[:limit]
    return f"{value[: limit - len(marker)]}{marker}"


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
