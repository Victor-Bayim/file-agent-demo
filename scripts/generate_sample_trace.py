"""Generate the committed provider-free, sanitized example Trace."""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.application import execute_task
from app.config import AgentLimits
from app.model_types import (
    ModelFinishReason,
    ModelMessage,
    ModelResponse,
    ModelRole,
    ModelToolCall,
    ModelUsage,
)
from app.runtime import AgentRunStatus

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples" / "sample-trace.jsonl"
SAMPLE_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC).isoformat().replace("+00:00", "Z")


class OfflineTraceModelClient:
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = deque(response.model_copy(deep=True) for response in responses)

    async def complete(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        del messages, tools
        if not self._responses:
            raise RuntimeError("Offline sample responses are exhausted.")
        return self._responses.popleft().model_copy(deep=True)


def _response(
    content: str | None = None,
    *,
    tool_call: ModelToolCall | None = None,
) -> ModelResponse:
    calls = [] if tool_call is None else [tool_call]
    finish_reason = ModelFinishReason.STOP if tool_call is None else ModelFinishReason.TOOL_CALLS
    return ModelResponse(
        message=ModelMessage(role=ModelRole.ASSISTANT, content=content, tool_calls=calls),
        usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5, exact=True),
        finish_reason=finish_reason,
        raw_finish_reason=finish_reason.value,
        provider_model="offline-fake-model",
    )


def _call(call_id: str, name: str, arguments: dict[str, object]) -> ModelToolCall:
    return ModelToolCall.from_arguments(id=call_id, name=name, arguments=arguments)


def _sanitize(value: object) -> object:
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if not isinstance(value, dict):
        return value
    sanitized = {key: _sanitize(item) for key, item in value.items()}
    if "run_id" in sanitized:
        sanitized["run_id"] = "sample-run"
    if "timestamp" in sanitized:
        sanitized["timestamp"] = SAMPLE_TIMESTAMP
    for key in ("duration_ms", "elapsed_ms"):
        if key in sanitized:
            sanitized[key] = 0
    if sanitized.get("tool") == "write_file":
        args = sanitized.get("args")
        if isinstance(args, dict):
            content = args.get("content")
            if isinstance(content, dict):
                content.pop("preview", None)
    return sanitized


async def _generate() -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="file-agent-trace-") as temporary:
        root = Path(temporary)
        workspace = root / "workspace"
        (workspace / "notes").mkdir(parents=True)
        (workspace / "notes" / "input.txt").write_text(
            "alpha\nbeta\n",
            encoding="utf-8",
            newline="",
        )
        trace = root / "trace.jsonl"
        model = OfflineTraceModelClient(
            [
                _response(tool_call=_call("list", "list_directory", {"path": "."})),
                _response(tool_call=_call("read-input", "read_file", {"path": "notes/input.txt"})),
                _response(
                    tool_call=_call(
                        "write-summary",
                        "write_file",
                        {
                            "path": "summary.txt",
                            "content": "Observed two generic input lines.\n",
                        },
                    )
                ),
                _response(tool_call=_call("verify-summary", "read_file", {"path": "summary.txt"})),
                _response("Created and verified summary.txt."),
            ]
        )
        result = await execute_task(
            workspace=workspace,
            task="Inspect the generic input and create a verified one-line summary.",
            trace_path=trace,
            limits=AgentLimits(max_model_turns=8, max_tool_calls=8),
            model_client=model,
        )
        if result.status is not AgentRunStatus.COMPLETED:
            raise RuntimeError("Fake sample run did not complete.")
        records = []
        with trace.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                records.append(_sanitize(json.loads(raw_line)))
        return records  # type: ignore[return-value]


def main() -> int:
    records = asyncio.run(_generate())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
