from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.runtime import TraceEvent
from app.trace import JsonlTraceWriter, TracePathError, TraceWriterClosedError


def make_event(*, step: int = 1, summary: str = "完成") -> TraceEvent:
    return TraceEvent(
        run_id="run-1",
        step=step,
        timestamp=datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC),
        tool="sample_tool",
        args={"文本": "你好"},
        ok=True,
        result_summary=summary,
        duration_ms=12.5,
    )


def test_jsonl_writer_creates_parent_flushes_and_writes_unicode(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "runs" / "nested" / "trace.jsonl"
    writer = JsonlTraceWriter(output, workspace_root=workspace)

    writer.write(make_event())
    text_before_close = output.read_text(encoding="utf-8")
    writer.write(make_event(step=2, summary="第二步"))
    writer.close()

    assert "你好" in text_before_close
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert list(first) == [
        "run_id",
        "step",
        "timestamp",
        "tool",
        "args",
        "ok",
        "result_summary",
        "duration_ms",
    ]
    assert first["timestamp"] == "2026-02-03T04:05:06Z"
    assert first["args"] == {"文本": "你好"}


def test_jsonl_writer_supports_context_manager_and_idempotent_close(tmp_path: Path) -> None:
    output = tmp_path / "trace.jsonl"

    with JsonlTraceWriter(output) as writer:
        writer.write(make_event())

    writer.close()
    assert writer.closed is True
    assert output.is_file()


def test_write_after_close_fails_clearly(tmp_path: Path) -> None:
    writer = JsonlTraceWriter(tmp_path / "trace.jsonl")
    writer.close()

    with pytest.raises(TraceWriterClosedError, match="closed"):
        writer.write(make_event())


def test_trace_path_inside_workspace_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(TracePathError, match="outside workspace"):
        JsonlTraceWriter(workspace / "trace.jsonl", workspace_root=workspace)

    assert not (workspace / "trace.jsonl").exists()


def test_resolved_symbolic_link_target_inside_workspace_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    output = outside / "trace-link.jsonl"
    path_type = type(output)
    original_resolve = path_type.resolve
    linked_target = original_resolve(workspace / "trace.jsonl", strict=False)

    def resolve_as_link(self: Path, strict: bool = False) -> Path:
        if self == output:
            return linked_target
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(path_type, "resolve", resolve_as_link)

    with pytest.raises(TracePathError, match="outside workspace"):
        JsonlTraceWriter(output, workspace_root=workspace)

    assert not output.exists()
