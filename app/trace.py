"""Reliable JSONL trace persistence outside the audited workspace."""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Protocol, TextIO

from app.runtime import TraceEvent


class TraceError(RuntimeError):
    """Base class for trace infrastructure failures."""


class TracePathError(TraceError):
    """Raised when a trace destination violates the workspace boundary."""


class TraceWriterClosedError(TraceError):
    """Raised when an event is written after the writer has closed."""


class TraceWriter(Protocol):
    def write(self, event: TraceEvent) -> None:
        """Persist a trace event."""
        ...

    def close(self) -> None:
        """Release trace resources."""
        ...


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class JsonlTraceWriter:
    """Append UTF-8 JSON events and flush each event immediately."""

    def __init__(self, output_path: Path, workspace_root: Path | None = None) -> None:
        self._workspace_root = (
            workspace_root.resolve(strict=False) if workspace_root is not None else None
        )
        self._output_path = self._resolve_and_validate(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path = self._resolve_and_validate(self._output_path)
        self._handle: TextIO = self._output_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        )
        self._closed = False

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def closed(self) -> bool:
        return self._closed

    def _resolve_and_validate(self, output_path: Path) -> Path:
        resolved_output = output_path.resolve(strict=False)
        if self._workspace_root is not None and _is_within(
            resolved_output,
            self._workspace_root,
        ):
            raise TracePathError(f"Trace output must be outside workspace: {resolved_output}")
        return resolved_output

    def write(self, event: TraceEvent) -> None:
        if self._closed:
            raise TraceWriterClosedError("Cannot write to a closed trace writer")
        payload = event.model_dump(mode="json")
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._handle.write(f"{line}\n")
        self._handle.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._handle.close()
        self._closed = True

    def __enter__(self) -> JsonlTraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
