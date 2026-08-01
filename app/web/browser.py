"""Read-only Web workspace browsing through the existing sandbox and tools."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.config import AgentLimits
from app.filesystem_tools import build_filesystem_registry
from app.run_paths import generate_run_id
from app.runtime import RunState, ToolExecutionResult
from app.sandbox import SandboxError, WorkspaceSandbox

WEB_TREE_MAX_ENTRIES = 1000
WEB_TREE_MAX_DEPTH = 10
WEB_FILE_MAX_LINES = 300
WEB_FILE_MAX_CHARS = 20_000


class BrowserError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.safe_message = message
        super().__init__(message)


def _registry(workspace: Path):  # type: ignore[no-untyped-def]
    try:
        sandbox = WorkspaceSandbox(workspace)
    except SandboxError as exc:
        raise BrowserError(exc.code, exc.safe_message) from None
    state = RunState(
        run_id=f"browser-{generate_run_id()}",
        workspace_root=sandbox.root,
        started_at=datetime.now(UTC),
    )
    registry = build_filesystem_registry(sandbox, state, AgentLimits())
    return registry, state


def _data(result: ToolExecutionResult) -> dict[str, object]:
    if not result.ok or result.data is None:
        if result.error is None:
            raise BrowserError("BROWSER_ERROR", "Workspace browsing failed.")
        raise BrowserError(result.error.code, result.error.message)
    return result.data


def list_workspace_tree(workspace: Path) -> list[dict[str, object]]:
    """Return a sorted, bounded tree without retaining Agent observations."""
    registry, _state = _registry(workspace)
    result = registry.execute(
        "list_directory",
        {
            "path": ".",
            "recursive": True,
            "max_depth": WEB_TREE_MAX_DEPTH,
            "max_entries": WEB_TREE_MAX_ENTRIES,
        },
    )
    data = _data(result)
    if data.get("truncated") is True:
        raise BrowserError("TREE_LIMIT_EXCEEDED", "Workspace tree exceeds the Web limit.")
    entries = []
    for item in data["entries"]:  # type: ignore[index,union-attr]
        rendered = {"path": item["path"], "type": item["type"]}
        if item["type"] == "file":
            rendered["size_bytes"] = item["size"]
        entries.append(rendered)
    return entries


def read_workspace_file(
    workspace: Path,
    *,
    path: str,
    start_line: int = 1,
    max_lines: int = WEB_FILE_MAX_LINES,
) -> dict[str, object]:
    """Read one UTF-8 file page with a browser-only observation state."""
    registry, _state = _registry(workspace)
    result = registry.execute(
        "read_file",
        {
            "path": path,
            "start_line": start_line,
            "max_lines": min(max_lines, WEB_FILE_MAX_LINES),
            "max_chars": WEB_FILE_MAX_CHARS,
        },
    )
    data = _data(result)
    return {
        "path": data["path"],
        "content": data["content"],
        "start_line": data["start_line"],
        "end_line": data["end_line"],
        "total_lines": data["total_lines"],
        "next_start_line": data["next_start_line"],
        "truncated": data["truncated"],
        "size_bytes": data["size"],
        "sha256": data["sha256"],
    }
