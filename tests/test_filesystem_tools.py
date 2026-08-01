from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import app.filesystem_tools as filesystem_module
from app.config import AgentLimits
from app.filesystem_tools import build_filesystem_registry
from app.runtime import RunState, ToolExecutionResult
from app.sandbox import WorkspaceSandbox
from app.tools import ToolRegistry

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEED_WORKSPACE = REPOSITORY_ROOT / "workspace"
BASELINE_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "workspace_baseline.json"


def make_registry(
    root: Path,
    *,
    limits: AgentLimits | None = None,
) -> tuple[ToolRegistry, RunState]:
    state = RunState(
        run_id="phase-2-test",
        workspace_root=root,
        started_at=datetime.now(UTC),
    )
    registry = build_filesystem_registry(
        WorkspaceSandbox(root),
        state,
        limits or AgentLimits(),
    )
    return registry, state


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def execute_ok(registry: ToolRegistry, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = registry.execute(name, arguments)
    assert result.ok, result
    assert result.trust == "untrusted_workspace_data"
    assert result.data is not None
    return result.data


def error_code(result: ToolExecutionResult) -> str:
    assert not result.ok
    assert result.error is not None
    return result.error.code


def test_list_directory_nonrecursive_is_sorted_and_reports_file_sizes(workspace: Path) -> None:
    (workspace / "z.txt").write_text("123", encoding="utf-8")
    (workspace / "a-dir").mkdir()
    (workspace / "a-dir" / "nested.txt").write_text("nested", encoding="utf-8")
    registry, _ = make_registry(workspace)

    data = execute_ok(registry, "list_directory", {})

    assert data["entries"] == [
        {"path": "a-dir", "type": "directory"},
        {"path": "z.txt", "type": "file", "size": 3},
    ]
    assert data["truncated"] is False


def test_list_directory_recursive_honors_depth_and_entry_limit(workspace: Path) -> None:
    (workspace / "one" / "two" / "three").mkdir(parents=True)
    (workspace / "one" / "root.txt").write_text("x", encoding="utf-8")
    (workspace / "one" / "two" / "deep.txt").write_text("xx", encoding="utf-8")
    registry, _ = make_registry(workspace)

    depth_data = execute_ok(
        registry,
        "list_directory",
        {"recursive": True, "max_depth": 2, "max_entries": 100},
    )
    limited_data = execute_ok(
        registry,
        "list_directory",
        {"recursive": True, "max_depth": 10, "max_entries": 2},
    )

    depth_paths = [entry["path"] for entry in depth_data["entries"]]
    assert depth_paths == sorted(depth_paths)
    assert "one/two/deep.txt" not in depth_paths
    assert limited_data["returned_entries"] == 2
    assert limited_data["truncated"] is True


def test_recursive_directory_order_is_globally_sorted(workspace: Path) -> None:
    (workspace / "a").mkdir()
    (workspace / "a" / "z.txt").write_text("nested", encoding="utf-8")
    (workspace / "a.txt").write_text("sibling", encoding="utf-8")
    registry, _ = make_registry(workspace)

    data = execute_ok(registry, "list_directory", {"recursive": True})
    paths = [entry["path"] for entry in data["entries"]]

    assert paths == ["a", "a.txt", "a/z.txt"]


def test_list_directory_reports_but_does_not_traverse_links(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked = workspace / "linked"
    linked.mkdir()
    (linked / "secret.txt").write_text("secret", encoding="utf-8")
    real_detector = filesystem_module.is_link_or_reparse_point
    monkeypatch.setattr(
        filesystem_module,
        "is_link_or_reparse_point",
        lambda path: path == linked or real_detector(path),
    )
    registry, _ = make_registry(workspace)

    data = execute_ok(
        registry,
        "list_directory",
        {"recursive": True, "max_depth": 10},
    )

    assert {entry["path"]: entry["type"] for entry in data["entries"]} == {"linked": "symlink"}


def test_search_text_case_glob_exclude_context_and_aggregation(workspace: Path) -> None:
    (workspace / "included").mkdir()
    (workspace / "excluded").mkdir()
    (workspace / "included" / "b.md").write_text(
        "before\nNeedle one\nmiddle\nNeedle two\nafter\n",
        encoding="utf-8",
    )
    (workspace / "included" / "a.txt").write_text("needle lower\n", encoding="utf-8")
    (workspace / "excluded" / "c.md").write_text("Needle hidden\n", encoding="utf-8")
    registry, _ = make_registry(workspace)

    sensitive = execute_ok(
        registry,
        "search_text",
        {
            "query": "Needle",
            "glob": "**/*.md",
            "exclude_paths": ["excluded"],
            "context_lines": 1,
        },
    )
    insensitive = execute_ok(
        registry,
        "search_text",
        {"query": "NEEDLE", "case_sensitive": False, "context_lines": 0},
    )

    assert sensitive["total_files"] == 1
    assert sensitive["returned_matches"] == 2
    assert sensitive["files"][0]["match_count"] == 2
    assert [match["line"] for match in sensitive["files"][0]["matches"]] == [2, 4]
    assert "before\nNeedle one\nmiddle" in sensitive["files"][0]["matches"][0]["snippet"]
    assert insensitive["returned_matches"] == 4
    assert [item["path"] for item in insensitive["files"]] == sorted(
        item["path"] for item in insensitive["files"]
    )


def test_search_text_counts_multiple_exact_matches_on_one_line(workspace: Path) -> None:
    (workspace / "phrases.txt").write_text("red blue red blue\n", encoding="utf-8")
    registry, _ = make_registry(workspace)

    data = execute_ok(
        registry,
        "search_text",
        {"query": "red blue", "context_lines": 0},
    )

    assert data["total_files"] == 1
    assert data["total_matches"] == 2
    assert [match["line"] for match in data["files"][0]["matches"]] == [1, 1]


def test_search_text_schema_explains_literal_and_aggregate_semantics(workspace: Path) -> None:
    registry, _ = make_registry(workspace)
    spec = registry.get("search_text")
    description = spec.description
    query_description = spec.json_schema()["properties"]["query"]["description"]

    assert "using query exactly as provided without semantic expansion" in description
    assert "total_files counts files with at least one match" in description
    assert "each file counted once" in description
    assert "total_matches counts all matching occurrences" in description
    assert "returned_matches is the number" in description
    assert "total_files and total_matches are complete when scan_complete=true" in description
    assert "truncated=false" in description
    assert "Count-only tasks" in description
    assert "Do not automatically broaden" in description
    assert "Literal text" in query_description
    assert "without semantic expansion" in query_description
    for seeded_value in ("Project Falcon", "Project Phoenix", "10 files", "14 matches"):
        assert seeded_value not in description
        assert seeded_value not in query_description


def test_write_file_description_distinguishes_commit_from_business_validation(
    workspace: Path,
) -> None:
    registry, _ = make_registry(workspace)
    description = registry.get("write_file").description

    assert "successful write confirms the filesystem commit" in description
    assert "not that the business structure or content satisfies the task" in description
    assert "read the file afterward" in description
    for task_specific_value in (
        "archive/MANIFEST.md",
        "api-v1-spec.md",
        "Project Falcon",
        "Project Phoenix",
    ):
        assert task_specific_value not in description


def test_write_file_description_explains_exact_complete_content_and_correction(
    workspace: Path,
) -> None:
    registry, _ = make_registry(workspace)
    description = registry.get("write_file").description

    assert "content argument is the complete file content" in description
    assert "adds no headings, blank lines, Markdown, or explanations" in description
    assert "exact format, content must contain only what the user requested" in description
    assert "read it to establish the current observation" in description
    assert "overwrite=true" in description
    assert "read it again" in description
    for task_specific_value in (
        "archive/MANIFEST.md",
        "api-v1-spec.md",
        "blog-post-launch.md",
        "onboarding-guide.md",
        "status: obsolete",
        "drafts/",
        "archive/",
    ):
        assert task_specific_value not in description


def test_search_text_complete_result_summary_is_deterministic_and_aggregate(
    workspace: Path,
) -> None:
    (workspace / "alpha.txt").write_text("token token\n", encoding="utf-8")
    (workspace / "beta.txt").write_text("token\n", encoding="utf-8")
    registry, _ = make_registry(workspace)

    result = registry.execute("search_text", {"query": "token", "context_lines": 0})

    assert result.ok is True
    assert result.trust == "untrusted_workspace_data"
    assert result.data is not None
    assert result.data["total_files"] == 2
    assert result.data["total_matches"] == 3
    assert result.data["returned_matches"] == 3
    assert result.data["scan_complete"] is True
    assert result.data["truncated"] is False
    assert result.result_summary == (
        "Literal search completed: 2 matching files and 3 matching occurrences; "
        "returned_matches=3; scan_complete=true; truncated=false."
    )
    assert "Project" not in result.result_summary


def test_search_text_incomplete_summary_does_not_claim_complete_totals(workspace: Path) -> None:
    (workspace / "many.txt").write_text("token\ntoken\n", encoding="utf-8")
    registry, _ = make_registry(workspace)

    result = registry.execute(
        "search_text",
        {"query": "token", "context_lines": 0, "max_results": 1},
    )

    assert result.ok is True
    assert "stopped at the result limit" in result.result_summary
    assert "so far" in result.result_summary
    assert "scan_complete=false" in result.result_summary
    assert "truncated=true" in result.result_summary
    assert "search completed" not in result.result_summary.lower()


def test_search_text_accepts_a_direct_file_and_rejects_unsafe_excludes(workspace: Path) -> None:
    (workspace / "direct.txt").write_text("direct match\n", encoding="utf-8")
    registry, _ = make_registry(workspace)

    direct = execute_ok(
        registry,
        "search_text",
        {"path": "direct.txt", "query": "match", "context_lines": 0},
    )
    unsafe_exclude = registry.execute(
        "search_text",
        {"query": "match", "exclude_paths": ["../outside"]},
    )

    assert direct["files"][0]["path"] == "direct.txt"
    assert error_code(unsafe_exclude) == "INVALID_PATH"


def test_search_text_truncates_immediately_at_max_results(workspace: Path) -> None:
    (workspace / "many.txt").write_text("hit\nhit\nhit\n", encoding="utf-8")
    registry, _ = make_registry(workspace)

    data = execute_ok(
        registry,
        "search_text",
        {"query": "hit", "max_results": 2, "context_lines": 0},
    )

    assert data["returned_matches"] == 2
    assert data["truncated"] is True
    assert data["scan_complete"] is False


def test_search_text_supports_unicode_and_skips_binary(workspace: Path) -> None:
    (workspace / "unicode.txt").write_text("第一行\n文件助手在这里\n", encoding="utf-8")
    (workspace / "binary.bin").write_bytes(b"prefix\x00needle")
    registry, _ = make_registry(workspace)

    unicode_data = execute_ok(registry, "search_text", {"query": "文件助手"})
    binary_data = execute_ok(registry, "search_text", {"query": "needle"})

    assert unicode_data["files"][0]["matches"][0]["line"] == 2
    assert binary_data["returned_matches"] == 0
    assert binary_data["skipped_binary"] == ["binary.bin"]


def test_search_finds_query_crossing_internal_buffer_boundary(workspace: Path) -> None:
    prefix = "x" * (8 * 1024 - 3)
    (workspace / "boundary.txt").write_text(f"{prefix}boundary-token\n", encoding="utf-8")
    registry, _ = make_registry(workspace)

    data = execute_ok(
        registry,
        "search_text",
        {"query": "boundary-token", "context_lines": 0},
    )

    assert data["returned_matches"] == 1
    assert "boundary-token" in data["files"][0]["matches"][0]["snippet"]


class GuardedFile:
    def __init__(self, handle: Any, read_sizes: list[int]) -> None:
        self._handle = handle
        self._read_sizes = read_sizes

    def read(self, size: int = -1) -> Any:
        assert size >= 0, "unbounded read() is forbidden"
        self._read_sizes.append(size)
        return self._handle.read(size)

    def __iter__(self) -> Any:
        return iter(self._handle)

    def __enter__(self) -> GuardedFile:
        self._handle.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._handle.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


def test_large_search_is_streamed_and_aggregates_distant_matches(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    large = workspace / "large.txt"
    lines = ["first marker\n"]
    lines.extend(f"ordinary line {number:06d}\n" for number in range(55_000))
    lines.append("second marker\n")
    large.write_text("".join(lines), encoding="utf-8")
    del lines
    registry, _ = make_registry(workspace)
    original_open = Path.open
    read_sizes: list[int] = []

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> GuardedFile:
        return GuardedFile(original_open(path, *args, **kwargs), read_sizes)

    monkeypatch.setattr(Path, "open", guarded_open)
    data = execute_ok(
        registry,
        "search_text",
        {"query": "marker", "context_lines": 0},
    )

    assert large.stat().st_size >= 1_000_000
    assert data["total_files"] == 1
    assert data["files"][0]["match_count"] == 2
    assert [match["line"] for match in data["files"][0]["matches"]] == [1, 55_002]
    assert read_sizes == [filesystem_module.BINARY_SAMPLE_SIZE]


def test_read_file_full_range_hash_and_observation(workspace: Path) -> None:
    content = "alpha\n中文\nomega\n"
    path = workspace / "sample.txt"
    path.write_text(content, encoding="utf-8", newline="")
    registry, state = make_registry(workspace)

    data = execute_ok(registry, "read_file", {"path": "sample.txt"})

    expected_hash = hashlib.sha256(content.encode()).hexdigest()
    assert data == {
        "path": "sample.txt",
        "start_line": 1,
        "end_line": 3,
        "total_lines": 3,
        "truncated": False,
        "next_start_line": None,
        "content": content,
        "size": len(content.encode()),
        "sha256": expected_hash,
    }
    assert state.get_observation("sample.txt").sha256 == expected_hash  # type: ignore[union-attr]


def test_read_file_start_line_max_lines_and_next_line(workspace: Path) -> None:
    (workspace / "lines.txt").write_text(
        "one\ntwo\nthree\nfour\n",
        encoding="utf-8",
        newline="",
    )
    registry, _ = make_registry(workspace)

    data = execute_ok(
        registry,
        "read_file",
        {"path": "lines.txt", "start_line": 2, "max_lines": 2},
    )

    assert data["content"] == "two\nthree\n"
    assert data["end_line"] == 3
    assert data["total_lines"] == 4
    assert data["truncated"] is True
    assert data["next_start_line"] == 4


def test_read_file_max_chars_counts_unicode_characters(workspace: Path) -> None:
    first = "界" * 60 + "\n"
    second = "文" * 60 + "\n"
    (workspace / "chars.txt").write_text(first + second, encoding="utf-8", newline="")
    registry, _ = make_registry(workspace)

    data = execute_ok(
        registry,
        "read_file",
        {"path": "chars.txt", "max_chars": 100},
    )

    assert data["content"] == first
    assert len(data["content"]) == 61
    assert data["next_start_line"] == 2


def test_read_file_rejects_one_logical_line_larger_than_character_limit(
    workspace: Path,
) -> None:
    (workspace / "long-line.txt").write_text("x" * 120, encoding="utf-8")
    registry, state = make_registry(workspace)

    result = registry.execute(
        "read_file",
        {"path": "long-line.txt", "max_chars": 100},
    )

    assert error_code(result) == "READ_LIMIT_EXCEEDED"
    assert state.get_observation("long-line.txt") is None


def test_read_file_empty_file_and_start_beyond_end(workspace: Path) -> None:
    (workspace / "empty.txt").write_bytes(b"")
    (workspace / "one.txt").write_text("one\n", encoding="utf-8")
    registry, _ = make_registry(workspace)

    empty = execute_ok(registry, "read_file", {"path": "empty.txt"})
    beyond = execute_ok(registry, "read_file", {"path": "one.txt", "start_line": 5})

    assert empty["total_lines"] == 0
    assert empty["end_line"] == 0
    assert empty["content"] == ""
    assert beyond["total_lines"] == 1
    assert beyond["end_line"] == 4
    assert beyond["truncated"] is False


def test_read_file_rejects_binary_directory_and_link(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (workspace / "binary.bin").write_bytes(b"abc\x00def")
    (workspace / "folder").mkdir()
    linked = workspace / "linked.txt"
    linked.write_text("text", encoding="utf-8")
    registry, _ = make_registry(workspace)

    assert error_code(registry.execute("read_file", {"path": "binary.bin"})) == (
        "BINARY_FILE_NOT_SUPPORTED"
    )
    assert error_code(registry.execute("read_file", {"path": "folder"})) == "NOT_A_FILE"

    import app.sandbox as sandbox_module

    real_detector = sandbox_module.is_link_or_reparse_point
    monkeypatch.setattr(
        sandbox_module,
        "is_link_or_reparse_point",
        lambda path: path == linked or real_detector(path),
    )
    assert error_code(registry.execute("read_file", {"path": "linked.txt"})) == (
        "SYMLINK_NOT_ALLOWED"
    )


def test_read_file_never_uses_unbounded_read(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (workspace / "large.txt").write_text("line\n" * 250_000, encoding="utf-8")
    registry, _ = make_registry(workspace)
    original_open = Path.open
    read_sizes: list[int] = []

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> GuardedFile:
        return GuardedFile(original_open(path, *args, **kwargs), read_sizes)

    monkeypatch.setattr(Path, "open", guarded_open)
    data = execute_ok(
        registry,
        "read_file",
        {"path": "large.txt", "max_lines": 2, "max_chars": 100},
    )

    assert data["total_lines"] == 250_000
    assert read_sizes
    assert set(read_sizes) == {filesystem_module.HASH_CHUNK_SIZE}


def test_create_directory_success_idempotence_conflicts_and_mutations(workspace: Path) -> None:
    (workspace / "parent").mkdir()
    (workspace / "taken").write_text("x", encoding="utf-8")
    registry, state = make_registry(workspace)

    created = registry.execute("create_directory", {"path": "parent/new"})
    repeated = registry.execute("create_directory", {"path": "parent/new"})
    conflict = registry.execute("create_directory", {"path": "taken"})
    missing_parent = registry.execute("create_directory", {"path": "missing/new"})
    outside = registry.execute("create_directory", {"path": "../outside"})

    assert created.ok and created.data["created"] is True  # type: ignore[index]
    assert repeated.ok and repeated.data["created"] is False  # type: ignore[index]
    assert error_code(conflict) == "TARGET_ALREADY_EXISTS"
    assert error_code(missing_parent) == "PARENT_NOT_FOUND"
    assert error_code(outside) == "INVALID_PATH"
    assert [record.status for record in state.mutations] == [
        "succeeded",
        "succeeded",
        "failed",
        "failed",
        "failed",
    ]
    assert [record.changed for record in state.mutations] == [True, False, False, False, False]
    assert all(record.error_code for record in state.mutations[2:])


def test_write_file_create_unicode_limit_and_default_no_overwrite(workspace: Path) -> None:
    registry, state = make_registry(workspace, limits=AgentLimits(max_write_bytes=12))

    created = registry.execute("write_file", {"path": "new.txt", "content": "你好"})
    conflict = registry.execute("write_file", {"path": "new.txt", "content": "again"})
    too_large = registry.execute("write_file", {"path": "large.txt", "content": "界" * 5})

    assert created.ok
    assert (workspace / "new.txt").read_text(encoding="utf-8") == "你好"
    assert created.data["bytes_written"] == 6  # type: ignore[index]
    assert error_code(conflict) == "TARGET_ALREADY_EXISTS"
    assert error_code(too_large) == "WRITE_TOO_LARGE"
    assert state.get_observation("new.txt") is not None
    assert [record.status for record in state.mutations] == ["succeeded", "failed", "failed"]
    assert [record.changed for record in state.mutations] == [True, False, False]


def test_write_overwrite_requires_current_observation(workspace: Path) -> None:
    target = workspace / "target.txt"
    target.write_text("old\n", encoding="utf-8", newline="")
    registry, state = make_registry(workspace)

    unobserved = registry.execute(
        "write_file",
        {"path": "target.txt", "content": "new\n", "overwrite": True},
    )
    execute_ok(registry, "read_file", {"path": "target.txt"})
    overwritten = registry.execute(
        "write_file",
        {"path": "target.txt", "content": "new\n", "overwrite": True},
    )

    assert error_code(unobserved) == "SOURCE_NOT_OBSERVED"
    assert overwritten.ok
    assert target.read_text(encoding="utf-8") == "new\n"
    observation = state.get_observation("target.txt")
    assert observation is not None
    assert observation.sha256 == hashlib.sha256(b"new\n").hexdigest()
    assert state.mutations[-1].before_sha256 == hashlib.sha256(b"old\n").hexdigest()


def test_write_rejects_file_changed_after_observation(workspace: Path) -> None:
    target = workspace / "target.txt"
    target.write_text("old", encoding="utf-8")
    registry, state = make_registry(workspace)
    execute_ok(registry, "read_file", {"path": "target.txt"})
    target.write_text("external", encoding="utf-8")

    result = registry.execute(
        "write_file",
        {"path": "target.txt", "content": "new", "overwrite": True},
    )

    assert error_code(result) == "SOURCE_CHANGED"
    assert target.read_text(encoding="utf-8") == "external"
    assert state.get_observation("target.txt") is None
    assert state.mutations[-1].status == "failed"


def test_write_uses_atomic_replace_and_cleans_temporary_file(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _ = make_registry(workspace)
    original_replace = os.replace
    calls: list[tuple[Path, Path]] = []

    def tracked_replace(source: str | Path, destination: str | Path) -> None:
        calls.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(filesystem_module.os, "replace", tracked_replace)
    result = registry.execute("write_file", {"path": "atomic.txt", "content": "content"})

    assert result.ok
    assert len(calls) == 1
    assert calls[0][0].parent == workspace
    assert calls[0][1] == workspace / "atomic.txt"
    assert list(workspace.glob(".atomic.txt.*.tmp")) == []


def test_write_failure_cleans_temporary_file_and_hides_internal_path(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, state = make_registry(workspace)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"failure at {source} and {destination}")

    monkeypatch.setattr(filesystem_module.os, "replace", fail_replace)
    result = registry.execute("write_file", {"path": "failed.txt", "content": "content"})

    assert error_code(result) == "INTERNAL_TOOL_ERROR"
    assert str(workspace) not in result.error.message  # type: ignore[union-attr]
    assert str(workspace) not in json.dumps(result.model_dump(mode="json"))
    assert list(workspace.glob(".failed.txt.*.tmp")) == []
    assert not (workspace / "failed.txt").exists()
    assert state.mutations[-1].error_code == "INTERNAL_TOOL_ERROR"


def test_write_rejects_link_destination(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked = workspace / "linked.txt"
    linked.write_text("old", encoding="utf-8")
    import app.sandbox as sandbox_module

    real_detector = sandbox_module.is_link_or_reparse_point
    monkeypatch.setattr(
        sandbox_module,
        "is_link_or_reparse_point",
        lambda path: path == linked or real_detector(path),
    )
    registry, state = make_registry(workspace)

    result = registry.execute("write_file", {"path": "linked.txt", "content": "new"})

    assert error_code(result) == "SYMLINK_NOT_ALLOWED"
    assert linked.read_text(encoding="utf-8") == "old"
    assert state.mutations[-1].status == "failed"


def test_write_requires_existing_directory_parent_and_regular_overwrite_target(
    workspace: Path,
) -> None:
    (workspace / "folder").mkdir()
    registry, _ = make_registry(workspace)

    missing_parent = registry.execute(
        "write_file",
        {"path": "missing/new.txt", "content": "new"},
    )
    directory_target = registry.execute(
        "write_file",
        {"path": "folder", "content": "new", "overwrite": True},
    )
    missing_overwrite = registry.execute(
        "write_file",
        {"path": "absent.txt", "content": "new", "overwrite": True},
    )

    assert error_code(missing_parent) == "PARENT_NOT_FOUND"
    assert error_code(directory_target) == "NOT_A_FILE"
    assert error_code(missing_overwrite) == "PATH_NOT_FOUND"


def observe(registry: ToolRegistry, path: str) -> str:
    data = execute_ok(registry, "read_file", {"path": path})
    return str(data["sha256"])


def test_move_file_preserves_hash_and_transfers_observation(workspace: Path) -> None:
    (workspace / "archive").mkdir()
    (workspace / "source.txt").write_text("content\n", encoding="utf-8")
    registry, state = make_registry(workspace)
    original_hash = observe(registry, "source.txt")

    result = registry.execute(
        "move_file",
        {"source": "source.txt", "destination": "archive/destination.txt"},
    )

    assert result.ok
    assert not (workspace / "source.txt").exists()
    assert (workspace / "archive" / "destination.txt").is_file()
    assert result.data["sha256"] == original_hash  # type: ignore[index]
    assert state.get_observation("source.txt") is None
    assert state.get_observation("archive/destination.txt").sha256 == original_hash  # type: ignore[union-attr]
    assert state.mutations[-1].before_sha256 == state.mutations[-1].after_sha256
    assert state.mutations[-1].changed is True


def test_move_requires_observation_and_rejects_changed_source(workspace: Path) -> None:
    source = workspace / "source.txt"
    source.write_text("original", encoding="utf-8")
    registry, state = make_registry(workspace)

    unobserved = registry.execute(
        "move_file",
        {"source": "source.txt", "destination": "first.txt"},
    )
    observe(registry, "source.txt")
    source.write_text("external", encoding="utf-8")
    changed = registry.execute(
        "move_file",
        {"source": "source.txt", "destination": "second.txt"},
    )

    assert error_code(unobserved) == "SOURCE_NOT_OBSERVED"
    assert error_code(changed) == "SOURCE_CHANGED"
    assert source.read_text(encoding="utf-8") == "external"
    assert state.get_observation("source.txt") is None
    assert all(record.status == "failed" for record in state.mutations)


def test_move_rejects_existing_target_missing_parent_same_path_and_directory(
    workspace: Path,
) -> None:
    (workspace / "source.txt").write_text("source", encoding="utf-8")
    (workspace / "target.txt").write_text("target", encoding="utf-8")
    (workspace / "folder").mkdir()
    registry, _ = make_registry(workspace)
    observe(registry, "source.txt")

    existing = registry.execute(
        "move_file",
        {"source": "source.txt", "destination": "target.txt"},
    )
    missing_parent = registry.execute(
        "move_file",
        {"source": "source.txt", "destination": "missing/target.txt"},
    )
    same = registry.execute(
        "move_file",
        {"source": "source.txt", "destination": "source.txt"},
    )
    directory = registry.execute(
        "move_file",
        {"source": "folder", "destination": "other"},
    )

    assert error_code(existing) == "TARGET_ALREADY_EXISTS"
    assert error_code(missing_parent) == "PARENT_NOT_FOUND"
    assert error_code(same) == "SAME_SOURCE_AND_DESTINATION"
    assert error_code(directory) == "NOT_A_FILE"
    assert (workspace / "source.txt").exists()


def test_move_exact_line_is_complete_line_only_and_ignores_newline(workspace: Path) -> None:
    (workspace / "accepted.txt").write_text("alpha\r\nexact value\r\nomega\r\n", encoding="utf-8")
    (workspace / "rejected.txt").write_text("prefix exact value suffix\n", encoding="utf-8")
    registry, state = make_registry(workspace)
    observe(registry, "accepted.txt")
    observe(registry, "rejected.txt")

    accepted = registry.execute(
        "move_file",
        {
            "source": "accepted.txt",
            "destination": "moved.txt",
            "require_exact_line": "exact value",
        },
    )
    rejected = registry.execute(
        "move_file",
        {
            "source": "rejected.txt",
            "destination": "not-moved.txt",
            "require_exact_line": "exact value",
        },
    )

    assert accepted.ok
    assert error_code(rejected) == "PRECONDITION_FAILED"
    assert (workspace / "rejected.txt").exists()
    assert state.mutations[-1].status == "failed"


def _current_seed_hashes() -> dict[str, str]:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {
        item["path"]: hashlib.sha256((SEED_WORKSPACE / item["path"]).read_bytes()).hexdigest()
        for item in baseline["files"]
    }


def test_real_workspace_read_only_search_and_read_preserve_baseline() -> None:
    before = _current_seed_hashes()
    registry, _ = make_registry(SEED_WORKSPACE)

    search = execute_ok(
        registry,
        "search_text",
        {"query": "Project Falcon", "context_lines": 0, "max_results": 50},
    )
    matching_large_files = [
        item for item in search["files"] if item["size"] >= 900_000 and item["match_count"] == 2
    ]
    first_path = search["files"][0]["path"]
    read = execute_ok(registry, "read_file", {"path": first_path, "max_lines": 1})
    after = _current_seed_hashes()

    assert search["total_files"] == 10
    assert search["total_matches"] == 14
    assert search["scan_complete"] is True
    assert len(matching_large_files) == 1
    assert read["path"] == first_path
    assert before == after
