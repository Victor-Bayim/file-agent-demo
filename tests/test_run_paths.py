from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

import app.run_paths as run_paths_module
from app.run_paths import (
    RunPathError,
    create_workspace_copy,
    default_trace_path,
    generate_run_id,
    safe_remove_workspace_copy,
)


def hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_generate_run_id_is_safe_and_unique() -> None:
    first = generate_run_id()
    second = generate_run_id()

    assert first != second
    assert re.fullmatch(r"[A-Za-z0-9_-]+", first)


def test_default_trace_path_is_deterministic_and_validates_run_id(tmp_path: Path) -> None:
    assert default_trace_path(tmp_path / "runs", "safe-run_1") == (
        tmp_path / "runs" / "safe-run_1" / "trace.jsonl"
    )
    assert not (tmp_path / "runs").exists()

    with pytest.raises(RunPathError, match="unsafe"):
        default_trace_path(tmp_path, "../escape")


def test_workspace_copy_preserves_paths_and_hashes(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    (seed / "nested").mkdir(parents=True)
    (seed / "a.txt").write_text("alpha", encoding="utf-8")
    (seed / "nested" / "b.txt").write_text("beta", encoding="utf-8")
    destination = tmp_path / "copies" / "workspace"

    copied = create_workspace_copy(seed, destination)

    assert copied == destination.resolve()
    assert hashes(copied) == hashes(seed)


def test_workspace_copy_rejects_conflict_and_destination_inside_seed(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(RunPathError, match="already exists"):
        create_workspace_copy(seed, destination)
    with pytest.raises(RunPathError, match="outside"):
        create_workspace_copy(seed, seed / "copy")


def test_workspace_copy_rejects_detected_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    linked = seed / "linked.txt"
    linked.write_text("content", encoding="utf-8")
    real_detector = run_paths_module.is_link_or_reparse_point
    monkeypatch.setattr(
        run_paths_module,
        "is_link_or_reparse_point",
        lambda path: path == linked or real_detector(path),
    )

    with pytest.raises(RunPathError, match="links"):
        create_workspace_copy(seed, tmp_path / "copy")


def test_safe_remove_workspace_copy_is_explicit_and_idempotent(tmp_path: Path) -> None:
    copy = tmp_path / "runtime" / "run-1" / "workspace"
    copy.mkdir(parents=True)
    (copy / "file.txt").write_text("content", encoding="utf-8")

    assert safe_remove_workspace_copy(copy) is True
    assert not copy.exists()
    assert safe_remove_workspace_copy(copy) is False


def test_run_paths_reject_detected_link_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.mkdir()
    real_detector = run_paths_module.is_link_or_reparse_point
    monkeypatch.setattr(
        run_paths_module,
        "is_link_or_reparse_point",
        lambda path: path == linked_parent or real_detector(path),
    )

    with pytest.raises(RunPathError, match="traverse links"):
        create_workspace_copy(seed, linked_parent / "copy")
