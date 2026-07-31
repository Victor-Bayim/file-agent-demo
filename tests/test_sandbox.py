from __future__ import annotations

from pathlib import Path

import pytest

import app.sandbox as sandbox_module
from app.sandbox import SandboxError, WorkspaceSandbox


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "folder").mkdir()
    (root / "folder" / "file.txt").write_text("hello", encoding="utf-8")
    return root


def test_workspace_root_must_exist_and_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="does not exist"):
        WorkspaceSandbox(tmp_path / "missing")
    file_root = tmp_path / "file.txt"
    file_root.write_text("x", encoding="utf-8")
    with pytest.raises(SandboxError, match="must be a directory"):
        WorkspaceSandbox(file_root)


def test_workspace_root_link_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "linked-root"
    root.mkdir()
    monkeypatch.setattr(sandbox_module, "is_link_or_reparse_point", lambda path: path == root)

    with pytest.raises(SandboxError) as raised:
        WorkspaceSandbox(root)

    assert raised.value.code == "SYMLINK_NOT_ALLOWED"


def test_root_is_allowed_only_when_requested(workspace: Path) -> None:
    sandbox = WorkspaceSandbox(workspace)

    assert sandbox.normalize_relative_path(".", allow_root=True) == "."
    with pytest.raises(SandboxError) as raised:
        sandbox.normalize_relative_path(".")
    assert raised.value.code == "INVALID_PATH"


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/etc/passwd",
        "C:/Windows/system.ini",
        "C:\\Windows\\system.ini",
        "\\\\server\\share",
        "folder\\file.txt",
        "folder/../outside.txt",
        "folder/\x00.txt",
    ],
)
def test_invalid_path_forms_are_rejected(workspace: Path, path: str) -> None:
    sandbox = WorkspaceSandbox(workspace)

    with pytest.raises(SandboxError) as raised:
        sandbox.normalize_relative_path(path, allow_root=True)

    assert raised.value.code == "INVALID_PATH"


def test_normalization_returns_stable_posix_paths(workspace: Path) -> None:
    sandbox = WorkspaceSandbox(workspace)

    assert sandbox.normalize_relative_path("./folder/./file.txt") == "folder/file.txt"
    assert sandbox.normalize_relative_path("folder//file.txt") == "folder/file.txt"
    assert sandbox.to_relative_posix(workspace / "folder" / "file.txt") == "folder/file.txt"


def test_existing_path_type_checks(workspace: Path) -> None:
    sandbox = WorkspaceSandbox(workspace)

    assert sandbox.resolve_existing("folder/file.txt", expected_type="file").is_file()
    assert sandbox.resolve_existing("folder", expected_type="directory").is_dir()
    with pytest.raises(SandboxError) as not_file:
        sandbox.resolve_existing("folder", expected_type="file")
    with pytest.raises(SandboxError) as not_directory:
        sandbox.resolve_existing("folder/file.txt", expected_type="directory")

    assert not_file.value.code == "NOT_A_FILE"
    assert not_directory.value.code == "NOT_A_DIRECTORY"


def test_missing_existing_path_is_rejected(workspace: Path) -> None:
    sandbox = WorkspaceSandbox(workspace)

    with pytest.raises(SandboxError) as raised:
        sandbox.resolve_existing("missing.txt", expected_type="any")

    assert raised.value.code == "PATH_NOT_FOUND"


@pytest.mark.parametrize("requested", ["link", "link/file.txt", "folder/link.txt"])
def test_source_and_intermediate_link_components_are_rejected(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
) -> None:
    (workspace / "link").mkdir(exist_ok=True)
    (workspace / "link" / "file.txt").write_text("x", encoding="utf-8")
    (workspace / "folder" / "link.txt").write_text("x", encoding="utf-8")
    real_detector = sandbox_module.is_link_or_reparse_point
    monkeypatch.setattr(
        sandbox_module,
        "is_link_or_reparse_point",
        lambda path: path.name in {"link", "link.txt"} or real_detector(path),
    )
    sandbox = WorkspaceSandbox(workspace)

    with pytest.raises(SandboxError) as raised:
        sandbox.resolve_existing(requested, expected_type="any")

    assert raised.value.code == "SYMLINK_NOT_ALLOWED"


def test_destination_parent_link_is_rejected(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked_parent = workspace / "linked-parent"
    linked_parent.mkdir()
    real_detector = sandbox_module.is_link_or_reparse_point
    monkeypatch.setattr(
        sandbox_module,
        "is_link_or_reparse_point",
        lambda path: path == linked_parent or real_detector(path),
    )
    sandbox = WorkspaceSandbox(workspace)

    with pytest.raises(SandboxError) as raised:
        sandbox.resolve_destination("linked-parent/new.txt")

    assert raised.value.code == "SYMLINK_NOT_ALLOWED"


def test_adjacent_prefix_directory_is_not_inside_workspace(workspace: Path) -> None:
    sandbox = WorkspaceSandbox(workspace)
    adjacent = workspace.parent / f"{workspace.name}-other" / "file.txt"

    with pytest.raises(SandboxError) as raised:
        sandbox.ensure_within_workspace(adjacent)

    assert raised.value.code == "PATH_OUTSIDE_WORKSPACE"
