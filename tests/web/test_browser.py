from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.web.browser import BrowserError, list_workspace_tree, read_workspace_file


def test_tree_and_bounded_file_pages(seed_workspace: Path) -> None:
    entries = list_workspace_tree(seed_workspace)
    first = read_workspace_file(seed_workspace, path="notes/alpha.txt", max_lines=2)
    second = read_workspace_file(
        seed_workspace,
        path="notes/alpha.txt",
        start_line=int(first["next_start_line"]),
        max_lines=2,
    )

    assert [entry["path"] for entry in entries] == [
        "notes",
        "notes/alpha.txt",
        "root.txt",
    ]
    assert all(not str(entry["path"]).startswith(str(seed_workspace)) for entry in entries)
    assert first["content"] == "one\ntwo\n"
    assert first["truncated"] is True
    assert second["content"] == "three\n"
    assert second["truncated"] is False
    assert first["sha256"] == second["sha256"]


@pytest.mark.parametrize("path", ["../root.txt", "/root.txt", "notes\\alpha.txt", "C:/x"])
def test_browser_rejects_unsafe_paths(seed_workspace: Path, path: str) -> None:
    with pytest.raises(BrowserError) as captured:
        read_workspace_file(seed_workspace, path=path)
    assert captured.value.code in {"INVALID_PATH", "PATH_OUTSIDE_WORKSPACE"}
    assert str(seed_workspace) not in captured.value.safe_message


def test_browser_rejects_binary(seed_workspace: Path) -> None:
    (seed_workspace / "binary.bin").write_bytes(b"a\x00b")
    with pytest.raises(BrowserError) as captured:
        read_workspace_file(seed_workspace, path="binary.bin")
    assert captured.value.code == "BINARY_FILE_NOT_SUPPORTED"


def test_large_file_page_is_bounded_by_character_limit(seed_workspace: Path) -> None:
    large = seed_workspace / "large.log"
    large.write_text("".join(f"{number:04d}-{'x' * 100}\n" for number in range(400)))

    page = read_workspace_file(seed_workspace, path="large.log", max_lines=300)

    assert len(page["content"]) <= 20_000
    assert page["truncated"] is True
    assert page["next_start_line"] is not None
    assert page["total_lines"] == 400


def test_browser_rejects_symbolic_link(seed_workspace: Path, tmp_path: Path) -> None:
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    link = seed_workspace / "linked.txt"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links are unavailable on this platform")

    with pytest.raises(BrowserError) as captured:
        read_workspace_file(seed_workspace, path="linked.txt")
    assert captured.value.code == "SYMLINK_NOT_ALLOWED"
