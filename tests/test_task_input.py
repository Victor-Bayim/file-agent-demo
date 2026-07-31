from __future__ import annotations

import ast
from pathlib import Path
from typing import BinaryIO

import pytest

from app.cli import build_parser
from app.task_input import (
    MAX_TASK_FILE_BYTES,
    TASK_FILE_CHUNK_BYTES,
    TaskFileError,
    load_task_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_parser_accepts_task_text_only() -> None:
    args = build_parser().parse_args(["--workspace", "workspace", "--task", "Do it."])

    assert args.task == "Do it."
    assert args.task_file is None


def test_parser_accepts_task_file_only(tmp_path: Path) -> None:
    task_file = tmp_path / "task.txt"
    args = build_parser().parse_args(["--workspace", "workspace", "--task-file", str(task_file)])

    assert args.task is None
    assert args.task_file == task_file


@pytest.mark.parametrize(
    "arguments",
    [
        ["--workspace", "workspace"],
        [
            "--workspace",
            "workspace",
            "--task",
            "text",
            "--task-file",
            "task.txt",
        ],
    ],
)
def test_parser_requires_exactly_one_task_input(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as captured:
        build_parser().parse_args(arguments)

    assert captured.value.code == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("中文任务", "中文任务"),
        ('保留 "双引号"', '保留 "双引号"'),
        ("保留 '单引号'", "保留 '单引号'"),
        ("第一行\n  第二行", "第一行\n  第二行"),
        ("任务正文\n", "任务正文"),
        ("任务正文\r\n", "任务正文"),
        ("常规 Unicode ✓ café", "常规 Unicode ✓ café"),
    ],
)
def test_task_file_preserves_supported_utf8_text(
    raw: str,
    expected: str,
    tmp_path: Path,
) -> None:
    task_file = tmp_path / "task.txt"
    task_file.write_bytes(raw.encode("utf-8"))

    assert load_task_file(task_file) == expected


@pytest.mark.parametrize("raw", [b"", b" \t\r\n"])
def test_empty_or_whitespace_task_file_is_rejected(raw: bytes, tmp_path: Path) -> None:
    task_file = tmp_path / "task.txt"
    task_file.write_bytes(raw)

    with pytest.raises(TaskFileError) as captured:
        load_task_file(task_file)

    assert captured.value.code == "TASK_FILE_EMPTY"


def test_invalid_utf8_is_rejected_without_content(tmp_path: Path) -> None:
    task_file = tmp_path / "task.txt"
    task_file.write_bytes(b"private-prefix-\xff-private-suffix")

    with pytest.raises(TaskFileError) as captured:
        load_task_file(task_file)

    assert captured.value.code == "TASK_FILE_ENCODING"
    assert "private-prefix" not in str(captured.value)
    assert "private-suffix" not in str(captured.value)


def test_nul_is_rejected_without_content(tmp_path: Path) -> None:
    task_file = tmp_path / "task.txt"
    task_file.write_bytes(b"private-prefix\x00private-suffix")

    with pytest.raises(TaskFileError) as captured:
        load_task_file(task_file)

    assert captured.value.code == "TASK_FILE_NUL"
    assert "private-prefix" not in str(captured.value)
    assert "private-suffix" not in str(captured.value)


def test_oversized_task_file_is_rejected(tmp_path: Path) -> None:
    task_file = tmp_path / "task.txt"
    task_file.write_bytes(b"x" * (MAX_TASK_FILE_BYTES + 1))

    with pytest.raises(TaskFileError) as captured:
        load_task_file(task_file)

    assert captured.value.code == "TASK_FILE_TOO_LARGE"


def test_missing_task_file_is_rejected(tmp_path: Path) -> None:
    task_file = tmp_path / "missing.txt"

    with pytest.raises(TaskFileError) as captured:
        load_task_file(task_file)

    assert captured.value.code == "TASK_FILE_NOT_FOUND"


def test_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TaskFileError) as captured:
        load_task_file(tmp_path)

    assert captured.value.code == "TASK_FILE_NOT_REGULAR"


def test_symbolic_link_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("task", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symbolic links are unavailable in this environment")

    with pytest.raises(TaskFileError) as captured:
        load_task_file(link)

    assert captured.value.code == "TASK_FILE_LINK"


def test_task_file_uses_only_bounded_binary_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_file = tmp_path / "task.txt"
    task_file.write_text("streamed task", encoding="utf-8")
    original_open = Path.open
    read_sizes: list[int] = []

    class BoundedReader:
        def __init__(self, wrapped: BinaryIO) -> None:
            self.wrapped = wrapped

        def __enter__(self) -> BoundedReader:
            return self

        def __exit__(self, *args: object) -> None:
            self.wrapped.close()

        def fileno(self) -> int:
            return self.wrapped.fileno()

        def read(self, size: int) -> bytes:
            read_sizes.append(size)
            return self.wrapped.read(size)

    def guarded_open(path: Path, *args: object, **kwargs: object) -> BinaryIO | BoundedReader:
        opened = original_open(path, *args, **kwargs)
        return BoundedReader(opened) if path == task_file else opened

    monkeypatch.setattr(Path, "open", guarded_open)

    assert load_task_file(task_file) == "streamed task"
    assert read_sizes
    assert set(read_sizes) == {TASK_FILE_CHUNK_BYTES}


def test_runtime_task_loader_has_no_unbounded_read_calls() -> None:
    source_path = REPOSITORY_ROOT / "app" / "task_input.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert ".read_text(" not in source
    assert ".readlines(" not in source
    assert ".splitlines(" not in source
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read"
        ):
            assert node.args or node.keywords
