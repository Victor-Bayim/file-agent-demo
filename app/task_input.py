"""Bounded UTF-8 task-file input for application entry points."""

from __future__ import annotations

import codecs
import os
import stat
from pathlib import Path

MAX_TASK_FILE_BYTES = 256 * 1024
TASK_FILE_CHUNK_BYTES = 64 * 1024
_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class TaskFileError(ValueError):
    """A safe task-file failure that never includes file contents."""

    def __init__(self, code: str, message: str, path: Path) -> None:
        self.code = code
        self.path = path
        super().__init__(f"{message}: {path}")


def _is_link_or_reparse_point(file_stat: os.stat_result) -> bool:
    attributes = getattr(file_stat, "st_file_attributes", 0)
    return stat.S_ISLNK(file_stat.st_mode) or bool(attributes & _REPARSE_POINT_ATTRIBUTE)


def _validate_file(file_stat: os.stat_result, path: Path) -> None:
    if _is_link_or_reparse_point(file_stat):
        raise TaskFileError("TASK_FILE_LINK", "Task file must not be a symbolic link", path)
    if not stat.S_ISREG(file_stat.st_mode):
        raise TaskFileError("TASK_FILE_NOT_REGULAR", "Task file must be a regular file", path)
    if file_stat.st_size > MAX_TASK_FILE_BYTES:
        raise TaskFileError("TASK_FILE_TOO_LARGE", "Task file exceeds the byte limit", path)


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def load_task_file(path: Path) -> str:
    """Load one regular task file with bounded reads and strict UTF-8 decoding."""
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        raise TaskFileError("TASK_FILE_NOT_FOUND", "Task file does not exist", path) from None
    _validate_file(path_stat, path)

    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    parts: list[str] = []
    total_bytes = 0
    try:
        with path.open("rb") as handle:
            opened_stat = os.fstat(handle.fileno())
            _validate_file(opened_stat, path)
            if not _same_file(path_stat, opened_stat):
                raise TaskFileError(
                    "TASK_FILE_CHANGED", "Task file changed before it was opened", path
                )
            while chunk := handle.read(TASK_FILE_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > MAX_TASK_FILE_BYTES:
                    raise TaskFileError(
                        "TASK_FILE_TOO_LARGE", "Task file exceeds the byte limit", path
                    )
                decoded = decoder.decode(chunk, final=False)
                if "\0" in decoded:
                    raise TaskFileError(
                        "TASK_FILE_NUL", "Task file must not contain NUL characters", path
                    )
                parts.append(decoded)
            final_text = decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        raise TaskFileError(
            "TASK_FILE_ENCODING", "Task file must contain valid UTF-8", path
        ) from None
    except OSError:
        raise TaskFileError("TASK_FILE_READ", "Task file could not be read", path) from None

    if "\0" in final_text:
        raise TaskFileError("TASK_FILE_NUL", "Task file must not contain NUL characters", path)
    parts.append(final_text)
    task = "".join(parts).rstrip("\r\n")
    if not task.strip():
        raise TaskFileError("TASK_FILE_EMPTY", "Task file must not be empty", path)
    return task
