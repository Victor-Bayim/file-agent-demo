"""Minimal, deterministic project ``.env`` loading using only the standard library."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_INTERPOLATION = re.compile(r"\$(?:[A-Za-z_]|\{)")
MAX_ENV_FILE_BYTES = 64 * 1024
MAX_ENV_LINE_CHARS = 8192


class EnvFileError(ValueError):
    """A safe parse failure that never includes source-line contents."""

    def __init__(self, line_number: int, reason: str) -> None:
        self.line_number = line_number
        self.reason = reason
        super().__init__(f"Invalid .env syntax at line {line_number}: {reason}.")


def _parse_value(raw_value: str, line_number: int) -> str:
    value = raw_value.strip()
    if not value:
        return ""

    if value[0] in {"'", '"'}:
        quote = value[0]
        if len(value) < 2 or value[-1] != quote or quote in value[1:-1]:
            raise EnvFileError(line_number, "malformed quoted value")
        parsed = value[1:-1]
    else:
        if "'" in value or '"' in value:
            raise EnvFileError(line_number, "malformed quoted value")
        if "#" in value:
            raise EnvFileError(line_number, "inline comments are not supported")
        parsed = value

    if _INTERPOLATION.search(parsed):
        raise EnvFileError(line_number, "variable interpolation is not supported")
    if parsed.endswith("\\"):
        raise EnvFileError(line_number, "multiline values are not supported")
    return parsed


def _iter_entries(file: TextIO) -> Iterator[tuple[str, str]]:
    for line_number, raw_line in enumerate(file, start=1):
        line = raw_line.rstrip("\r\n")
        if len(line) > MAX_ENV_LINE_CHARS:
            raise EnvFileError(line_number, "line exceeds the character limit")
        if "\0" in line:
            raise EnvFileError(line_number, "NUL characters are not supported")

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "export" or stripped.startswith(("export ", "export\t")):
            raise EnvFileError(line_number, "export syntax is not supported")
        if "=" not in line:
            raise EnvFileError(line_number, "expected KEY=VALUE")

        raw_name, raw_value = line.split("=", 1)
        name = raw_name.strip()
        if _VARIABLE_NAME.fullmatch(name) is None:
            raise EnvFileError(line_number, "invalid variable name")
        yield name, _parse_value(raw_value, line_number)


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _validate_file_stat(file_stat: os.stat_result) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise EnvFileError(1, "path is not a regular file")
    if file_stat.st_size > MAX_ENV_FILE_BYTES:
        raise EnvFileError(1, "file exceeds the byte limit")


def load_project_env(env_path: Path, *, override: bool = False) -> None:
    """Stream and load one bounded UTF-8 env file without exposing values."""
    try:
        path_stat = env_path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(path_stat.st_mode):
        raise EnvFileError(1, "symbolic links are not supported")
    _validate_file_stat(path_stat)

    try:
        with env_path.open("r", encoding="utf-8", newline=None) as file:
            opened_stat = os.fstat(file.fileno())
            _validate_file_stat(opened_stat)
            if not _same_file(path_stat, opened_stat):
                raise EnvFileError(1, "file changed before it was opened")

            for _name, _value in _iter_entries(file):
                pass
            if os.fstat(file.fileno()).st_size > MAX_ENV_FILE_BYTES:
                raise EnvFileError(1, "file exceeds the byte limit")

            file.seek(0)
            for name, value in _iter_entries(file):
                if override:
                    os.environ[name] = value
                else:
                    os.environ.setdefault(name, value)
    except UnicodeDecodeError as exc:
        line_number = exc.object[: exc.start].count(b"\n") + 1
        raise EnvFileError(line_number, "file is not valid UTF-8") from None
