from __future__ import annotations

import os
from pathlib import Path
from typing import TextIO

import pytest

from app.env_loader import (
    MAX_ENV_FILE_BYTES,
    MAX_ENV_LINE_CHARS,
    EnvFileError,
    load_project_env,
)


def test_missing_env_file_is_a_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = "FILE_AGENT_MISSING_ENV_TEST"
    monkeypatch.delenv(name, raising=False)

    load_project_env(tmp_path / ".env")

    assert name not in os.environ


def test_empty_lines_comments_assignments_whitespace_and_empty_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "FILE_AGENT_PLAIN_TEST",
        "FILE_AGENT_SPACED_TEST",
        "FILE_AGENT_EMPTY_TEST",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n   \n# comment\nFILE_AGENT_PLAIN_TEST=plain\n"
        " FILE_AGENT_SPACED_TEST = spaced value \nFILE_AGENT_EMPTY_TEST=\n",
        encoding="utf-8",
    )

    load_project_env(env_path)

    assert os.environ["FILE_AGENT_PLAIN_TEST"] == "plain"
    assert os.environ["FILE_AGENT_SPACED_TEST"] == "spaced value"
    assert os.environ["FILE_AGENT_EMPTY_TEST"] == ""


def test_single_and_double_quoted_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FILE_AGENT_SINGLE_QUOTE_TEST", raising=False)
    monkeypatch.delenv("FILE_AGENT_DOUBLE_QUOTE_TEST", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "FILE_AGENT_SINGLE_QUOTE_TEST='single value'\n"
        'FILE_AGENT_DOUBLE_QUOTE_TEST="double value"\n',
        encoding="utf-8",
    )

    load_project_env(env_path)

    assert os.environ["FILE_AGENT_SINGLE_QUOTE_TEST"] == "single value"
    assert os.environ["FILE_AGENT_DOUBLE_QUOTE_TEST"] == "double value"


def test_existing_environment_wins_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "FILE_AGENT_PRIORITY_TEST"
    monkeypatch.setenv(name, "process-placeholder")
    env_path = tmp_path / ".env"
    env_path.write_text(f"{name}=file-placeholder\n", encoding="utf-8")

    load_project_env(env_path)

    assert os.environ[name] == "process-placeholder"


def test_override_true_replaces_existing_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "FILE_AGENT_OVERRIDE_TEST"
    monkeypatch.setenv(name, "process-placeholder")
    env_path = tmp_path / ".env"
    env_path.write_text(f"{name}=file-placeholder\n", encoding="utf-8")

    load_project_env(env_path, override=True)

    assert os.environ[name] == "file-placeholder"


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("1INVALID=value", "invalid variable name"),
        ("MISSING_EQUALS", "expected KEY=VALUE"),
        ("export EXPORTED=value", "export syntax is not supported"),
        ("BROKEN='unterminated", "malformed quoted value"),
        ('BROKEN=unterminated"', "malformed quoted value"),
        ("INLINE=value # comment", "inline comments are not supported"),
        ("INTERPOLATED=${OTHER}", "variable interpolation is not supported"),
    ],
)
def test_unsupported_syntax_fails_safely(
    line: str,
    reason: str,
    tmp_path: Path,
) -> None:
    placeholder = "value-that-must-not-appear"
    env_path = tmp_path / ".env"
    env_path.write_text(f"SAFE={placeholder}\n{line}\n", encoding="utf-8")

    with pytest.raises(EnvFileError) as captured:
        load_project_env(env_path)

    message = str(captured.value)
    assert captured.value.line_number == 2
    assert "line 2" in message
    assert reason in message
    assert placeholder not in message
    assert "SAFE" not in os.environ


def test_invalid_utf8_fails_without_exposing_bytes(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_bytes(b"VALID=first\nSECRET=\xff\xfe\n")

    with pytest.raises(EnvFileError) as captured:
        load_project_env(env_path)

    assert captured.value.line_number == 2
    assert "not valid UTF-8" in str(captured.value)
    assert "SECRET" not in str(captured.value)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_lf_and_crlf_are_supported(
    newline: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "FILE_AGENT_NEWLINE_TEST"
    monkeypatch.delenv(name, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_bytes(f"# comment{newline}{name}=value{newline}".encode())

    load_project_env(env_path)

    assert os.environ[name] == "value"


def test_oversized_file_is_rejected(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_bytes(b"#" * (MAX_ENV_FILE_BYTES + 1))

    with pytest.raises(EnvFileError, match="byte limit"):
        load_project_env(env_path)


def test_oversized_line_is_rejected(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(f"NAME={'x' * MAX_ENV_LINE_CHARS}\n", encoding="utf-8")

    with pytest.raises(EnvFileError, match="line 1.*character limit"):
        load_project_env(env_path)


def test_nul_character_is_rejected_without_exposing_line(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("NAME=hidden\0value\n", encoding="utf-8")

    with pytest.raises(EnvFileError) as captured:
        load_project_env(env_path)

    assert "line 1" in str(captured.value)
    assert "hidden" not in str(captured.value)


def test_symbolic_link_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.env"
    target.write_text("NAME=value\n", encoding="utf-8")
    link = tmp_path / ".env"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symbolic links are unavailable in this environment")

    with pytest.raises(EnvFileError, match="symbolic links"):
        load_project_env(link)


def test_non_regular_file_is_rejected(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.mkdir()

    with pytest.raises(EnvFileError, match="not a regular file"):
        load_project_env(env_path)


def test_loader_uses_iteration_without_unbounded_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "FILE_AGENT_STREAMING_TEST"
    monkeypatch.delenv(name, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(f"{name}=value\n", encoding="utf-8")
    original_open = Path.open
    iterations = 0

    class StreamingGuard:
        def __init__(self, wrapped: TextIO) -> None:
            self.wrapped = wrapped

        def __enter__(self) -> StreamingGuard:
            return self

        def __exit__(self, *args: object) -> None:
            self.wrapped.close()

        def __iter__(self) -> StreamingGuard:
            return self

        def __next__(self) -> str:
            nonlocal iterations
            line = next(self.wrapped)
            iterations += 1
            return line

        def fileno(self) -> int:
            return self.wrapped.fileno()

        def seek(self, offset: int) -> int:
            return self.wrapped.seek(offset)

        def read(self, *args: object) -> str:
            raise AssertionError(f"unbounded read attempted: {args!r}")

        def readlines(self, *args: object) -> list[str]:
            raise AssertionError(f"readlines attempted: {args!r}")

    def guarded_open(path: Path, *args: object, **kwargs: object) -> TextIO | StreamingGuard:
        opened = original_open(path, *args, **kwargs)
        return StreamingGuard(opened) if path == env_path else opened

    monkeypatch.setattr(Path, "open", guarded_open)

    load_project_env(env_path)

    assert os.environ[name] == "value"
    assert iterations == 2
