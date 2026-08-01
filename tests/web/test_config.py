from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

import app.web.config as config_module
from app.web.config import WebConfigurationError, WebSettings


def test_environment_requires_nonempty_access_code(seed_workspace: Path, tmp_path: Path) -> None:
    environ = {
        "FILE_AGENT_WEB_SEED_WORKSPACE": str(seed_workspace),
        "FILE_AGENT_WEB_SESSION_ROOT": str(tmp_path / "sessions"),
        "FILE_AGENT_WEB_RUNS_ROOT": str(tmp_path / "runs"),
    }

    with pytest.raises(WebConfigurationError) as captured:
        WebSettings.from_environment(environ)

    assert "access_code" in str(captured.value)
    assert str(seed_workspace) not in str(captured.value)


def test_secret_repr_and_json_do_not_expose_value(web_settings: WebSettings) -> None:
    secret = web_settings.access_code.get_secret_value()

    assert secret not in repr(web_settings)
    assert secret not in web_settings.model_dump_json()
    assert "**********" in repr(web_settings)


@pytest.mark.parametrize("field", ["session_root", "web_runs_root"])
def test_storage_cannot_overlap_seed(seed_workspace: Path, tmp_path: Path, field: str) -> None:
    payload = {
        "seed_workspace": seed_workspace,
        "session_root": tmp_path / "sessions",
        "web_runs_root": tmp_path / "runs",
        "access_code": "test",
        field: seed_workspace / "nested",
    }

    with pytest.raises(ValidationError, match="must not overlap"):
        WebSettings(**payload)


def test_storage_roots_cannot_overlap(seed_workspace: Path, tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="must not overlap"):
        WebSettings(
            seed_workspace=seed_workspace,
            session_root=tmp_path / "storage",
            web_runs_root=tmp_path / "storage" / "runs",
            access_code="test",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("port", 0), ("max_sessions", 0), ("max_task_chars", 0), ("web_max_tool_calls", 0)],
)
def test_positive_numeric_limits(
    seed_workspace: Path,
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    payload = {
        "seed_workspace": seed_workspace,
        "session_root": tmp_path / "sessions",
        "web_runs_root": tmp_path / "runs",
        "access_code": "test",
        field: value,
    }
    with pytest.raises(ValidationError):
        WebSettings(**payload)


def test_import_has_no_environment_or_filesystem_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FILE_AGENT_WEB_ACCESS_CODE", raising=False)
    reloaded = importlib.reload(config_module)

    assert reloaded.WebSettings is not None
