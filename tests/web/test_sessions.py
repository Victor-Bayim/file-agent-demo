from __future__ import annotations

import re
from pathlib import Path

import pytest

import app.web.sessions as sessions_module
from app.web.config import WebSettings
from app.web.sessions import SessionActiveRunError, SessionCapacityError, SessionManager


def test_sessions_have_independent_verified_workspaces(
    web_settings: WebSettings,
    seed_workspace: Path,
) -> None:
    manager = SessionManager(web_settings)
    manager.start()
    first = manager.create_session(client_ip="127.0.0.1")
    second = manager.create_session(client_ip="127.0.0.2")

    assert first.workspace_path != second.workspace_path
    assert first.runs_path != second.runs_path
    assert re.fullmatch(r"[A-Za-z0-9_-]{40,}", first.session_id)
    (first.workspace_path / "root.txt").write_text("changed\n", encoding="utf-8")
    assert (second.workspace_path / "root.txt").read_text(encoding="utf-8") == "root\n"
    assert (seed_workspace / "root.txt").read_text(encoding="utf-8") == "root\n"
    manager.shutdown()


def test_expiration_skips_active_sessions(web_settings: WebSettings) -> None:
    now = [0.0]
    settings = web_settings.model_copy(update={"session_ttl_seconds": 10})
    manager = SessionManager(settings, clock=lambda: now[0])
    manager.start()
    expired = manager.create_session(client_ip="one")
    active = manager.create_session(client_ip="two")
    active.active_run_id = "active-run"
    now[0] = 11.0

    removed = manager.cleanup_expired()

    assert removed == [expired.session_id]
    assert manager.get(active.session_id, touch=False) is active
    active.active_run_id = None
    manager.shutdown()


def test_capacity_evicts_oldest_inactive_and_rejects_all_active(
    web_settings: WebSettings,
) -> None:
    now = [0.0]
    settings = web_settings.model_copy(update={"max_sessions": 1})
    manager = SessionManager(settings, clock=lambda: now[0])
    manager.start()
    first = manager.create_session(client_ip="one")
    first.active_run_id = "run"

    with pytest.raises(SessionCapacityError):
        manager.create_session(client_ip="two")

    first.active_run_id = None
    now[0] = 1.0
    second = manager.create_session(client_ip="two")
    assert manager.get(first.session_id, touch=False) is None
    assert manager.get(second.session_id, touch=False) is second
    manager.shutdown()


def test_reset_restores_seed_without_affecting_other_session(
    web_settings: WebSettings,
) -> None:
    manager = SessionManager(web_settings)
    manager.start()
    first = manager.create_session(client_ip="one")
    second = manager.create_session(client_ip="two")
    (first.workspace_path / "generated.txt").write_text("output", encoding="utf-8")
    (second.workspace_path / "other.txt").write_text("other", encoding="utf-8")

    revision = manager.reset_workspace(first)

    assert revision == 1
    assert not (first.workspace_path / "generated.txt").exists()
    assert (second.workspace_path / "other.txt").is_file()
    assert (first.workspace_path / "root.txt").read_text(encoding="utf-8") == "root\n"
    manager.shutdown()


def test_reset_refuses_an_active_session(web_settings: WebSettings) -> None:
    manager = SessionManager(web_settings)
    manager.start()
    session = manager.create_session(client_ip="one")
    session.active_run_id = "active-run"

    with pytest.raises(SessionActiveRunError, match="while a run is active"):
        manager.reset_workspace(session)

    session.active_run_id = None
    manager.shutdown()


def test_shutdown_contains_cleanup_failure_and_clears_memory(
    web_settings: WebSettings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = SessionManager(web_settings)
    manager.start()
    manager.create_session(client_ip="one")

    def fail_cleanup(target: Path, root: Path) -> bool:
        del target, root
        raise OSError("unsafe detail must not be logged")

    monkeypatch.setattr(sessions_module, "_remove_controlled_tree", fail_cleanup)
    manager.shutdown()

    assert manager.sessions == {}
    assert "category=OS_ERROR" in caplog.text
    assert "unsafe detail" not in caplog.text
