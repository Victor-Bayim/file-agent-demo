"""Opaque in-memory sessions with verified, isolated workspace copies."""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.run_paths import RunPathError, create_workspace_copy, safe_remove_workspace_copy
from app.sandbox import is_link_or_reparse_point
from app.web.config import WebSettings

LOGGER = logging.getLogger("file_agent.web.sessions")


class SessionError(RuntimeError):
    code = "SESSION_ERROR"

    def __init__(self, message: str) -> None:
        self.safe_message = message
        super().__init__(message)


class SessionCapacityError(SessionError):
    code = "SESSION_CAPACITY"


class SessionActiveRunError(SessionError):
    code = "ACTIVE_RUN"


@dataclass
class SessionRecord:
    session_id: str
    csrf_token: str
    workspace_path: Path
    runs_path: Path
    created_at: datetime
    created_monotonic: float
    last_access_at: float
    client_ip: str
    active_run_id: str | None = None
    recent_run_timestamps: list[float] = field(default_factory=list)
    workspace_revision: int = 0

    def public_status(self) -> dict[str, object]:
        return {
            "authenticated": True,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "workspace_revision": self.workspace_revision,
            "active_run_id": self.active_run_id,
        }


def _validate_link_free_tree(root: Path) -> None:
    pending = [root]
    while pending:
        directory = pending.pop()
        if is_link_or_reparse_point(directory):
            raise SessionError("Session storage contains an unsupported link.")
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if is_link_or_reparse_point(path):
                    raise SessionError("Session storage contains an unsupported link.")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif not entry.is_file(follow_symlinks=False):
                    raise SessionError("Session storage contains an unsupported entry.")


def _remove_controlled_tree(target: Path, root: Path) -> bool:
    if not os.path.lexists(target):
        return False
    resolved_root = root.resolve(strict=True)
    absolute_target = target.absolute()
    if absolute_target.parent.resolve(strict=True) != resolved_root:
        raise SessionError("Refusing to remove storage outside the controlled root.")
    if is_link_or_reparse_point(absolute_target) or not absolute_target.is_dir():
        raise SessionError("Session storage is not a regular directory.")
    _validate_link_free_tree(absolute_target)
    shutil.rmtree(absolute_target)
    return True


class SessionManager:
    """Own random session identifiers and their filesystem lifecycle."""

    def __init__(
        self,
        settings: WebSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._clock = clock
        self._sessions: dict[str, SessionRecord] = {}

    @property
    def sessions(self) -> dict[str, SessionRecord]:
        return dict(self._sessions)

    def start(self) -> None:
        self.settings.session_root.mkdir(parents=True, exist_ok=True)
        self.settings.web_runs_root.mkdir(parents=True, exist_ok=True)
        for root in (self.settings.session_root, self.settings.web_runs_root):
            if is_link_or_reparse_point(root) or not root.is_dir():
                raise SessionError("Web storage root is not a regular directory.")

    def create_session(self, *, client_ip: str) -> SessionRecord:
        self.cleanup_expired()
        self._ensure_capacity()
        now = self._clock()
        for _attempt in range(10):
            session_id = secrets.token_urlsafe(32)
            if session_id not in self._sessions:
                break
        else:
            raise SessionError("Unable to allocate a session identifier.")

        session_directory = self.settings.session_root / session_id
        workspace = session_directory / "workspace"
        runs_path = self.settings.web_runs_root / session_id
        session_directory.mkdir(parents=False, exist_ok=False)
        try:
            create_workspace_copy(self.settings.seed_workspace, workspace)
            runs_path.mkdir(parents=False, exist_ok=False)
        except Exception:
            if session_directory.exists():
                _remove_controlled_tree(session_directory, self.settings.session_root)
            if runs_path.exists():
                _remove_controlled_tree(runs_path, self.settings.web_runs_root)
            raise SessionError("Unable to initialize isolated session storage.") from None

        record = SessionRecord(
            session_id=session_id,
            csrf_token=secrets.token_urlsafe(32),
            workspace_path=workspace.resolve(strict=True),
            runs_path=runs_path.resolve(strict=True),
            created_at=datetime.now(UTC),
            created_monotonic=now,
            last_access_at=now,
            client_ip=client_ip,
        )
        self._sessions[session_id] = record
        return record

    def get(self, session_id: str | None, *, touch: bool = True) -> SessionRecord | None:
        if not session_id:
            return None
        record = self._sessions.get(session_id)
        if record is not None and touch:
            record.last_access_at = self._clock()
        return record

    def cleanup_expired(self) -> list[str]:
        now = self._clock()
        expired = [
            record.session_id
            for record in self._sessions.values()
            if record.active_run_id is None
            and now - record.last_access_at >= self.settings.session_ttl_seconds
        ]
        removed = []
        for session_id in expired:
            try:
                self.remove_session(session_id)
            except SessionError as exc:
                LOGGER.warning("session_cleanup_failed category=%s", exc.code)
                continue
            removed.append(session_id)
        return removed

    def reset_workspace(self, record: SessionRecord) -> int:
        if record.active_run_id is not None:
            raise SessionActiveRunError("Workspace cannot be reset while a run is active.")
        session_directory = record.workspace_path.parent
        temporary = session_directory / f"workspace-reset-{secrets.token_hex(8)}"
        backup = session_directory / f"workspace-old-{secrets.token_hex(8)}"
        try:
            create_workspace_copy(self.settings.seed_workspace, temporary)
            record.workspace_path.rename(backup)
            try:
                temporary.rename(record.workspace_path)
            except Exception:
                backup.rename(record.workspace_path)
                raise
            safe_remove_workspace_copy(backup)
        except (OSError, RunPathError):
            if temporary.exists() and not is_link_or_reparse_point(temporary):
                safe_remove_workspace_copy(temporary)
            raise SessionError("Workspace reset failed safely.") from None
        record.workspace_revision += 1
        record.last_access_at = self._clock()
        return record.workspace_revision

    def remove_session(self, session_id: str) -> bool:
        record = self._sessions.get(session_id)
        if record is None:
            return False
        if record.active_run_id is not None:
            raise SessionActiveRunError("Active sessions cannot be removed.")
        session_directory = record.workspace_path.parent
        _remove_controlled_tree(session_directory, self.settings.session_root)
        _remove_controlled_tree(record.runs_path, self.settings.web_runs_root)
        self._sessions.pop(session_id, None)
        return True

    def shutdown(self) -> None:
        for session_id in list(self._sessions):
            record = self._sessions[session_id]
            if record.active_run_id is not None:
                continue
            try:
                self.remove_session(session_id)
            except SessionError as exc:
                LOGGER.warning("session_shutdown_cleanup_failed category=%s", exc.code)
                continue

    def _ensure_capacity(self) -> None:
        while len(self._sessions) >= self.settings.max_sessions:
            inactive = [
                record for record in self._sessions.values() if record.active_run_id is None
            ]
            if not inactive:
                raise SessionCapacityError("No inactive Web session can be evicted.")
            oldest = min(inactive, key=lambda record: record.last_access_at)
            self.remove_session(oldest.session_id)
