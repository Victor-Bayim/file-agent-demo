"""Run identifiers, trace locations, and verified workspace-copy lifecycle."""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path

from app.sandbox import is_link_or_reparse_point

HASH_CHUNK_SIZE = 64 * 1024


class RunPathError(ValueError):
    """Raised when a run path or workspace copy violates a safety boundary."""


def generate_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{secrets.token_hex(6)}"


def default_trace_path(runs_dir: Path, run_id: str) -> Path:
    safe_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not run_id or any(character not in safe_characters for character in run_id):
        raise RunPathError("run_id contains unsafe path characters")
    return runs_dir / run_id / "trace.jsonl"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_inventory(root: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        if is_link_or_reparse_point(directory):
            raise RunPathError("workspace copies cannot contain links or reparse points")
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if is_link_or_reparse_point(path):
                    raise RunPathError("workspace copies cannot contain links or reparse points")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    inventory[path.relative_to(root).as_posix()] = _hash_file(path)
                else:
                    raise RunPathError("workspace contains an unsupported filesystem entry")
    return dict(sorted(inventory.items()))


def _reject_link_ancestors(path: Path) -> None:
    current = path
    while current.parent != current:
        if os.path.lexists(current) and is_link_or_reparse_point(current):
            raise RunPathError("run paths cannot traverse links or reparse points")
        current = current.parent


def create_workspace_copy(seed: Path, destination: Path) -> Path:
    """Copy a link-free seed and verify every relative file path and digest."""
    source = seed.absolute()
    target = destination.absolute()
    if not source.exists() or not source.is_dir():
        raise RunPathError("seed workspace must be an existing directory")
    if is_link_or_reparse_point(source):
        raise RunPathError("seed workspace must not be a link")
    _reject_link_ancestors(source.parent)
    _reject_link_ancestors(target.parent)
    source = source.resolve(strict=True)
    target = target.resolve(strict=False)
    try:
        target.relative_to(source)
    except ValueError:
        pass
    else:
        raise RunPathError("destination must be outside the seed workspace")
    if os.path.lexists(target):
        raise RunPathError("destination already exists")
    expected = _validated_inventory(source)
    try:
        shutil.copytree(source, target, symlinks=True)
        actual = _validated_inventory(target)
        if actual != expected:
            raise RunPathError("workspace copy verification failed")
    except Exception:
        if target.exists() and not is_link_or_reparse_point(target):
            shutil.rmtree(target)
        raise
    return target.resolve(strict=True)


def safe_remove_workspace_copy(destination: Path) -> bool:
    """Remove one explicit, link-free copy directory; missing paths are idempotent."""
    target = destination.absolute()
    if not os.path.lexists(target):
        return False
    if target.parent == target or len(target.parts) < 3:
        raise RunPathError("refusing to remove a broad filesystem path")
    _reject_link_ancestors(target.parent)
    if is_link_or_reparse_point(target) or not target.is_dir():
        raise RunPathError("workspace copy must be a regular directory")
    _validated_inventory(target)
    shutil.rmtree(target)
    return True
