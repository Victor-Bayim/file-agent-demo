"""Canonical workspace path validation shared by every filesystem tool."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Literal

WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class SandboxError(ValueError):
    """A path failure with a stable, model-facing error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


def is_link_or_reparse_point(path: Path) -> bool:
    """Return true for symlinks and detectable Windows reparse points."""
    try:
        metadata = path.lstat()
    except OSError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & REPARSE_POINT_ATTRIBUTE)


class WorkspaceSandbox:
    """Resolve POSIX-relative paths without following workspace links."""

    def __init__(self, workspace_root: Path) -> None:
        supplied_root = workspace_root.absolute()
        if not supplied_root.exists():
            raise SandboxError("PATH_NOT_FOUND", "Workspace root does not exist")
        if is_link_or_reparse_point(supplied_root):
            raise SandboxError("SYMLINK_NOT_ALLOWED", "Workspace root must not be a link")
        if not supplied_root.is_dir():
            raise SandboxError("NOT_A_DIRECTORY", "Workspace root must be a directory")
        self._root = supplied_root.resolve(strict=True)

    @property
    def root(self) -> Path:
        return self._root

    def normalize_relative_path(self, path: str, *, allow_root: bool = False) -> str:
        """Validate and normalize one model-facing POSIX relative path."""
        if not isinstance(path, str):
            raise SandboxError("INVALID_PATH", "Path must be a string")
        if not path:
            raise SandboxError("INVALID_PATH", "Path must not be empty")
        if "\x00" in path:
            raise SandboxError("INVALID_PATH", "Path must not contain NUL characters")
        if "\\" in path:
            raise SandboxError("INVALID_PATH", "Path must use POSIX separators")
        if path.startswith("/") or WINDOWS_DRIVE_PATTERN.match(path):
            raise SandboxError("INVALID_PATH", "Path must be relative to the workspace")

        raw_parts = path.split("/")
        if ".." in raw_parts:
            raise SandboxError("INVALID_PATH", "Path traversal is not allowed")
        if any(":" in part for part in raw_parts):
            raise SandboxError("INVALID_PATH", "Path must not contain drive components")

        normalized = PurePosixPath(path).as_posix()
        if normalized == ".":
            if allow_root:
                return normalized
            raise SandboxError("INVALID_PATH", "Workspace root is not allowed here")
        if PurePosixPath(normalized).is_absolute():
            raise SandboxError("INVALID_PATH", "Path must be relative to the workspace")
        return normalized

    def resolve_existing(
        self,
        path: str,
        *,
        expected_type: Literal["file", "directory", "any"],
    ) -> Path:
        """Resolve an existing path after rejecting every link component."""
        relative = self.normalize_relative_path(path, allow_root=True)
        candidate = self._candidate(relative)
        self._ensure_no_link_components(candidate)
        if not candidate.exists():
            raise SandboxError("PATH_NOT_FOUND", "Path does not exist")
        resolved = candidate.resolve(strict=True)
        self.ensure_within_workspace(resolved)
        if expected_type == "file" and not resolved.is_file():
            raise SandboxError("NOT_A_FILE", "Path is not a regular file")
        if expected_type == "directory" and not resolved.is_dir():
            raise SandboxError("NOT_A_DIRECTORY", "Path is not a directory")
        return resolved

    def resolve_destination(self, path: str) -> Path:
        """Resolve a non-root destination and reject links in existing components."""
        relative = self.normalize_relative_path(path)
        candidate = self._candidate(relative)
        self._ensure_no_link_components(candidate)
        self.ensure_within_workspace(candidate)
        return candidate

    def ensure_within_workspace(self, path: Path) -> None:
        """Reject paths outside root using path components, never string prefixes."""
        self._ensure_lexically_within(path)

        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise SandboxError(
                "PATH_OUTSIDE_WORKSPACE",
                "Path resolves outside the workspace",
            ) from exc

    def to_relative_posix(self, path: Path) -> str:
        """Convert an internal path to a stable workspace-relative POSIX path."""
        lexical = Path(os.path.abspath(path))
        self._ensure_lexically_within(lexical)
        relative = lexical.relative_to(self._root).as_posix()
        return relative or "."

    def _candidate(self, relative: str) -> Path:
        if relative == ".":
            return self._root
        candidate = self._root.joinpath(*PurePosixPath(relative).parts)
        self._ensure_lexically_within(candidate)
        return candidate

    def _ensure_lexically_within(self, path: Path) -> None:
        lexical = Path(os.path.abspath(path))
        try:
            lexical.relative_to(self._root)
        except ValueError as exc:
            raise SandboxError(
                "PATH_OUTSIDE_WORKSPACE",
                "Path resolves outside the workspace",
            ) from exc

    def _ensure_no_link_components(self, candidate: Path) -> None:
        lexical = Path(os.path.abspath(candidate))
        try:
            relative = lexical.relative_to(self._root)
        except ValueError as exc:
            raise SandboxError(
                "PATH_OUTSIDE_WORKSPACE",
                "Path resolves outside the workspace",
            ) from exc

        current = self._root
        for part in relative.parts:
            current = current / part
            if is_link_or_reparse_point(current):
                raise SandboxError(
                    "SYMLINK_NOT_ALLOWED",
                    "Symbolic links and reparse points are not allowed",
                )
