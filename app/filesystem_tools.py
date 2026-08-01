"""Six model-independent filesystem tools with deterministic safety boundaries."""

from __future__ import annotations

import codecs
import hashlib
import heapq
import os
import stat
import tempfile
from collections import deque
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import AgentLimits
from app.runtime import MutationRecord, RunState, ToolExecutionResult
from app.sandbox import SandboxError, WorkspaceSandbox, is_link_or_reparse_point
from app.tools import ToolErrorCode, ToolHandlerError, ToolRegistry, ToolSpec

HASH_CHUNK_SIZE = 64 * 1024
BINARY_SAMPLE_SIZE = 8 * 1024


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListDirectoryArgs(ToolArguments):
    path: str = Field(default=".", description="POSIX-relative directory path.")
    recursive: bool = Field(default=False, description="Whether to include descendants.")
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum descendant depth when recursive is true.",
    )
    max_entries: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="Maximum number of directory entries to return.",
    )


class SearchTextArgs(ToolArguments):
    query: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Literal text to find within each logical line, used exactly as provided without "
            "semantic expansion."
        ),
    )
    path: str = Field(default=".", description="POSIX-relative file or directory path.")
    case_sensitive: bool = Field(default=True, description="Whether letter case must match.")
    glob: str = Field(
        default="**/*",
        min_length=1,
        max_length=500,
        description="Workspace-relative POSIX glob limiting candidate files.",
    )
    context_lines: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Logical lines to include before and after a match.",
    )
    max_results: int = Field(
        default=50,
        ge=1,
        le=200,
        description=(
            "Maximum individual matches to return; reaching the limit makes the scan "
            "incomplete and truncated."
        ),
    )
    max_snippet_chars: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum Unicode characters in each match snippet.",
    )
    exclude_paths: list[str] = Field(
        default_factory=list,
        description="Exact workspace-relative paths whose trees are skipped.",
    )

    @field_validator("query")
    @classmethod
    def reject_empty_query(cls, value: str) -> str:
        if not value:
            raise ValueError("query must not be empty")
        return value

    @field_validator("glob")
    @classmethod
    def validate_glob(cls, value: str) -> str:
        if "\x00" in value or "\\" in value or value.startswith("/"):
            raise ValueError("glob must be a relative POSIX pattern")
        if ".." in value.split("/"):
            raise ValueError("glob must not contain traversal segments")
        return value


class ReadFileArgs(ToolArguments):
    path: str = Field(description="POSIX-relative path of the UTF-8 text file.")
    start_line: int = Field(default=1, ge=1, description="First logical line to return.")
    max_lines: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="Maximum complete logical lines to return.",
    )
    max_chars: int = Field(
        default=12_000,
        ge=100,
        le=50_000,
        description="Maximum Python Unicode characters returned in content.",
    )


class CreateDirectoryArgs(ToolArguments):
    path: str = Field(description="POSIX-relative path of one directory to create.")


class WriteFileArgs(ToolArguments):
    path: str = Field(description="POSIX-relative destination file path.")
    content: str = Field(description="UTF-8 text content to write atomically.")
    overwrite: bool = Field(
        default=False,
        description="Permit replacement only after this run observed the file.",
    )


class MoveFileArgs(ToolArguments):
    source: str = Field(description="Observed POSIX-relative source file path.")
    destination: str = Field(description="New POSIX-relative destination file path.")
    require_exact_line: str | None = Field(
        default=None,
        max_length=50_000,
        description=(
            "Optional complete logical line, excluding its newline, that must exist in the source."
        ),
    )


class FilesystemToolService:
    """Handlers sharing one sandbox, run state, and deterministic limits."""

    def __init__(
        self,
        sandbox: WorkspaceSandbox,
        state: RunState,
        limits: AgentLimits,
    ) -> None:
        self.sandbox = sandbox
        self.state = state
        self.limits = limits

    def list_directory(self, arguments: BaseModel) -> ToolExecutionResult:
        args = cast(ListDirectoryArgs, arguments)
        directory = self._resolve_existing(args.path, expected_type="directory")
        entries: list[dict[str, Any]] = []
        truncated = False
        for entry in self._walk_entries(
            directory,
            recursive=args.recursive,
            max_depth=args.max_depth,
        ):
            if len(entries) >= args.max_entries:
                truncated = True
                break
            entries.append(entry)
        return _success(
            {
                "path": self.sandbox.to_relative_posix(directory),
                "entries": entries,
                "returned_entries": len(entries),
                "truncated": truncated,
            },
            f"Listed {len(entries)} entries; truncated={_json_bool(truncated)}",
        )

    def search_text(self, arguments: BaseModel) -> ToolExecutionResult:
        args = cast(SearchTextArgs, arguments)
        search_root = self._resolve_existing(args.path, expected_type="any")
        if not search_root.is_file() and not search_root.is_dir():
            raise ToolHandlerError(
                ToolErrorCode.NOT_A_FILE,
                "Search path must be a regular file or directory",
                result_summary="Search rejected: path is not a file or directory",
            )
        excludes = [self._validate_exclude(path) for path in args.exclude_paths]
        files: list[dict[str, Any]] = []
        skipped_binary: list[str] = []
        returned_matches = 0
        truncated = False

        for candidate in self._iter_candidate_files(search_root):
            relative = self.sandbox.to_relative_posix(candidate)
            if self._excluded(relative, excludes) or not _glob_matches(relative, args.glob):
                continue
            remaining = args.max_results - returned_matches
            matches, binary, reached_limit = self._search_one_file(candidate, args, remaining)
            if binary:
                skipped_binary.append(relative)
                continue
            if matches:
                files.append(
                    {
                        "path": relative,
                        "size": candidate.stat().st_size,
                        "match_count": len(matches),
                        "matches": matches,
                    }
                )
                returned_matches += len(matches)
            if reached_limit:
                truncated = True
                break

        scan_complete = not truncated
        if scan_complete:
            summary = (
                f"Literal search completed: {len(files)} matching files and "
                f"{returned_matches} matching occurrences; "
                f"returned_matches={returned_matches}; scan_complete=true; truncated=false."
            )
        else:
            summary = (
                f"Literal search stopped at the result limit: observed {len(files)} matching "
                f"files and {returned_matches} matching occurrences so far; "
                f"returned_matches={returned_matches}; scan_complete=false; truncated=true."
            )

        return _success(
            {
                "files": files,
                "returned_matches": returned_matches,
                "total_matches": returned_matches,
                "total_files": len(files),
                "truncated": truncated,
                "scan_complete": scan_complete,
                "skipped_binary": sorted(skipped_binary),
            },
            summary,
        )

    def read_file(self, arguments: BaseModel) -> ToolExecutionResult:
        """Return complete lines bounded by Unicode character and line counts."""
        args = cast(ReadFileArgs, arguments)
        path = self._resolve_existing(args.path, expected_type="file")
        relative = self.sandbox.to_relative_posix(path)
        sha256, size = self._hash_utf8_file(path)

        selected: list[str] = []
        selected_chars = 0
        total_lines = 0
        next_start_line: int | None = None
        try:
            with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
                for line_number, line in enumerate(handle, start=1):
                    total_lines = line_number
                    if line_number < args.start_line or next_start_line is not None:
                        continue
                    if len(selected) >= args.max_lines:
                        next_start_line = line_number
                        continue
                    if selected_chars + len(line) > args.max_chars:
                        if not selected:
                            raise ToolHandlerError(
                                ToolErrorCode.READ_LIMIT_EXCEEDED,
                                "The first requested logical line exceeds max_chars",
                                result_summary="Read rejected: one line exceeds max_chars",
                            )
                        next_start_line = line_number
                        continue
                    selected.append(line)
                    selected_chars += len(line)
        except UnicodeDecodeError as exc:
            raise ToolHandlerError(
                ToolErrorCode.BINARY_FILE_NOT_SUPPORTED,
                "File is not valid UTF-8 text",
                result_summary="Read rejected: binary file",
            ) from exc

        end_line = args.start_line + len(selected) - 1
        truncated = next_start_line is not None
        self.state.observe_file(relative, sha256, self._step)
        return _success(
            {
                "path": relative,
                "start_line": args.start_line,
                "end_line": end_line,
                "total_lines": total_lines,
                "truncated": truncated,
                "next_start_line": next_start_line,
                "content": "".join(selected),
                "size": size,
                "sha256": sha256,
            },
            (
                f"Read lines {args.start_line}-{end_line} of {total_lines}"
                if selected
                else f"Read 0 lines of {total_lines}"
            ),
        )

    def create_directory(self, arguments: BaseModel) -> ToolExecutionResult:
        args = cast(CreateDirectoryArgs, arguments)
        destination = args.path or "<invalid>"
        try:
            path = self._resolve_destination(args.path)
            destination = self.sandbox.to_relative_posix(path)
            self._require_parent(path)
            if os.path.lexists(path):
                self._reject_link(path)
                if path.is_dir():
                    self._record_success(
                        "create_directory",
                        None,
                        destination,
                        changed=False,
                    )
                    return _success(
                        {"path": destination, "created": False},
                        f"Directory already exists: {destination}",
                    )
                raise ToolHandlerError(
                    ToolErrorCode.TARGET_ALREADY_EXISTS,
                    "A non-directory already exists at the destination",
                    result_summary="Create directory rejected: target exists",
                )
            path.mkdir()
            self._record_success(
                "create_directory",
                None,
                destination,
                changed=True,
            )
            return _success(
                {"path": destination, "created": True},
                f"Created directory {destination}",
            )
        except ToolHandlerError as exc:
            self._record_failure("create_directory", None, destination, exc.code)
            raise
        except Exception:
            self._record_failure(
                "create_directory",
                None,
                destination,
                ToolErrorCode.INTERNAL_TOOL_ERROR.value,
            )
            raise

    def write_file(self, arguments: BaseModel) -> ToolExecutionResult:
        args = cast(WriteFileArgs, arguments)
        destination = args.path or "<invalid>"
        before_sha256: str | None = None
        temporary_path: Path | None = None
        try:
            encoded = args.content.encode("utf-8")
            if len(encoded) > self.limits.max_write_bytes:
                raise ToolHandlerError(
                    ToolErrorCode.WRITE_TOO_LARGE,
                    "UTF-8 content exceeds the configured write limit",
                    result_summary="Write rejected: content is too large",
                )

            path = self._resolve_destination(args.path)
            destination = self.sandbox.to_relative_posix(path)
            self._require_parent(path)
            exists = os.path.lexists(path)
            if exists:
                self._reject_link(path)
                if not args.overwrite:
                    raise ToolHandlerError(
                        ToolErrorCode.TARGET_ALREADY_EXISTS,
                        "Destination already exists and overwrite is false",
                        result_summary="Write rejected: target already exists",
                    )
                if not path.is_file():
                    raise ToolHandlerError(
                        ToolErrorCode.NOT_A_FILE,
                        "Overwrite destination is not a regular file",
                        result_summary="Write rejected: target is not a file",
                    )
                observation = self.state.get_observation(destination)
                if observation is None:
                    raise ToolHandlerError(
                        ToolErrorCode.SOURCE_NOT_OBSERVED,
                        "Destination must be read before it can be overwritten",
                        result_summary="Write rejected: target was not observed",
                    )
                before_sha256 = _sha256_file(path)
                if before_sha256 != observation.sha256:
                    self.state.remove_observation(destination)
                    raise ToolHandlerError(
                        ToolErrorCode.SOURCE_CHANGED,
                        "Destination changed after it was read",
                        result_summary="Write rejected: target changed after observation",
                    )
            elif args.overwrite:
                raise ToolHandlerError(
                    ToolErrorCode.PATH_NOT_FOUND,
                    "Overwrite destination does not exist",
                    result_summary="Write rejected: overwrite target does not exist",
                )

            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())

            if not exists and os.path.lexists(path):
                raise ToolHandlerError(
                    ToolErrorCode.TARGET_ALREADY_EXISTS,
                    "Destination appeared before the atomic commit",
                    result_summary="Write rejected: target appeared during write",
                )
            if exists:
                current_sha256 = _sha256_file(path)
                if current_sha256 != before_sha256:
                    self.state.remove_observation(destination)
                    raise ToolHandlerError(
                        ToolErrorCode.SOURCE_CHANGED,
                        "Destination changed before the atomic commit",
                        result_summary="Write rejected: target changed during write",
                    )
            os.replace(temporary_path, path)
            temporary_path = None

            after_sha256 = hashlib.sha256(encoded).hexdigest()
            self.state.observe_file(destination, after_sha256, self._step)
            self._record_success(
                "write_file",
                None,
                destination,
                changed=True,
                before_sha256=before_sha256,
                after_sha256=after_sha256,
            )
            return _success(
                {
                    "path": destination,
                    "bytes_written": len(encoded),
                    "sha256": after_sha256,
                    "overwritten": exists,
                },
                f"Wrote {len(encoded)} bytes to {destination}",
            )
        except ToolHandlerError as exc:
            self._record_failure(
                "write_file",
                None,
                destination,
                exc.code,
                before_sha256=before_sha256,
            )
            raise
        except Exception:
            self._record_failure(
                "write_file",
                None,
                destination,
                ToolErrorCode.INTERNAL_TOOL_ERROR.value,
                before_sha256=before_sha256,
            )
            raise
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)

    def move_file(self, arguments: BaseModel) -> ToolExecutionResult:
        args = cast(MoveFileArgs, arguments)
        source = args.source or "<invalid>"
        destination = args.destination or "<invalid>"
        before_sha256: str | None = None
        try:
            source_path = self._resolve_existing(args.source, expected_type="file")
            source = self.sandbox.to_relative_posix(source_path)
            destination_path = self._resolve_destination(args.destination)
            destination = self.sandbox.to_relative_posix(destination_path)
            if source == destination:
                raise ToolHandlerError(
                    ToolErrorCode.SAME_SOURCE_AND_DESTINATION,
                    "Source and destination must be different",
                    result_summary="Move rejected: source equals destination",
                )

            observation = self.state.get_observation(source)
            if observation is None:
                raise ToolHandlerError(
                    ToolErrorCode.SOURCE_NOT_OBSERVED,
                    "Source must be read before it can be moved",
                    result_summary="Move rejected: source was not observed",
                )
            before_sha256 = _sha256_file(source_path)
            if before_sha256 != observation.sha256:
                self.state.remove_observation(source)
                raise ToolHandlerError(
                    ToolErrorCode.SOURCE_CHANGED,
                    "Source changed after it was read",
                    result_summary="Move rejected: source changed after observation",
                )
            self._require_parent(destination_path)
            if os.path.lexists(destination_path):
                self._reject_link(destination_path)
                raise ToolHandlerError(
                    ToolErrorCode.TARGET_ALREADY_EXISTS,
                    "Destination already exists",
                    result_summary="Move rejected: destination already exists",
                )
            if args.require_exact_line is not None and not _contains_exact_line(
                source_path,
                args.require_exact_line,
            ):
                raise ToolHandlerError(
                    ToolErrorCode.PRECONDITION_FAILED,
                    "Source does not contain the required complete logical line",
                    result_summary="Move rejected: exact-line precondition failed",
                )

            os.rename(source_path, destination_path)
            after_sha256 = _sha256_file(destination_path)
            if after_sha256 != before_sha256:
                raise RuntimeError("Hash changed during same-filesystem move")
            self.state.remove_observation(source)
            self.state.observe_file(destination, after_sha256, self._step)
            self._record_success(
                "move_file",
                source,
                destination,
                changed=True,
                before_sha256=before_sha256,
                after_sha256=after_sha256,
            )
            return _success(
                {
                    "source": source,
                    "destination": destination,
                    "sha256": after_sha256,
                },
                f"Moved {source} to {destination}",
            )
        except ToolHandlerError as exc:
            self._record_failure(
                "move_file",
                source,
                destination,
                exc.code,
                before_sha256=before_sha256,
            )
            raise
        except Exception:
            self._record_failure(
                "move_file",
                source,
                destination,
                ToolErrorCode.INTERNAL_TOOL_ERROR.value,
                before_sha256=before_sha256,
            )
            raise

    @property
    def _step(self) -> int:
        return max(1, self.state.tool_calls)

    def _resolve_existing(self, path: str, *, expected_type: str) -> Path:
        try:
            return self.sandbox.resolve_existing(
                path,
                expected_type=cast(Any, expected_type),
            )
        except SandboxError as exc:
            raise ToolHandlerError(exc.code, exc.safe_message) from exc

    def _resolve_destination(self, path: str) -> Path:
        try:
            return self.sandbox.resolve_destination(path)
        except SandboxError as exc:
            raise ToolHandlerError(exc.code, exc.safe_message) from exc

    def _validate_exclude(self, path: str) -> str:
        try:
            normalized = self.sandbox.normalize_relative_path(path, allow_root=True)
            self.sandbox.resolve_destination(normalized) if normalized != "." else None
            return normalized
        except SandboxError as exc:
            raise ToolHandlerError(exc.code, exc.safe_message) from exc

    def _require_parent(self, path: Path) -> None:
        relative = self.sandbox.to_relative_posix(path.parent)
        try:
            self.sandbox.resolve_existing(relative, expected_type="directory")
        except SandboxError as exc:
            if exc.code == ToolErrorCode.PATH_NOT_FOUND.value:
                raise ToolHandlerError(
                    ToolErrorCode.PARENT_NOT_FOUND,
                    "Destination parent directory does not exist",
                    result_summary="Operation rejected: parent directory does not exist",
                ) from exc
            raise ToolHandlerError(exc.code, exc.safe_message) from exc

    def _reject_link(self, path: Path) -> None:
        if is_link_or_reparse_point(path):
            raise ToolHandlerError(
                ToolErrorCode.SYMLINK_NOT_ALLOWED,
                "Symbolic links and reparse points are not allowed",
                result_summary="Operation rejected: link target is not allowed",
            )

    def _walk_entries(
        self,
        directory: Path,
        *,
        recursive: bool,
        max_depth: int,
    ) -> Any:
        pending: list[tuple[str, Path, int]] = []

        def add_children(current: Path, depth: int) -> None:
            with os.scandir(current) as iterator:
                children = list(iterator)
            for child in children:
                child_path = Path(child.path)
                relative = self.sandbox.to_relative_posix(child_path)
                heapq.heappush(pending, (relative, child_path, depth))

        add_children(directory, 1)
        while pending:
            relative, child_path, depth = heapq.heappop(pending)
            metadata = child_path.lstat()
            linked = is_link_or_reparse_point(child_path)
            if linked:
                entry_type = "symlink"
            elif stat.S_ISREG(metadata.st_mode):
                entry_type = "file"
            elif stat.S_ISDIR(metadata.st_mode):
                entry_type = "directory"
            else:
                entry_type = "other"
            item: dict[str, Any] = {"path": relative, "type": entry_type}
            if entry_type == "file":
                item["size"] = metadata.st_size
            yield item
            if recursive and entry_type == "directory" and depth < max_depth:
                add_children(child_path, depth + 1)

    def _iter_candidate_files(self, root: Path) -> Any:
        if root.is_file():
            yield root
            return

        pending: list[tuple[str, Path]] = []

        def add_children(directory: Path) -> None:
            with os.scandir(directory) as iterator:
                children = list(iterator)
            for child in children:
                path = Path(child.path)
                relative = self.sandbox.to_relative_posix(path)
                heapq.heappush(pending, (relative, path))

        add_children(root)
        while pending:
            _, path = heapq.heappop(pending)
            metadata = path.lstat()
            if is_link_or_reparse_point(path):
                continue
            if stat.S_ISREG(metadata.st_mode):
                yield path
            elif stat.S_ISDIR(metadata.st_mode):
                add_children(path)

    def _excluded(self, relative: str, excludes: list[str]) -> bool:
        path_parts = PurePosixPath(relative).parts
        for excluded in excludes:
            if excluded == ".":
                return True
            excluded_parts = PurePosixPath(excluded).parts
            if path_parts[: len(excluded_parts)] == excluded_parts:
                return True
        return False

    def _search_one_file(
        self,
        path: Path,
        args: SearchTextArgs,
        remaining: int,
    ) -> tuple[list[dict[str, Any]], bool, bool]:
        with path.open("rb") as raw:
            sample = raw.read(BINARY_SAMPLE_SIZE)
        if b"\x00" in sample:
            return [], True, False

        previous: deque[str] = deque(maxlen=args.context_lines)
        pending: list[dict[str, Any]] = []
        matches: list[dict[str, Any]] = []
        reached_limit = False
        try:
            with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
                for line_number, line in enumerate(handle, start=1):
                    logical_line = line.rstrip("\r\n")
                    context_fragment = _clip_context(logical_line, args.max_snippet_chars)
                    for pending_match in pending:
                        if pending_match["_remaining"] > 0:
                            pending_match["_parts"].append(context_fragment)
                            pending_match["_remaining"] -= 1

                    if not reached_limit:
                        haystack = logical_line if args.case_sensitive else logical_line.casefold()
                        needle = args.query if args.case_sensitive else args.query.casefold()
                        for position in _non_overlapping_positions(haystack, needle):
                            match = {
                                "line": line_number,
                                "_parts": [
                                    *previous,
                                    _focus_fragment(
                                        logical_line,
                                        position,
                                        len(args.query),
                                        args.max_snippet_chars,
                                    ),
                                ],
                                "_remaining": args.context_lines,
                            }
                            matches.append(match)
                            pending.append(match)
                            if len(matches) >= remaining:
                                reached_limit = True
                                break

                    previous.append(context_fragment)
                    if reached_limit and all(item["_remaining"] == 0 for item in pending):
                        break
        except UnicodeDecodeError:
            return [], True, False

        rendered = [
            {
                "line": item["line"],
                "snippet": _bounded_text("\n".join(item["_parts"]), args.max_snippet_chars),
            }
            for item in matches
        ]
        return rendered, False, reached_limit

    def _hash_utf8_file(self, path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        size = 0
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(HASH_CHUNK_SIZE):
                    if b"\x00" in chunk:
                        raise ToolHandlerError(
                            ToolErrorCode.BINARY_FILE_NOT_SUPPORTED,
                            "File contains binary NUL bytes",
                            result_summary="Read rejected: binary file",
                        )
                    size += len(chunk)
                    digest.update(chunk)
                    decoder.decode(chunk, final=False)
                decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ToolHandlerError(
                ToolErrorCode.BINARY_FILE_NOT_SUPPORTED,
                "File is not valid UTF-8 text",
                result_summary="Read rejected: binary file",
            ) from exc
        return digest.hexdigest(), size

    def _record_success(
        self,
        operation: str,
        source: str | None,
        destination: str,
        *,
        changed: bool,
        before_sha256: str | None = None,
        after_sha256: str | None = None,
    ) -> None:
        self.state.record_mutation(
            MutationRecord(
                step=self._step,
                operation=operation,
                source=source,
                destination=destination,
                status="succeeded",
                changed=changed,
                before_sha256=before_sha256,
                after_sha256=after_sha256,
            )
        )

    def _record_failure(
        self,
        operation: str,
        source: str | None,
        destination: str,
        error_code: str,
        *,
        before_sha256: str | None = None,
    ) -> None:
        self.state.record_mutation(
            MutationRecord(
                step=self._step,
                operation=operation,
                source=source,
                destination=destination,
                status="failed",
                changed=False,
                before_sha256=before_sha256,
                after_sha256=None,
                error_code=error_code,
            )
        )


def build_filesystem_registry(
    sandbox: WorkspaceSandbox,
    state: RunState,
    limits: AgentLimits,
) -> ToolRegistry:
    """Build the six filesystem tools in stable model-schema order."""
    service = FilesystemToolService(sandbox, state, limits)
    registry = ToolRegistry(state=state)
    registrations = [
        (
            ToolSpec(
                name="list_directory",
                description=(
                    "List bounded metadata for a workspace directory before choosing unknown paths."
                ),
                args_model=ListDirectoryArgs,
                is_mutating=False,
            ),
            service.list_directory,
        ),
        (
            ToolSpec(
                name="search_text",
                description=(
                    "Stream-search literal text in UTF-8 workspace files, using query exactly as "
                    "provided without semantic expansion. Top-level total_files counts files with "
                    "at least one match, with each file counted once even when it has multiple "
                    "matches. total_matches counts all matching occurrences. total_files and "
                    "total_matches are complete when scan_complete=true. returned_matches is the "
                    "number of individual matches "
                    "returned. truncated=false means no tool result limit truncated the scan. "
                    "Count-only tasks with a complete scan usually require no file reads. Do not "
                    "automatically broaden a completed exact query."
                ),
                args_model=SearchTextArgs,
                is_mutating=False,
            ),
            service.search_text,
        ),
        (
            ToolSpec(
                name="read_file",
                description=(
                    "Read a bounded range from one UTF-8 workspace file and record its version "
                    "for any later overwrite or move."
                ),
                args_model=ReadFileArgs,
                is_mutating=False,
            ),
            service.read_file,
        ),
        (
            ToolSpec(
                name="create_directory",
                description="Create one workspace directory whose safe parent already exists.",
                args_model=CreateDirectoryArgs,
                is_mutating=True,
            ),
            service.create_directory,
        ),
        (
            ToolSpec(
                name="write_file",
                description=(
                    "Create one new UTF-8 file, or safely replace an existing file only after it "
                    "was read and overwrite is explicitly enabled. The content argument is the "
                    "complete file content: the tool adds no headings, blank lines, Markdown, or "
                    "explanations. For an exact format, content must contain only what the user "
                    "requested. A successful write confirms the filesystem commit, not that the "
                    "business structure or content satisfies the task; read the file afterward "
                    "to validate it. Before correcting an existing file, read it to establish the "
                    "current observation; call write_file with overwrite=true, then read it again "
                    "to verify the correction."
                ),
                args_model=WriteFileArgs,
                is_mutating=True,
            ),
            service.write_file,
        ),
        (
            ToolSpec(
                name="move_file",
                description=(
                    "Move one previously read file to a new destination, optionally requiring an "
                    "exact source line as a deterministic precondition."
                ),
                args_model=MoveFileArgs,
                is_mutating=True,
            ),
            service.move_file,
        ),
    ]
    for spec, handler in registrations:
        registry.register(spec, handler)
    return registry


def _success(data: dict[str, Any], summary: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        ok=True,
        data=data,
        trust="untrusted_workspace_data",
        result_summary=summary,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_exact_line(path: Path, required: str) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
            return any(line.rstrip("\r\n") == required for line in handle)
    except UnicodeDecodeError as exc:
        raise ToolHandlerError(
            ToolErrorCode.BINARY_FILE_NOT_SUPPORTED,
            "Exact-line preconditions require a UTF-8 text file",
            result_summary="Move rejected: source is binary",
        ) from exc


def _glob_matches(path: str, pattern: str) -> bool:
    candidate = PurePosixPath(path)
    if candidate.match(pattern):
        return True
    return pattern.startswith("**/") and candidate.match(pattern[3:])


def _non_overlapping_positions(haystack: str, needle: str) -> Any:
    start = 0
    while True:
        position = haystack.find(needle, start)
        if position < 0:
            return
        yield position
        start = position + len(needle)


def _focus_fragment(line: str, position: int, query_length: int, limit: int) -> str:
    if len(line) <= limit:
        return line
    available = max(0, limit - query_length)
    start = max(0, position - available // 2)
    end = min(len(line), start + limit)
    start = max(0, end - limit)
    fragment = line[start:end]
    if start > 0 and fragment:
        fragment = f"…{fragment[1:]}"
    if end < len(line) and fragment:
        fragment = f"{fragment[:-1]}…"
    return fragment


def _clip_context(line: str, limit: int) -> str:
    if len(line) <= limit:
        return line
    return f"{line[: limit - 1]}…"


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}…"


def _json_bool(value: bool) -> str:
    return "true" if value else "false"
