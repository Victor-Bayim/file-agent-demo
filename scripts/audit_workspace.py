"""Read-only audit utility for a workspace seed.

The target workspace is never modified.  An optional baseline file may be
written outside the audited scope so later phases can prove that the seed is
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

AUDIT_SCHEMA_VERSION = 1
BASELINE_SCHEMA_VERSION = 1
DEFAULT_PHRASE = "Project Falcon"
READ_CHUNK_SIZE = 64 * 1024


class AuditError(RuntimeError):
    """Raised when the requested audit scope is invalid or unsafe."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit files without modifying the target workspace.",
    )
    parser.add_argument(
        "workspace",
        type=Path,
        help="Workspace root to audit.",
    )
    parser.add_argument(
        "--include-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "Relative file or directory to include. Repeat to audit a seed "
            "stored alongside project files. Without this option, the entire "
            "workspace root is audited."
        ),
    )
    parser.add_argument(
        "--phrase",
        default=DEFAULT_PHRASE,
        help=f"Exact phrase to count (default: {DEFAULT_PHRASE!r}).",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--write-baseline",
        type=Path,
        help="Write a deterministic path/SHA256 baseline to this file.",
    )
    output_group.add_argument(
        "--verify-baseline",
        type=Path,
        help="Compare the audit result with an existing baseline.",
    )
    return parser.parse_args(argv)


def ensure_within_workspace(workspace: Path, candidate: Path) -> Path:
    """Resolve a path and reject paths or symlinks outside ``workspace``."""
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise AuditError(f"Included path does not exist: {candidate}") from exc

    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise AuditError(f"Included path escapes workspace: {candidate}") from exc
    return resolved


def collect_files(workspace: Path, include_roots: Sequence[Path]) -> list[Path]:
    """Return sorted regular files in the requested scope without following links."""
    workspace = workspace.resolve(strict=True)
    requested_roots = list(include_roots) or [Path(".")]
    files: dict[str, Path] = {}

    for relative_root in requested_roots:
        if relative_root.is_absolute() or ".." in relative_root.parts:
            raise AuditError(f"Include root must be workspace-relative: {relative_root}")

        raw_root = workspace / relative_root
        resolved_root = ensure_within_workspace(workspace, raw_root)
        candidates = [resolved_root] if resolved_root.is_file() else resolved_root.rglob("*")

        for candidate in candidates:
            if candidate.is_symlink():
                raise AuditError(f"Symbolic links are not allowed in audit scope: {candidate}")
            if not candidate.is_file():
                continue
            resolved_file = ensure_within_workspace(workspace, candidate)
            relative_path = resolved_file.relative_to(workspace).as_posix()
            files[relative_path] = resolved_file

    return [files[path] for path in sorted(files)]


def scan_file(path: Path, phrase: bytes) -> tuple[str, int]:
    """Stream a file once and return its SHA256 plus exact phrase count."""
    digest = hashlib.sha256()
    phrase_count = 0
    overlap = max(len(phrase) - 1, 0)
    tail = b""

    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK_SIZE):
            digest.update(chunk)
            if not phrase:
                continue

            combined = tail + chunk
            if len(combined) > overlap:
                searchable_end = len(combined) - overlap
                search_from = 0
                while True:
                    match_at = combined.find(phrase, search_from)
                    if match_at < 0 or match_at >= searchable_end:
                        break
                    phrase_count += 1
                    search_from = match_at + len(phrase)
                tail = combined[searchable_end:]
            else:
                tail = combined

    if phrase:
        phrase_count += tail.count(phrase)

    return digest.hexdigest(), phrase_count


def file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix[1:] if suffix else "no_extension"


def read_front_matter_status(path: Path) -> str | None:
    """Read a simple YAML-style ``status`` field from leading front matter."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first_line = handle.readline()
        if first_line.strip() != "---":
            return None

        for line_number, line in enumerate(handle, start=2):
            if line.strip() == "---":
                return None
            if line_number > 200:
                return None

            key, separator, value = line.partition(":")
            if separator and key.strip() == "status":
                normalized = value.strip().strip("'\"")
                return normalized or None

    return None


def normalized_include_roots(include_roots: Sequence[Path]) -> list[str]:
    roots = include_roots or [Path(".")]
    return sorted({root.as_posix() for root in roots})


def build_audit(
    workspace: Path,
    include_roots: Sequence[Path],
    phrase: str,
) -> dict[str, Any]:
    workspace = workspace.resolve(strict=True)
    files = collect_files(workspace, include_roots)
    phrase_bytes = phrase.encode("utf-8")
    directory_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    phrase_by_file: dict[str, int] = {}
    draft_statuses: dict[str, str | None] = {}
    file_records: list[dict[str, Any]] = []
    total_bytes = 0

    for path in files:
        relative_path = path.relative_to(workspace).as_posix()
        size_bytes = path.stat().st_size
        sha256, matches = scan_file(path, phrase_bytes)
        kind = file_type(path)
        parent = path.relative_to(workspace).parent.as_posix()
        directory = "_root" if parent == "." else parent

        total_bytes += size_bytes
        directory_counts[directory] += 1
        type_counts[kind] += 1
        phrase_by_file[relative_path] = matches

        if relative_path.startswith("drafts/"):
            draft_statuses[relative_path] = read_front_matter_status(path)

        file_records.append(
            {
                "path": relative_path,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "file_type": kind,
                "phrase_matches": matches,
            }
        )

    total_matches = sum(phrase_by_file.values())
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "workspace": str(workspace),
        "included_roots": normalized_include_roots(include_roots),
        "summary": {
            "total_files": len(file_records),
            "total_bytes": total_bytes,
            "files_by_directory": dict(sorted(directory_counts.items())),
            "files_by_type": dict(sorted(type_counts.items())),
        },
        "phrase_audit": {
            "phrase": phrase,
            "matching_files": sum(count > 0 for count in phrase_by_file.values()),
            "total_matches": total_matches,
            "matches_by_file": phrase_by_file,
        },
        "draft_front_matter_status": dict(sorted(draft_statuses.items())),
        "files": file_records,
    }


def baseline_from_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "included_roots": audit["included_roots"],
        "file_count": audit["summary"]["total_files"],
        "files": [
            {"path": record["path"], "sha256": record["sha256"]} for record in audit["files"]
        ],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically; this function never writes inside audited roots implicitly."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def ensure_output_outside_audit_scope(
    workspace: Path,
    include_roots: Sequence[Path],
    output: Path,
) -> None:
    """Reject baseline output paths that would modify the audited seed."""
    workspace = workspace.resolve(strict=True)
    destination = output.resolve()
    scoped_roots = include_roots or [Path(".")]
    for relative_root in scoped_roots:
        audited_root = ensure_within_workspace(workspace, workspace / relative_root)
        try:
            destination.relative_to(audited_root)
        except ValueError:
            continue
        raise AuditError(f"Baseline output is inside audited scope: {destination}")


def compare_baselines(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    expected_files = {item["path"]: item["sha256"] for item in expected["files"]}
    actual_files = {item["path"]: item["sha256"] for item in actual["files"]}
    expected_paths = set(expected_files)
    actual_paths = set(actual_files)
    modified = sorted(
        path for path in expected_paths & actual_paths if expected_files[path] != actual_files[path]
    )
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    return {
        "matches": not (modified or missing or unexpected),
        "modified": modified,
        "missing": missing,
        "unexpected": unexpected,
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"Unable to read baseline {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise AuditError(f"Invalid baseline format: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        audit = build_audit(args.workspace, args.include_root, args.phrase)

        if args.write_baseline is not None:
            ensure_output_outside_audit_scope(
                args.workspace,
                args.include_root,
                args.write_baseline,
            )
            write_json_atomic(args.write_baseline, baseline_from_audit(audit))
            audit["baseline_written"] = str(args.write_baseline.resolve())

        exit_code = 0
        if args.verify_baseline is not None:
            expected = load_json(args.verify_baseline)
            comparison = compare_baselines(expected, baseline_from_audit(audit))
            audit["baseline_verification"] = comparison
            if not comparison["matches"]:
                exit_code = 3

        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return exit_code
    except (AuditError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
