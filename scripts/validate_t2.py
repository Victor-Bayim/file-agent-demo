"""Deterministic acceptance validator for the seed workspace T2 task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

MOVED_SOURCES = (
    "drafts/api-v1-spec.md",
    "drafts/blog-post-launch.md",
    "drafts/onboarding-guide.md",
)
ACTIVE_DRAFTS = (
    "drafts/design-tokens.md",
    "drafts/pricing-review-obsolete.md",
    "drafts/retention-policy.md",
    "drafts/roadmap-2026.md",
    "drafts/runbook-backup.md",
)
MANIFEST_PATH = "archive/MANIFEST.md"
MANIFEST_LINES = tuple(f"- {Path(path).name}" for path in MOVED_SOURCES)
HASH_CHUNK_SIZE = 64 * 1024
REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _is_link(path: Path) -> bool:
    metadata = path.lstat()
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & REPARSE_POINT_ATTRIBUTE)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> tuple[dict[str, str], set[str]]:
    files: dict[str, str] = {}
    directories: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        if _is_link(directory):
            raise ValueError("workspace contains a link or reparse point")
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if _is_link(path):
                    raise ValueError("workspace contains a link or reparse point")
                if entry.is_dir(follow_symlinks=False):
                    directories.add(path.relative_to(root).as_posix())
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files[path.relative_to(root).as_posix()] = _hash_file(path)
                else:
                    raise ValueError("workspace contains an unsupported entry")
    return dict(sorted(files.items())), directories


def _load_baseline(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise ValueError("invalid baseline format")
    return value


def _expected_directories(paths: set[str]) -> set[str]:
    directories: set[str] = set()
    for value in paths:
        parent = PurePosixPath(value).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def validate_t2(baseline_path: Path, workspace: Path) -> dict[str, Any]:
    baseline = _load_baseline(baseline_path)
    baseline_hashes = {item["path"]: item["sha256"] for item in baseline["files"]}
    actual, actual_directories = _inventory(workspace)
    moved_destinations = {source: f"archive/{Path(source).name}" for source in MOVED_SOURCES}
    expected_hashes = {
        (moved_destinations.get(path) or path): sha256 for path, sha256 in baseline_hashes.items()
    }
    expected_paths = set(expected_hashes) | {MANIFEST_PATH}
    expected_directories = _expected_directories(expected_paths)
    expected_archive_paths = {
        MANIFEST_PATH,
        *moved_destinations.values(),
    }
    manifest = workspace / MANIFEST_PATH
    manifest_lines = (
        manifest.read_text(encoding="utf-8").rstrip("\r\n").splitlines()
        if manifest.is_file()
        else []
    )

    moved_hashes_match = all(
        source not in actual and actual.get(destination) == baseline_hashes[source]
        for source, destination in moved_destinations.items()
    )
    unchanged_paths = {
        path: actual.get(path) == sha256
        for path, sha256 in baseline_hashes.items()
        if path not in MOVED_SOURCES
    }
    checks = {
        "sources_absent": all(source not in actual for source in MOVED_SOURCES),
        "destinations_exist": all(
            destination in actual for destination in moved_destinations.values()
        ),
        "moved_hashes_match": moved_hashes_match,
        "active_drafts_remain": all(path in actual for path in ACTIVE_DRAFTS),
        "misleading_name_remains": "drafts/pricing-review-obsolete.md" in actual,
        "manifest_exists": manifest.is_file(),
        "manifest_exact": manifest_lines == list(MANIFEST_LINES),
        "manifest_sorted": manifest_lines == sorted(manifest_lines),
        "archive_contents_exact": {path for path in actual if path.startswith("archive/")}
        == expected_archive_paths,
        "unchanged_files_match": all(unchanged_paths.values()),
        "no_other_paths": set(actual) == expected_paths,
        "no_other_directories": actual_directories == expected_directories,
    }
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "validator": "t2",
        "valid": not errors,
        "checks": checks,
        "errors": errors,
        "manifest_lines": manifest_lines,
        "modified_unchanged_paths": [
            path for path, passed in unchanged_paths.items() if not passed
        ],
        "unexpected_paths": sorted(set(actual) - expected_paths),
        "missing_paths": sorted(expected_paths - set(actual)),
        "unexpected_directories": sorted(actual_directories - expected_directories),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a completed T2 workspace copy.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("workspace", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = validate_t2(args.baseline, args.workspace)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"validator": "t2", "valid": False, "errors": [type(exc).__name__]}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
