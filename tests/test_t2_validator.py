from __future__ import annotations

import hashlib
from pathlib import Path

from app.run_paths import create_workspace_copy
from scripts.validate_t2 import MANIFEST_LINES, MOVED_SOURCES, validate_t2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEED_WORKSPACE = REPOSITORY_ROOT / "workspace"
BASELINE_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "workspace_baseline.json"


def completed_copy(tmp_path: Path) -> Path:
    workspace = create_workspace_copy(SEED_WORKSPACE, tmp_path / "workspace")
    archive = workspace / "archive"
    archive.mkdir()
    for source in MOVED_SOURCES:
        source_path = workspace.joinpath(*source.split("/"))
        source_path.replace(archive / source_path.name)
    (archive / "MANIFEST.md").write_text("\n".join(MANIFEST_LINES) + "\n", encoding="utf-8")
    return workspace


def workspace_snapshot(workspace: Path) -> tuple[dict[str, str], set[str]]:
    files = {
        path.relative_to(workspace).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    directories = {
        path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_dir()
    }
    return files, directories


def test_t2_validator_accepts_exact_moves_and_manifest(tmp_path: Path) -> None:
    workspace = completed_copy(tmp_path)
    before = workspace_snapshot(workspace)

    report = validate_t2(BASELINE_PATH, workspace)

    assert report["valid"] is True
    assert all(report["checks"].values())
    assert report["manifest_lines"] == list(MANIFEST_LINES)
    assert all(line.startswith("- ") for line in report["manifest_lines"])
    assert workspace_snapshot(workspace) == before


def test_t2_validator_rejects_wrong_manifest_order_or_content(tmp_path: Path) -> None:
    workspace = completed_copy(tmp_path)
    (workspace / "archive" / "MANIFEST.md").write_text(
        "\n".join(reversed(MANIFEST_LINES)) + "\n",
        encoding="utf-8",
    )

    report = validate_t2(BASELINE_PATH, workspace)

    assert report["valid"] is False
    assert report["checks"]["manifest_exact"] is False
    assert report["checks"]["manifest_sorted"] is False


def test_t2_validator_rejects_manifest_path_prefix_or_extra_explanation(tmp_path: Path) -> None:
    workspace = completed_copy(tmp_path)
    manifest = workspace / "archive" / "MANIFEST.md"
    manifest.write_text(
        "- archive/api-v1-spec.md\n- blog-post-launch.md — moved\n- onboarding-guide.md\n",
        encoding="utf-8",
    )

    report = validate_t2(BASELINE_PATH, workspace)

    assert report["valid"] is False
    assert report["checks"]["manifest_exact"] is False


def test_t2_validator_rejects_missing_move(tmp_path: Path) -> None:
    workspace = completed_copy(tmp_path)
    destination = workspace / "archive" / "api-v1-spec.md"
    destination.replace(workspace / "drafts" / destination.name)

    report = validate_t2(BASELINE_PATH, workspace)

    assert report["valid"] is False
    assert report["checks"]["sources_absent"] is False
    assert report["checks"]["destinations_exist"] is False


def test_t2_validator_rejects_changed_moved_file(tmp_path: Path) -> None:
    workspace = completed_copy(tmp_path)
    (workspace / "archive" / "blog-post-launch.md").write_text("changed", encoding="utf-8")

    report = validate_t2(BASELINE_PATH, workspace)

    assert report["valid"] is False
    assert report["checks"]["moved_hashes_match"] is False


def test_t2_validator_rejects_active_move_and_extra_file(tmp_path: Path) -> None:
    workspace = completed_copy(tmp_path)
    active = workspace / "drafts" / "pricing-review-obsolete.md"
    active.replace(workspace / "archive" / active.name)
    (workspace / "extra.txt").write_text("extra", encoding="utf-8")

    report = validate_t2(BASELINE_PATH, workspace)

    assert report["valid"] is False
    assert report["checks"]["active_drafts_remain"] is False
    assert report["checks"]["misleading_name_remains"] is False
    assert report["checks"]["no_other_paths"] is False


def test_t2_validator_rejects_unexpected_archive_file_and_empty_directory(
    tmp_path: Path,
) -> None:
    workspace = completed_copy(tmp_path)
    (workspace / "archive" / "extra.txt").write_text("extra", encoding="utf-8")
    (workspace / "archive" / "unexpected-empty").mkdir()

    report = validate_t2(BASELINE_PATH, workspace)

    assert report["valid"] is False
    assert report["checks"]["archive_contents_exact"] is False
    assert report["checks"]["no_other_paths"] is False
    assert report["checks"]["no_other_directories"] is False
    assert report["unexpected_directories"] == ["archive/unexpected-empty"]


def test_t2_validator_ignores_manifest_trailing_newline_only(tmp_path: Path) -> None:
    workspace = completed_copy(tmp_path)
    manifest = workspace / "archive" / "MANIFEST.md"
    manifest.write_text("\n".join(MANIFEST_LINES), encoding="utf-8")

    report = validate_t2(BASELINE_PATH, workspace)

    assert report["valid"] is True
