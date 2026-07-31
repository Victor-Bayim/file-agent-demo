from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from scripts import audit_workspace

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEED_WORKSPACE = REPOSITORY_ROOT / "workspace"
BASELINE_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "workspace_baseline.json"


@pytest.fixture(scope="session")
def seed_workspace() -> Path:
    """Return the repository's immutable, clean workspace seed."""
    assert SEED_WORKSPACE.is_dir()
    return SEED_WORKSPACE


@pytest.fixture(scope="session")
def workspace_baseline() -> dict[str, Any]:
    return audit_workspace.load_json(BASELINE_PATH)


@pytest.fixture
def workspace_copy(seed_workspace: Path, tmp_path: Path) -> Iterator[Path]:
    """Give mutating tests an isolated copy, never the repository seed."""
    destination = tmp_path / "workspace"
    shutil.copytree(seed_workspace, destination)
    yield destination


def audit_entire_workspace(workspace: Path) -> dict[str, Any]:
    return audit_workspace.build_audit(
        workspace,
        [],
        audit_workspace.DEFAULT_PHRASE,
    )


def compare_to_fixture(
    workspace: Path,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    audit = audit_entire_workspace(workspace)
    return audit_workspace.compare_baselines(
        baseline,
        audit_workspace.baseline_from_audit(audit),
    )


def test_build_audit_collects_metadata_and_front_matter(tmp_path: Path) -> None:
    workspace = tmp_path / "seed"
    drafts = workspace / "drafts"
    logs = workspace / "logs"
    drafts.mkdir(parents=True)
    logs.mkdir()
    draft_file = drafts / "sample.md"
    log_file = logs / "sample.log"
    draft_file.write_text(
        "---\nstatus: obsolete\n---\nProject Falcon\n",
        encoding="utf-8",
    )
    log_file.write_text(
        "Project Falcon\nProject Falcon\n",
        encoding="utf-8",
    )

    audit = audit_workspace.build_audit(
        workspace,
        [Path("drafts"), Path("logs")],
        "Project Falcon",
    )

    assert audit["summary"] == {
        "total_files": 2,
        "total_bytes": draft_file.stat().st_size + log_file.stat().st_size,
        "files_by_directory": {"drafts": 1, "logs": 1},
        "files_by_type": {"log": 1, "md": 1},
    }
    assert audit["phrase_audit"]["matching_files"] == 2
    assert audit["phrase_audit"]["total_matches"] == 3
    assert audit["draft_front_matter_status"] == {
        "drafts/sample.md": "obsolete",
    }
    assert [record["path"] for record in audit["files"]] == [
        "drafts/sample.md",
        "logs/sample.log",
    ]


def test_scan_file_counts_phrase_across_chunk_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit_workspace, "READ_CHUNK_SIZE", 8)
    path = tmp_path / "boundary.txt"
    path.write_bytes(b"12345Project Falcon678Project Falcon")

    _, matches = audit_workspace.scan_file(path, b"Project Falcon")

    assert matches == 2


def test_compare_baselines_reports_all_drift_categories() -> None:
    expected = {
        "files": [
            {"path": "modified.txt", "sha256": "old"},
            {"path": "missing.txt", "sha256": "same"},
        ]
    }
    actual = {
        "files": [
            {"path": "modified.txt", "sha256": "new"},
            {"path": "unexpected.txt", "sha256": "same"},
        ]
    }

    comparison = audit_workspace.compare_baselines(expected, actual)

    assert comparison == {
        "matches": False,
        "modified": ["modified.txt"],
        "missing": ["missing.txt"],
        "unexpected": ["unexpected.txt"],
    }


def test_baseline_output_inside_audited_scope_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "seed"
    workspace.mkdir()

    with pytest.raises(audit_workspace.AuditError, match="inside audited scope"):
        audit_workspace.ensure_output_outside_audit_scope(
            workspace,
            [],
            workspace / "baseline.json",
        )


def test_seed_workspace_matches_phase_zero_baseline(
    seed_workspace: Path,
    workspace_baseline: dict[str, Any],
) -> None:
    audit = audit_entire_workspace(seed_workspace)
    comparison = audit_workspace.compare_baselines(
        workspace_baseline,
        audit_workspace.baseline_from_audit(audit),
    )

    assert comparison == {
        "matches": True,
        "modified": [],
        "missing": [],
        "unexpected": [],
    }
    assert audit["included_roots"] == ["."]
    assert audit["summary"]["total_files"] == workspace_baseline["file_count"]
    assert all("\\" not in record["path"] for record in audit["files"])


def test_trace_jsonl_is_reported_as_unexpected(
    workspace_copy: Path,
    workspace_baseline: dict[str, Any],
) -> None:
    (workspace_copy / "trace.jsonl").write_text("{}\n", encoding="utf-8")

    comparison = compare_to_fixture(workspace_copy, workspace_baseline)

    assert comparison == {
        "matches": False,
        "modified": [],
        "missing": [],
        "unexpected": ["trace.jsonl"],
    }


def test_unknown_top_level_file_is_reported_as_unexpected(
    workspace_copy: Path,
    workspace_baseline: dict[str, Any],
) -> None:
    unknown_directory = workspace_copy / "unknown"
    unknown_directory.mkdir()
    (unknown_directory / "surprise.txt").write_text("unexpected\n", encoding="utf-8")

    comparison = compare_to_fixture(workspace_copy, workspace_baseline)

    assert comparison == {
        "matches": False,
        "modified": [],
        "missing": [],
        "unexpected": ["unknown/surprise.txt"],
    }


def test_repository_code_changes_do_not_affect_workspace_baseline(
    seed_workspace: Path,
    workspace_baseline: dict[str, Any],
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    copied_workspace = repository / "workspace"
    shutil.copytree(seed_workspace, copied_workspace)
    (repository / "scripts").mkdir()
    (repository / "tests").mkdir()
    (repository / "scripts" / "changed.py").write_text(
        "changed = True\n",
        encoding="utf-8",
    )
    (repository / "tests" / "changed.txt").write_text(
        "changed\n",
        encoding="utf-8",
    )

    comparison = compare_to_fixture(copied_workspace, workspace_baseline)

    assert comparison["matches"] is True
    assert comparison["unexpected"] == []


def test_path_resolving_outside_workspace_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")

    with pytest.raises(audit_workspace.AuditError, match="escapes workspace"):
        audit_workspace.ensure_within_workspace(workspace.resolve(), outside)


def test_symbolic_link_escape_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    link = MagicMock(spec=Path)
    link.resolve.return_value = outside.resolve()
    link.__str__.return_value = str(workspace / "link.txt")

    with pytest.raises(audit_workspace.AuditError, match="escapes workspace"):
        audit_workspace.ensure_within_workspace(workspace.resolve(), link)

    link.resolve.assert_called_once_with(strict=True)
