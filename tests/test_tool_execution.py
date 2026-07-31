from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import AgentLimits
from app.filesystem_tools import build_filesystem_registry
from app.runtime import MutationRecord, RunState, ToolExecutionResult
from app.sandbox import WorkspaceSandbox
from app.tools import (
    ToolErrorCode,
    ToolHandlerError,
    ToolRegistry,
    ToolSpec,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "workspace_baseline.json"
EXPECTED_TOOL_NAMES = [
    "list_directory",
    "search_text",
    "read_file",
    "create_directory",
    "write_file",
    "move_file",
]


def make_filesystem_registry(root: Path) -> tuple[ToolRegistry, RunState]:
    state = RunState(
        run_id="execution-test",
        workspace_root=root,
        started_at=datetime.now(UTC),
    )
    return (
        build_filesystem_registry(WorkspaceSandbox(root), state, AgentLimits()),
        state,
    )


def test_pydantic_rejects_extra_fields_and_wrong_types(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry, _ = make_filesystem_registry(root)

    extra = registry.execute("list_directory", {"unexpected": True})
    wrong_type = registry.execute("list_directory", {"max_entries": "not-an-integer"})

    assert extra.error is not None
    assert wrong_type.error is not None
    assert extra.error.code == "INVALID_ARGUMENTS"
    assert wrong_type.error.code == "INVALID_ARGUMENTS"
    assert extra.trust == "trusted_runtime_data"


def test_unknown_tool_returns_structured_runtime_error(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry, state = make_filesystem_registry(root)

    result = registry.execute("missing_tool", {})

    assert result == ToolExecutionResult(
        ok=False,
        error={"code": "UNKNOWN_TOOL", "message": "Unknown tool: missing_tool", "details": {}},
        trust="trusted_runtime_data",
        result_summary="Unknown tool rejected: missing_tool",
    )
    assert state.tool_calls == 0  # The Agent Loop owns tool-step allocation.


def test_expected_handler_error_is_structured(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry, _ = make_filesystem_registry(root)

    result = registry.execute("read_file", {"path": "missing.txt"})

    assert not result.ok
    assert result.error is not None
    assert result.error.code == "PATH_NOT_FOUND"
    assert result.trust == "untrusted_workspace_data"
    assert result.result_summary


class ExplodingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = Field(description="A harmless test value.")


def test_unexpected_handler_error_does_not_leak_absolute_paths(tmp_path: Path) -> None:
    secret_path = tmp_path / "private" / "server.txt"

    def explode(arguments: BaseModel) -> ToolExecutionResult:
        raise RuntimeError(f"failure at {secret_path}: {arguments}")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="explode",
            description="Raise a test exception.",
            args_model=ExplodingArgs,
            is_mutating=False,
        ),
        explode,
    )

    result = registry.execute("explode", {"value": 1})
    serialized = json.dumps(result.model_dump(mode="json"))

    assert result.error is not None
    assert result.error.code == "INTERNAL_TOOL_ERROR"
    assert result.error.details == {"exception_type": "RuntimeError"}
    assert str(secret_path) not in serialized
    assert "Traceback" not in serialized


def test_debug_registry_can_retain_internal_exception_for_tests(tmp_path: Path) -> None:
    marker = tmp_path / "debug-only.txt"

    def explode(arguments: BaseModel) -> ToolExecutionResult:
        raise RuntimeError(f"debug marker: {marker}: {arguments}")

    registry = ToolRegistry(expose_internal_errors=True)
    registry.register(
        ToolSpec(
            name="explode",
            description="Raise a test exception.",
            args_model=ExplodingArgs,
            is_mutating=False,
        ),
        explode,
    )

    result = registry.execute("explode", {"value": 1})

    assert result.error is not None
    assert "debug marker" in result.error.details["exception"]
    assert marker.name in result.error.details["exception"]


def test_tool_handler_error_is_converted_without_crashing() -> None:
    def reject(arguments: BaseModel) -> ToolExecutionResult:
        raise ToolHandlerError(
            ToolErrorCode.PRECONDITION_FAILED,
            "A deterministic precondition failed",
            result_summary="Rejected by precondition",
            details={"retryable": False},
        )

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="reject",
            description="Reject a test invocation.",
            args_model=ExplodingArgs,
            is_mutating=False,
        ),
        reject,
    )

    result = registry.execute("reject", {"value": 1})

    assert result.error is not None
    assert result.error.code == "PRECONDITION_FAILED"
    assert result.error.details == {"retryable": False}


def test_filesystem_registry_has_exact_stable_schema_order(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry, _ = make_filesystem_registry(root)

    schemas = registry.model_schemas()

    assert [schema["function"]["name"] for schema in schemas] == EXPECTED_TOOL_NAMES
    assert len(schemas) == 6
    assert all(schema["type"] == "function" for schema in schemas)
    for schema in schemas:
        parameters = schema["function"]["parameters"]
        assert parameters["additionalProperties"] is False
        assert all("description" in field for field in parameters["properties"].values())
        json.dumps(schema)


def test_schemas_contain_only_json_friendly_types_and_no_seed_answers(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry, _ = make_filesystem_registry(root)
    schemas = registry.model_schemas()
    serialized = json.dumps(schemas, ensure_ascii=False)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert '"format": "path"' not in serialized
    assert "Project Falcon" not in serialized
    assert "Project Phoenix" not in serialized
    for item in baseline["files"]:
        assert item["path"] not in serialized
        assert Path(item["path"]).name not in serialized
    for top_directory in {item["path"].split("/")[0] for item in baseline["files"]}:
        assert f'"{top_directory}/' not in serialized


def test_all_required_error_codes_are_defined() -> None:
    assert {code.value for code in ToolErrorCode} == {
        "INVALID_ARGUMENTS",
        "INVALID_PATH",
        "PATH_OUTSIDE_WORKSPACE",
        "PATH_NOT_FOUND",
        "NOT_A_FILE",
        "NOT_A_DIRECTORY",
        "SYMLINK_NOT_ALLOWED",
        "BINARY_FILE_NOT_SUPPORTED",
        "READ_LIMIT_EXCEEDED",
        "WRITE_TOO_LARGE",
        "TARGET_ALREADY_EXISTS",
        "PARENT_NOT_FOUND",
        "SOURCE_NOT_OBSERVED",
        "SOURCE_CHANGED",
        "PRECONDITION_FAILED",
        "SAME_SOURCE_AND_DESTINATION",
        "UNKNOWN_TOOL",
        "INTERNAL_TOOL_ERROR",
        "INVALID_TOOL_BATCH",
        "INVALID_TOOL_CALL_JSON",
        "REPEATED_TOOL_CALL_LIMIT",
    }


def test_observation_refresh_and_removal_are_deterministic(tmp_path: Path) -> None:
    state = RunState(
        run_id="state-test",
        workspace_root=tmp_path,
        started_at=datetime.now(UTC),
    )

    first = state.observe_file("file.txt", "a" * 64, observed_at_step=1)
    second = state.observe_file("file.txt", "b" * 64, observed_at_step=2)
    removed = state.remove_observation("file.txt")

    assert first.sha256 == "a" * 64
    assert second.sha256 == "b" * 64
    assert removed == second
    assert state.get_observation("file.txt") is None


def test_failed_mutation_requires_error_code() -> None:
    with pytest.raises(ValidationError, match="must contain an error_code"):
        MutationRecord(
            step=1,
            operation="write_file",
            destination="file.txt",
            status="failed",
            changed=False,
        )


def test_failed_mutation_cannot_claim_a_filesystem_change() -> None:
    with pytest.raises(ValidationError, match="cannot be marked changed"):
        MutationRecord(
            step=1,
            operation="move_file",
            destination="file.txt",
            status="failed",
            changed=True,
            error_code="PRECONDITION_FAILED",
        )


def test_runtime_source_has_no_seed_answers_frameworks_or_unbounded_reads() -> None:
    runtime_files = sorted((REPOSITORY_ROOT / "app").glob("*.py")) + [REPOSITORY_ROOT / "agent.py"]
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    seed_paths = [item["path"] for item in baseline["files"]]
    seed_names = [Path(path).name for path in seed_paths]
    top_directories = {path.split("/")[0] for path in seed_paths}
    forbidden_text = [
        "Project Falcon",
        "Project Phoenix",
        "workspace_baseline.json",
        "LangChain",
        "LangGraph",
        "CrewAI",
        "OpenAI Agents SDK",
        "FastAPI",
        "falcon_index.md",
        "archive/MANIFEST.md",
        "reasoning_content",
        "thinking enabled",
        "strict Beta",
        "delete_file",
    ]

    for runtime_file in runtime_files:
        source = runtime_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for value in [*forbidden_text, *seed_paths, *seed_names]:
            assert value not in source
        for directory in top_directories:
            assert f'"{directory}/' not in source
            assert f"'{directory}/" not in source

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported = [alias.name for alias in node.names]
                assert "subprocess" not in imported
                assert all(not name.startswith("scripts.validate_t") for name in imported)
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec"}
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"read_text", "readlines", "splitlines", "system"}
                if node.func.attr == "read":
                    assert node.args or node.keywords, "unbounded read() found in runtime code"

        if runtime_file.name != "config.py":
            assert "deepseek-chat" not in source
            assert "deepseek-reasoner" not in source
