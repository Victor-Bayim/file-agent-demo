from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import app.application as application_module
from app.application import ApplicationStartupError, execute_task
from app.config import DeepSeekConfig
from app.model_types import (
    ModelFinishReason,
    ModelMessage,
    ModelResponse,
    ModelRole,
    ModelToolCall,
    ModelUsage,
)
from app.runtime import AgentRunStatus
from tests.fake_model import FakeModelClient


def response(
    content: str | None = None,
    calls: list[ModelToolCall] | None = None,
) -> ModelResponse:
    tool_calls = calls or []
    finish = ModelFinishReason.TOOL_CALLS if tool_calls else ModelFinishReason.STOP
    return ModelResponse(
        message=ModelMessage(
            role=ModelRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        ),
        usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3, exact=True),
        finish_reason=finish,
        raw_finish_reason=finish.value,
    )


def test_execute_task_assembles_prompt_registry_loop_and_default_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FILE_AGENT_RUNS_DIR", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file.txt").write_text("content", encoding="utf-8")
    model = FakeModelClient(
        [
            response(
                calls=[
                    ModelToolCall.from_arguments(
                        id="list-1",
                        name="list_directory",
                        arguments={"path": "."},
                    )
                ]
            ),
            response("Completed safely."),
        ]
    )

    result = asyncio.run(
        execute_task(workspace=workspace, task="Inspect the root.", model_client=model)
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert result.trace_path is not None
    assert result.trace_path.parent.parent == (tmp_path / "runs").resolve()
    assert not result.trace_path.is_relative_to(workspace)
    trace_lines = result.trace_path.read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 1
    assert json.loads(trace_lines[0])["tool"] == "list_directory"
    assert [message.role for message in model.calls[0].messages] == [
        ModelRole.SYSTEM,
        ModelRole.USER,
    ]
    assert model.calls[0].messages[1].content == "Inspect the root."
    assert len(model.calls[0].tools) == 6
    assert model.calls[1].messages[-1].role is ModelRole.TOOL


def test_execute_task_honors_explicit_external_trace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace = tmp_path / "observability" / "trace.jsonl"

    result = asyncio.run(
        execute_task(
            workspace=workspace,
            task="Finish without changes.",
            trace_path=trace,
            model_client=FakeModelClient([response("Done.")]),
        )
    )

    assert result.trace_path == trace.resolve()
    assert trace.is_file()


def test_execute_task_uses_configured_runs_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    configured_runs = tmp_path / "configured-runs"
    monkeypatch.setenv("FILE_AGENT_RUNS_DIR", str(configured_runs))

    result = asyncio.run(
        execute_task(
            workspace=workspace,
            task="Finish safely.",
            model_client=FakeModelClient([response("Done.")]),
        )
    )

    assert result.trace_path is not None
    assert result.trace_path.parent.parent == configured_runs.resolve()


def test_execute_task_rejects_trace_inside_workspace_without_creating_it(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace = workspace / "trace.jsonl"

    with pytest.raises(ApplicationStartupError) as captured:
        asyncio.run(
            execute_task(
                workspace=workspace,
                task="Do nothing.",
                trace_path=trace,
                model_client=FakeModelClient([response("Done.")]),
            )
        )

    assert captured.value.code == "TRACE_INVALID"
    assert not trace.exists()


@pytest.mark.parametrize("workspace_kind", ["missing", "file"])
def test_execute_task_rejects_invalid_workspace(
    workspace_kind: str,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    if workspace_kind == "file":
        workspace.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ApplicationStartupError) as captured:
        asyncio.run(
            execute_task(
                workspace=workspace,
                task="Task",
                model_client=FakeModelClient([response("Done.")]),
            )
        )

    assert captured.value.code == "WORKSPACE_INVALID"


def test_execute_task_requires_key_only_for_default_model_client(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ApplicationStartupError) as captured:
        asyncio.run(
            execute_task(
                workspace=workspace,
                task="Task",
                deepseek_config=DeepSeekConfig(),
            )
        )

    assert captured.value.code == "MODEL_CONFIGURATION"
    assert "DEEPSEEK_API_KEY" in captured.value.safe_message


def test_execute_task_does_not_close_injected_model_client(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = FakeModelClient([response("Done.")])
    model.closed = False  # type: ignore[attr-defined]

    asyncio.run(execute_task(workspace=workspace, task="Task", model_client=model))

    assert model.closed is False  # type: ignore[attr-defined]


def test_execute_task_closes_internally_created_deepseek_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    created = []

    class ClosingClient:
        def __init__(self, config: DeepSeekConfig) -> None:
            del config
            self.closed = False
            created.append(self)

        async def complete(self, messages, tools):  # type: ignore[no-untyped-def]
            del messages, tools
            return response("Done.")

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(application_module, "DeepSeekClient", ClosingClient)

    result = asyncio.run(
        execute_task(
            workspace=workspace,
            task="Task",
            deepseek_config=DeepSeekConfig(api_key="fake-test-key"),
        )
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert len(created) == 1
    assert created[0].closed is True


def test_api_key_is_absent_from_trace_and_serialized_result(tmp_path: Path) -> None:
    placeholder = "trace-redaction-placeholder"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace = tmp_path / "trace.jsonl"
    model = FakeModelClient(
        [
            response(
                calls=[
                    ModelToolCall.from_arguments(
                        id="list-root",
                        name="list_directory",
                        arguments={"path": "."},
                    )
                ]
            ),
            response("No changes made."),
        ]
    )

    result = asyncio.run(
        execute_task(
            workspace=workspace,
            task="List the root.",
            trace_path=trace,
            deepseek_config=DeepSeekConfig(api_key=placeholder),
            model_client=model,
        )
    )

    assert placeholder not in trace.read_text(encoding="utf-8")
    assert placeholder not in result.model_dump_json()
