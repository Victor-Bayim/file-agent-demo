from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import app.cli as cli_module
from app.cli import main
from app.runtime import AgentRunResult, AgentRunStatus, UsageStats


@pytest.fixture(autouse=True)
def isolate_project_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "project-root"
    project_root.mkdir()
    monkeypatch.setattr(cli_module, "PROJECT_ROOT", project_root)


def run_result(status: AgentRunStatus, trace_path: Path) -> AgentRunResult:
    completed = status is AgentRunStatus.COMPLETED
    return AgentRunResult(
        run_id="cli-run",
        status=status,
        trace_path=trace_path,
        answer="Completed." if completed else None,
        reason=None if completed else "Stopped safely.",
        reason_code=None if completed else f"TEST_{status.value.upper()}",
        usage=UsageStats(input_tokens=4, output_tokens=2, total_tokens=6),
        model_calls=2,
        tool_calls=1,
        finish_reason="stop",
        raw_finish_reason="stop",
        provider_model="test-provider-model",
        elapsed_ms=12.5,
    )


def base_args(tmp_path: Path) -> list[str]:
    return ["--workspace", str(tmp_path / "workspace"), "--task", "Do the task."]


def test_cli_missing_key_returns_two_before_execution_or_trace_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = False

    async def unexpected_execute(**kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError(kwargs)

    monkeypatch.setattr(cli_module, "execute_task", unexpected_execute)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    trace = tmp_path / "trace.jsonl"

    exit_code = main([*base_args(tmp_path), "--trace", str(trace)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "DEEPSEEK_API_KEY" in captured.err
    assert called is False
    assert not trace.exists()


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [
        (AgentRunStatus.COMPLETED, 0),
        (AgentRunStatus.INCOMPLETE, 3),
        (AgentRunStatus.FAILED, 4),
        (AgentRunStatus.CANCELLED, 5),
    ],
)
def test_cli_exit_codes_and_human_output(
    status: AgentRunStatus,
    expected_exit: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-secret")
    trace = tmp_path / "runs" / "trace.jsonl"

    async def fake_execute(**kwargs: Any) -> AgentRunResult:
        return run_result(status, trace)

    monkeypatch.setattr(cli_module, "execute_task", fake_execute)

    exit_code = main(base_args(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == expected_exit
    assert f"status: {status.value}" in output
    assert "model calls: 2" in output
    assert "tool calls: 1" in output
    assert "finish reason: stop" in output
    assert "provider model: test-provider-model" in output
    assert "token usage:" in output
    assert "elapsed time:" in output
    assert f"trace path: {trace}" in output
    assert "changed mutations: 0" in output
    assert "failed mutations: 0" in output
    assert "unit-test-secret" not in output


def test_cli_json_is_one_valid_object_and_contains_no_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "never-print-this-secret"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    trace = tmp_path / "trace.jsonl"

    async def fake_execute(**kwargs: Any) -> AgentRunResult:
        return run_result(AgentRunStatus.COMPLETED, trace)

    monkeypatch.setattr(cli_module, "execute_task", fake_execute)

    assert main([*base_args(tmp_path), "--json"]) == 0

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert output.count("\n") == 1
    assert payload["status"] == "completed"
    assert payload["usage"]["total_tokens"] == 6
    assert payload["trace_path"] == str(trace)
    assert payload["finish_reason"] == "stop"
    assert payload["provider_model"] == "test-provider-model"
    assert secret not in output


def test_cli_loads_key_from_project_dotenv_without_leaking_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    placeholder = "dotenv-test-placeholder"
    (cli_module.PROJECT_ROOT / ".env").write_text(
        f"DEEPSEEK_API_KEY={placeholder}\n", encoding="utf-8"
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    captured: dict[str, Any] = {}

    async def fake_execute(**kwargs: Any) -> AgentRunResult:
        captured.update(kwargs)
        return run_result(AgentRunStatus.COMPLETED, tmp_path / "trace.jsonl")

    monkeypatch.setattr(cli_module, "execute_task", fake_execute)

    assert main([*base_args(tmp_path), "--json"]) == 0

    output = capsys.readouterr().out
    config = captured["deepseek_config"]
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == placeholder
    assert placeholder not in output


def test_process_environment_takes_priority_over_project_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dotenv_placeholder = "dotenv-lower-priority-placeholder"
    environment_placeholder = "environment-priority-placeholder"
    (cli_module.PROJECT_ROOT / ".env").write_text(
        f"DEEPSEEK_API_KEY={dotenv_placeholder}\n", encoding="utf-8"
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", environment_placeholder)
    captured: dict[str, Any] = {}

    async def fake_execute(**kwargs: Any) -> AgentRunResult:
        captured.update(kwargs)
        return run_result(AgentRunStatus.COMPLETED, tmp_path / "trace.jsonl")

    monkeypatch.setattr(cli_module, "execute_task", fake_execute)

    assert main([*base_args(tmp_path), "--json"]) == 0

    output = capsys.readouterr().out
    config = captured["deepseek_config"]
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == environment_placeholder
    assert dotenv_placeholder not in output
    assert environment_placeholder not in output


def test_invalid_dotenv_configuration_is_safely_redacted_in_json_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    placeholder = "invalid-config-placeholder"
    (cli_module.PROJECT_ROOT / ".env").write_text(
        f"DEEPSEEK_API_KEY={placeholder}\nDEEPSEEK_THINKING=enabled\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)

    exit_code = main([*base_args(tmp_path), "--json"])

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 2
    assert payload["status"] == "startup_error"
    assert placeholder not in output


def test_malformed_dotenv_error_contains_only_safe_line_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    placeholder = "malformed-file-placeholder"
    (cli_module.PROJECT_ROOT / ".env").write_text(
        f"DEEPSEEK_API_KEY={placeholder}\nexport UNSUPPORTED=value\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    exit_code = main([*base_args(tmp_path), "--json"])

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 2
    assert payload["status"] == "startup_error"
    assert "line 2" in payload["error"]
    assert placeholder not in output


def test_cli_overrides_are_validated_and_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-secret")
    monkeypatch.setenv("DEEPSEEK_THINKING", "disabled")
    captured: dict[str, Any] = {}

    async def fake_execute(**kwargs: Any) -> AgentRunResult:
        captured.update(kwargs)
        return run_result(AgentRunStatus.COMPLETED, tmp_path / "trace.jsonl")

    monkeypatch.setattr(cli_module, "execute_task", fake_execute)

    exit_code = main(
        [
            *base_args(tmp_path),
            "--model",
            "deepseek-v4-pro",
            "--timeout",
            "8.5",
            "--max-turns",
            "7",
            "--max-total-tokens",
            "900",
        ]
    )

    assert exit_code == 0
    assert captured["deepseek_config"].model == "deepseek-v4-pro"
    assert captured["deepseek_config"].timeout_seconds == 8.5
    assert captured["limits"].max_model_turns == 7
    assert captured["limits"].max_total_tokens == 900
