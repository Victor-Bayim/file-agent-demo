from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.config import AgentLimits, ConfigurationError, DeepSeekConfig, RuntimeConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_agent_limits_defaults() -> None:
    limits = AgentLimits()

    assert limits.model_dump() == {
        "max_model_turns": 20,
        "max_tool_calls": 50,
        "max_runtime_seconds": 120.0,
        "max_identical_calls": 3,
        "max_write_bytes": 100_000,
        "max_tool_result_chars": 16_000,
        "max_total_tokens": 100_000,
        "max_tool_history_chars": 60_000,
    }


@pytest.mark.parametrize(
    "field_name",
    [
        "max_model_turns",
        "max_tool_calls",
        "max_runtime_seconds",
        "max_identical_calls",
        "max_write_bytes",
        "max_tool_result_chars",
        "max_total_tokens",
        "max_tool_history_chars",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_agent_limits_reject_non_positive_values(field_name: str, value: int) -> None:
    with pytest.raises(ValidationError):
        AgentLimits(**{field_name: value})


def test_identical_call_limit_cannot_exceed_tool_limit() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        AgentLimits(max_tool_calls=2, max_identical_calls=3)


def test_runtime_config_environment_overrides(tmp_path: Path) -> None:
    runs_dir = tmp_path / "configured-runs"
    config = RuntimeConfig.from_environment(
        {
            "FILE_AGENT_RUNS_DIR": str(runs_dir),
            "FILE_AGENT_MAX_MODEL_TURNS": "7",
            "FILE_AGENT_MAX_TOOL_CALLS": "19",
            "FILE_AGENT_MAX_RUNTIME_SECONDS": "45.5",
            "FILE_AGENT_MAX_IDENTICAL_CALLS": "2",
            "FILE_AGENT_MAX_WRITE_BYTES": "2048",
            "FILE_AGENT_MAX_TOOL_RESULT_CHARS": "4096",
            "FILE_AGENT_MAX_TOTAL_TOKENS": "8192",
            "FILE_AGENT_MAX_TOOL_HISTORY_CHARS": "12000",
            "FILE_AGENT_LOG_LEVEL": "debug",
            "OPENAI_API_KEY": "must-not-be-read",
        }
    )

    assert config.runs_dir == runs_dir
    assert config.log_level == "DEBUG"
    assert config.limits == AgentLimits(
        max_model_turns=7,
        max_tool_calls=19,
        max_runtime_seconds=45.5,
        max_identical_calls=2,
        max_write_bytes=2048,
        max_tool_result_chars=4096,
        max_total_tokens=8192,
        max_tool_history_chars=12_000,
    )
    assert not runs_dir.exists()


@pytest.mark.parametrize(
    ("environment_name", "value", "expected_field"),
    [
        ("FILE_AGENT_MAX_MODEL_TURNS", "not-an-int", "max_model_turns"),
        ("FILE_AGENT_MAX_RUNTIME_SECONDS", "0", "max_runtime_seconds"),
        ("FILE_AGENT_MAX_RUNTIME_SECONDS", "nan", "max_runtime_seconds"),
        ("FILE_AGENT_LOG_LEVEL", "verbose", "log_level"),
    ],
)
def test_invalid_environment_values_raise_clear_error(
    environment_name: str,
    value: str,
    expected_field: str,
) -> None:
    with pytest.raises(ConfigurationError, match=expected_field):
        RuntimeConfig.from_environment({environment_name: value})


def test_runtime_config_defaults_do_not_create_directories(tmp_path: Path) -> None:
    runs_dir = tmp_path / "not-created"

    config = RuntimeConfig(runs_dir=runs_dir)

    assert config.default_trace_filename == "trace.jsonl"
    assert config.log_level == "INFO"
    assert not runs_dir.exists()


@pytest.mark.parametrize("value", ["disabled", "none", "NULL", "off", "unlimited"])
def test_total_token_budget_can_be_disabled_from_environment(value: str) -> None:
    config = RuntimeConfig.from_environment({"FILE_AGENT_MAX_TOTAL_TOKENS": value})

    assert config.limits.max_total_tokens is None


def test_deepseek_config_defaults_do_not_require_a_key() -> None:
    config = DeepSeekConfig()

    assert config.api_key is None
    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-v4-flash"
    assert config.thinking == "disabled"
    assert config.temperature == 0.1
    assert config.max_output_tokens == 4096
    assert config.timeout_seconds == 45.0
    assert config.max_retries == 2


def test_deepseek_config_accepts_pro_and_masks_key() -> None:
    secret = "unit-test-secret-value"
    config = DeepSeekConfig(api_key=SecretStr(secret), model="deepseek-v4-pro")

    assert config.model == "deepseek-v4-pro"
    assert secret not in repr(config)
    assert secret not in config.model_dump_json()
    assert config.model_dump(mode="json")["api_key"] == "**********"


def test_deepseek_environment_mapping_is_explicit() -> None:
    config = DeepSeekConfig.from_environment(
        {
            "DEEPSEEK_API_KEY": "environment-secret",
            "DEEPSEEK_BASE_URL": "https://example.test/v1/",
            "DEEPSEEK_MODEL": "deepseek-v4-pro",
            "DEEPSEEK_THINKING": "disabled",
            "DEEPSEEK_TEMPERATURE": "0.25",
            "DEEPSEEK_MAX_OUTPUT_TOKENS": "8192",
            "DEEPSEEK_TIMEOUT_SECONDS": "12.5",
            "DEEPSEEK_MAX_RETRIES": "4",
            "OPENAI_API_KEY": "ignored",
        }
    )

    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "environment-secret"
    assert config.base_url == "https://example.test/v1"
    assert config.model == "deepseek-v4-pro"
    assert config.temperature == 0.25
    assert config.max_output_tokens == 8192
    assert config.timeout_seconds == 12.5
    assert config.max_retries == 4


@pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner"])
def test_deepseek_config_rejects_deprecated_models(model: str) -> None:
    with pytest.raises(ValidationError, match="deprecated"):
        DeepSeekConfig(model=model)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("api_key", "   "),
        ("base_url", "http://api.example.test"),
        ("base_url", "not-a-url"),
        ("thinking", "enabled"),
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("max_output_tokens", 0),
        ("max_output_tokens", 65_537),
        ("timeout_seconds", 0),
        ("max_retries", -1),
        ("max_retries", 6),
    ],
)
def test_deepseek_config_rejects_invalid_values(field_name: str, value: object) -> None:
    with pytest.raises(ValidationError):
        DeepSeekConfig(**{field_name: value})


def test_deepseek_environment_errors_do_not_echo_secret() -> None:
    secret = "should-never-appear"

    with pytest.raises(ConfigurationError) as captured:
        DeepSeekConfig.from_environment(
            {"DEEPSEEK_API_KEY": secret, "DEEPSEEK_THINKING": "enabled"}
        )

    assert secret not in str(captured.value)


def test_env_example_contains_only_documented_placeholders_and_defaults() -> None:
    lines = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    values = dict(line.split("=", 1) for line in lines if line)

    assert values["DEEPSEEK_API_KEY"] == ""
    assert values["DEEPSEEK_BASE_URL"] == "https://api.deepseek.com"
    assert values["DEEPSEEK_MODEL"] == "deepseek-v4-flash"
    assert values["DEEPSEEK_THINKING"] == "disabled"
    assert set(values) == {
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_THINKING",
        "DEEPSEEK_TEMPERATURE",
        "DEEPSEEK_MAX_OUTPUT_TOKENS",
        "DEEPSEEK_TIMEOUT_SECONDS",
        "DEEPSEEK_MAX_RETRIES",
        "FILE_AGENT_RUNS_DIR",
        "FILE_AGENT_MAX_MODEL_TURNS",
        "FILE_AGENT_MAX_TOOL_CALLS",
        "FILE_AGENT_MAX_RUNTIME_SECONDS",
        "FILE_AGENT_MAX_IDENTICAL_CALLS",
        "FILE_AGENT_MAX_TOTAL_TOKENS",
        "FILE_AGENT_MAX_TOOL_HISTORY_CHARS",
        "FILE_AGENT_WEB_ACCESS_CODE",
        "FILE_AGENT_WEB_SEED_WORKSPACE",
        "FILE_AGENT_WEB_SESSION_ROOT",
        "FILE_AGENT_WEB_RUNS_ROOT",
        "FILE_AGENT_WEB_HOST",
        "FILE_AGENT_WEB_PORT",
        "FILE_AGENT_WEB_COOKIE_SECURE",
        "FILE_AGENT_WEB_SESSION_TTL_SECONDS",
        "FILE_AGENT_WEB_MAX_SESSIONS",
        "FILE_AGENT_WEB_MAX_TASK_CHARS",
        "FILE_AGENT_WEB_MAX_CONCURRENT_RUNS",
        "FILE_AGENT_WEB_MAX_RUNS_PER_SESSION_HOUR",
        "FILE_AGENT_WEB_MAX_RUNS_PER_IP_HOUR",
        "FILE_AGENT_WEB_MAX_MODEL_TURNS",
        "FILE_AGENT_WEB_MAX_TOOL_CALLS",
        "FILE_AGENT_WEB_MAX_RUNTIME_SECONDS",
        "FILE_AGENT_WEB_MAX_TOTAL_TOKENS",
    }


def test_project_env_is_ignored_and_not_tracked() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", ".env"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert ignored.returncode == 0
    assert tracked.returncode != 0
