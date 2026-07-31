"""Command-line composition for one real file-Agent run."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.application import ApplicationStartupError, execute_task
from app.config import AgentLimits, ConfigurationError, DeepSeekConfig, RuntimeConfig
from app.env_loader import EnvFileError, load_project_env
from app.runtime import AgentRunResult, AgentRunStatus

EXIT_BY_STATUS = {
    AgentRunStatus.COMPLETED: 0,
    AgentRunStatus.INCOMPLETE: 3,
    AgentRunStatus.FAILED: 4,
    AgentRunStatus.CANCELLED: 5,
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliConfigurationError(ValueError):
    """Safe user-facing configuration failure."""


def load_project_environment(project_root: Path | None = None) -> None:
    """Load only the project-root .env without replacing process variables."""
    root = PROJECT_ROOT if project_root is None else project_root.resolve()
    load_project_env(root / ".env", override=False)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="file-agent",
        description="Run a natural-language task against an isolated workspace.",
    )
    parser.add_argument("--workspace", required=True, type=Path, help="Workspace root path.")
    parser.add_argument("--task", required=True, help="Natural-language task to execute.")
    parser.add_argument("--trace", type=Path, help="Optional JSONL trace output path.")
    parser.add_argument("--max-turns", type=positive_int, help="Model-turn limit override.")
    parser.add_argument(
        "--model",
        choices=("deepseek-v4-flash", "deepseek-v4-pro"),
        help="DeepSeek model override.",
    )
    parser.add_argument("--timeout", type=positive_float, help="Provider timeout in seconds.")
    parser.add_argument(
        "--max-total-tokens",
        type=positive_int,
        help="Total logical token budget override.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON.")
    return parser


def _configured_values(args: argparse.Namespace) -> tuple[DeepSeekConfig, AgentLimits]:
    deepseek = DeepSeekConfig.from_environment()
    if deepseek.api_key is None:
        raise CliConfigurationError("Set DEEPSEEK_API_KEY before running a task.")
    deepseek_payload = deepseek.model_dump()
    if args.model is not None:
        deepseek_payload["model"] = args.model
    if args.timeout is not None:
        deepseek_payload["timeout_seconds"] = args.timeout

    runtime = RuntimeConfig.from_environment()
    limits_payload = runtime.limits.model_dump()
    if args.max_turns is not None:
        limits_payload["max_model_turns"] = args.max_turns
    if args.max_total_tokens is not None:
        limits_payload["max_total_tokens"] = args.max_total_tokens
    try:
        return DeepSeekConfig.model_validate(deepseek_payload), AgentLimits.model_validate(
            limits_payload
        )
    except ValidationError:
        raise CliConfigurationError("Command-line limits are inconsistent.") from None


def _result_payload(result: AgentRunResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "answer": result.answer,
        "reason": result.reason,
        "reason_code": result.reason_code,
        "model_calls": result.model_calls,
        "tool_calls": result.tool_calls,
        "finish_reason": result.finish_reason,
        "raw_finish_reason": result.raw_finish_reason,
        "provider_model": result.provider_model,
        "usage": result.usage.model_dump(mode="json"),
        "elapsed_ms": result.elapsed_ms,
        "trace_path": str(result.trace_path) if result.trace_path is not None else None,
        "changed_mutations": [
            mutation.model_dump(mode="json") for mutation in result.changed_mutations
        ],
        "failed_mutations": [
            mutation.model_dump(mode="json") for mutation in result.failed_mutations
        ],
    }


def _print_result(result: AgentRunResult, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(_result_payload(result), ensure_ascii=False, separators=(",", ":")))
        return
    usage = result.usage
    usage_text = (
        f"input={usage.input_tokens}, output={usage.output_tokens}, total={usage.total_tokens}, "
        f"exact={str(usage.exact).lower()}"
        if usage.available
        else "unavailable"
    )
    print(f"status: {result.status.value}")
    answer_label = "final answer" if result.answer is not None else "reason"
    print(f"{answer_label}: {result.answer or result.reason}")
    print(f"model calls: {result.model_calls}")
    print(f"tool calls: {result.tool_calls}")
    print(f"finish reason: {result.finish_reason}")
    print(f"provider model: {result.provider_model}")
    print(f"token usage: {usage_text}")
    print(f"elapsed time: {result.elapsed_ms:.1f} ms")
    print(f"trace path: {result.trace_path}")
    print(f"changed mutations: {len(result.changed_mutations)}")
    print(f"failed mutations: {len(result.failed_mutations)}")


def _print_startup_error(message: str, *, json_output: bool) -> None:
    if json_output:
        print(
            json.dumps(
                {"status": "startup_error", "error": message},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    else:
        print(f"configuration/startup error: {message}", file=sys.stderr)


async def _run(args: argparse.Namespace) -> AgentRunResult:
    deepseek, limits = _configured_values(args)
    return await execute_task(
        workspace=args.workspace,
        task=args.task,
        trace_path=args.trace,
        deepseek_config=deepseek,
        limits=limits,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_project_environment()
        result = asyncio.run(_run(args))
    except (
        CliConfigurationError,
        ConfigurationError,
        EnvFileError,
        ApplicationStartupError,
    ) as exc:
        message = exc.safe_message if isinstance(exc, ApplicationStartupError) else str(exc)
        _print_startup_error(message, json_output=args.json_output)
        return 2
    except KeyboardInterrupt:
        _print_startup_error("The run was cancelled by the user.", json_output=args.json_output)
        return 5
    _print_result(result, json_output=args.json_output)
    return EXIT_BY_STATUS[result.status]
