"""Composition root for one CLI or future Web file-Agent run."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from app.agent_loop import run_agent
from app.config import AgentLimits, ConfigurationError, DeepSeekConfig, RuntimeConfig
from app.deepseek_client import DeepSeekClient, DeepSeekConfigurationError
from app.events import AgentEventSink
from app.filesystem_tools import build_filesystem_registry
from app.model_types import ModelClient
from app.prompts import build_initial_messages
from app.run_paths import default_trace_path, generate_run_id
from app.runtime import AgentRunResult, RunState
from app.sandbox import SandboxError, WorkspaceSandbox
from app.trace import JsonlTraceWriter, TraceError


class ApplicationStartupError(RuntimeError):
    """Safe startup failure with a stable public category."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.safe_message = message
        super().__init__(message)


async def execute_task(
    *,
    workspace: Path,
    task: str,
    trace_path: Path | None = None,
    deepseek_config: DeepSeekConfig | None = None,
    limits: AgentLimits | None = None,
    event_sink: AgentEventSink | None = None,
    cancel_event: asyncio.Event | None = None,
    model_client: ModelClient | None = None,
) -> AgentRunResult:
    """Assemble every component around the single handwritten ``run_agent`` loop."""
    selected_limits = limits or AgentLimits()
    run_id = generate_run_id()
    try:
        sandbox = WorkspaceSandbox(workspace)
    except SandboxError as exc:
        raise ApplicationStartupError("WORKSPACE_INVALID", exc.safe_message) from None

    state = RunState(
        run_id=run_id,
        workspace_root=sandbox.root,
        started_at=datetime.now(UTC),
    )
    registry = build_filesystem_registry(sandbox, state, selected_limits)
    try:
        messages = build_initial_messages(task)
    except ValueError as exc:
        raise ApplicationStartupError("TASK_INVALID", str(exc)) from None

    if model_client is None:
        try:
            selected_config = deepseek_config or DeepSeekConfig.from_environment()
            model_client = DeepSeekClient(selected_config)
        except ConfigurationError as exc:
            raise ApplicationStartupError("MODEL_CONFIGURATION", str(exc)) from None
        except DeepSeekConfigurationError:
            raise ApplicationStartupError(
                "MODEL_CONFIGURATION",
                "DEEPSEEK_API_KEY is required to construct the model client.",
            ) from None

    if trace_path is None:
        try:
            runs_dir = RuntimeConfig.from_environment().runs_dir
        except ConfigurationError as exc:
            raise ApplicationStartupError("RUNTIME_CONFIGURATION", str(exc)) from None
        trace_path = default_trace_path(runs_dir, run_id)
    try:
        with JsonlTraceWriter(trace_path, workspace_root=sandbox.root) as trace_writer:
            result = await run_agent(
                model=model_client,
                registry=registry,
                state=state,
                messages=messages,
                trace_writer=trace_writer,
                limits=selected_limits,
                event_sink=event_sink,
                cancel_event=cancel_event,
            )
            resolved_trace_path = trace_writer.output_path
    except TraceError as exc:
        raise ApplicationStartupError("TRACE_INVALID", str(exc)) from None
    except OSError:
        raise ApplicationStartupError(
            "TRACE_INITIALIZATION", "Trace initialization failed."
        ) from None
    return result.model_copy(update={"trace_path": resolved_trace_path})
