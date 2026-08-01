"""Run the local Web demo with a deterministic provider-free model client.

This development-only entry point intentionally does not load the project ``.env``
and never constructs a DeepSeek client. Web settings must be supplied explicitly in
the process environment.
"""

from __future__ import annotations

import sys

import uvicorn

from app.model_types import (
    ModelFinishReason,
    ModelMessage,
    ModelResponse,
    ModelRole,
    ModelToolCall,
    ModelUsage,
)
from app.web.app import create_web_app
from app.web.config import WebConfigurationError, WebSettings
from tests.fake_model import FakeModelClient


def _model_factory() -> FakeModelClient:
    list_call = ModelToolCall.from_arguments(
        id="offline-list",
        name="list_directory",
        arguments={"path": ".", "recursive": False},
    )
    return FakeModelClient(
        [
            ModelResponse(
                message=ModelMessage(
                    role=ModelRole.ASSISTANT,
                    tool_calls=[list_call],
                ),
                usage=ModelUsage(
                    input_tokens=2,
                    output_tokens=1,
                    total_tokens=3,
                    exact=True,
                ),
                finish_reason=ModelFinishReason.TOOL_CALLS,
                raw_finish_reason="tool_calls",
                provider_model="offline-fake-model",
            ),
            ModelResponse(
                message=ModelMessage(
                    role=ModelRole.ASSISTANT,
                    content="Offline fake run completed after listing the workspace root.",
                ),
                usage=ModelUsage(
                    input_tokens=2,
                    output_tokens=1,
                    total_tokens=3,
                    exact=True,
                ),
                finish_reason=ModelFinishReason.STOP,
                raw_finish_reason="stop",
                provider_model="offline-fake-model",
            ),
        ]
    )


def main() -> int:
    try:
        settings = WebSettings.from_environment()
        app = create_web_app(settings=settings, model_client_factory=_model_factory)
    except WebConfigurationError as exc:
        sys.stderr.write(f"Fake Web configuration error: {exc}\n")
        return 2
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=False,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
