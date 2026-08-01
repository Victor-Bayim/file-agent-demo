from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.model_types import (
    ModelFinishReason,
    ModelMessage,
    ModelResponse,
    ModelRole,
    ModelToolCall,
    ModelUsage,
)
from app.web.app import create_web_app
from app.web.config import WebSettings
from tests.fake_model import FakeModelClient

TEST_ACCESS_CODE = "offline-test-passphrase"


def model_response(
    content: str | None = None,
    *,
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
        provider_model="fake-model",
    )


def tool_call(name: str, arguments: dict[str, object], call_id: str) -> ModelToolCall:
    return ModelToolCall.from_arguments(id=call_id, name=name, arguments=arguments)


@pytest.fixture
def seed_workspace(tmp_path: Path) -> Path:
    seed = tmp_path / "seed"
    (seed / "notes").mkdir(parents=True)
    (seed / "notes" / "alpha.txt").write_text(
        "one\ntwo\nthree\n",
        encoding="utf-8",
        newline="",
    )
    (seed / "root.txt").write_text("root\n", encoding="utf-8", newline="")
    return seed


@pytest.fixture
def web_settings(seed_workspace: Path, tmp_path: Path) -> WebSettings:
    return WebSettings(
        seed_workspace=seed_workspace,
        session_root=tmp_path / "sessions",
        web_runs_root=tmp_path / "runs",
        access_code=TEST_ACCESS_CODE,
        sse_keepalive_seconds=0.02,
    )


@pytest.fixture
def simple_model_factory() -> Callable[[], FakeModelClient]:
    return lambda: FakeModelClient([model_response("Completed offline.")])


@pytest.fixture
def web_client(
    web_settings: WebSettings,
    simple_model_factory: Callable[[], FakeModelClient],
):  # type: ignore[no-untyped-def]
    app = create_web_app(settings=web_settings, model_client_factory=simple_model_factory)
    with TestClient(app) as client:
        yield client


def authenticate(client: TestClient) -> str:
    response = client.post("/api/auth", json={"access_code": TEST_ACCESS_CODE})
    assert response.status_code == 200
    return str(response.json()["csrf_token"])
