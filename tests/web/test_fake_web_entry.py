from __future__ import annotations

import asyncio
from pathlib import Path

from app.model_types import ModelMessage, ModelRole
from scripts.run_fake_web import _model_factory

ROOT = Path(__file__).resolve().parents[2]


def test_fake_web_entry_is_provider_free_and_generic() -> None:
    source = (ROOT / "scripts" / "run_fake_web.py").read_text(encoding="utf-8")
    client = _model_factory()
    response = asyncio.run(
        client.complete(
            [ModelMessage(role=ModelRole.USER, content="Run a generic offline demo.")],
            [],
        )
    )

    assert response.provider_model == "offline-fake-model"
    assert response.message.tool_calls[0].name == "list_directory"
    assert "DeepSeekClient" not in source
    assert "load_project_env" not in source
    assert "DEEPSEEK_API_KEY" not in source
