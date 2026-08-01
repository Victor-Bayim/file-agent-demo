from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SECRET_PATTERN = re.compile(r"(?:sk-|Bearer\s+)[A-Za-z0-9._-]{20,}")


def _service() -> tuple[dict[str, object], str]:
    source = (ROOT / "render.yaml").read_text(encoding="utf-8")
    document = yaml.safe_load(source)
    assert isinstance(document, dict)
    assert list(document) == ["services"]
    services = document["services"]
    assert isinstance(services, list) and len(services) == 1
    service = services[0]
    assert isinstance(service, dict)
    return service, source


def _environment(service: dict[str, object]) -> dict[str, dict[str, object]]:
    variables = service["envVars"]
    assert isinstance(variables, list)
    return {str(item["key"]): item for item in variables if isinstance(item, dict)}


def test_render_blueprint_service_shape() -> None:
    service, source = _service()

    assert service["name"] == "file-agent-demo"
    assert service["type"] == "web"
    assert service["runtime"] == "docker"
    assert service["plan"] == "free"
    assert service["region"] == "singapore"
    assert service["healthCheckPath"] == "/healthz"
    assert service["autoDeployTrigger"] == "commit"
    assert "maxShutdownDelaySeconds" not in service
    assert "disk" not in service
    assert "numInstances" not in service
    assert "dockerCommand" not in service
    assert "workers" not in source.casefold()
    assert SECRET_PATTERN.search(source) is None


def test_render_secrets_and_public_safety_settings() -> None:
    service, _source = _service()
    environment = _environment(service)

    for key in ("DEEPSEEK_API_KEY", "FILE_AGENT_WEB_ACCESS_CODE"):
        assert environment[key] == {"key": key, "sync": False}
    assert environment["FILE_AGENT_WEB_PUBLIC_MODE"]["value"] == "true"
    assert environment["FILE_AGENT_WEB_COOKIE_SECURE"]["value"] == "true"
    assert environment["FILE_AGENT_WEB_MAX_CONCURRENT_RUNS"]["value"] == "1"
    assert environment["FILE_AGENT_WEB_HOST"]["value"] == "0.0.0.0"
    assert environment["FILE_AGENT_WEB_SESSION_ROOT"]["value"] == "/tmp/file-agent/sessions"
    assert environment["FILE_AGENT_WEB_RUNS_ROOT"]["value"] == "/tmp/file-agent/runs"
    assert "FILE_AGENT_WEB_PORT" not in environment
