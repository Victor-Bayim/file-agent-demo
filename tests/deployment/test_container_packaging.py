from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _lines(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip() and not line.startswith("#")}


def test_dockerfile_is_nonroot_minimal_and_contains_seed() -> None:
    path = ROOT / "Dockerfile"
    source = path.read_text(encoding="utf-8")

    assert path.is_file()
    assert source.startswith("FROM python:3.12-slim\n")
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "PYTHONUNBUFFERED=1" in source
    assert "PIP_NO_CACHE_DIR=1" in source
    assert "FILE_AGENT_WEB_HOST=0.0.0.0" in source
    assert "FILE_AGENT_WEB_SESSION_ROOT=/tmp/file-agent/sessions" in source
    assert "FILE_AGENT_WEB_RUNS_ROOT=/tmp/file-agent/runs" in source
    assert "COPY app/ ./app/" in source
    assert "COPY web/ ./web/" in source
    assert "COPY workspace/ ./workspace/" in source
    assert "COPY agent.py web_server.py ./" in source
    assert "USER fileagent" in source
    assert source.rstrip().endswith('CMD ["python", "web_server.py"]')
    assert "HEALTHCHECK" in source and "/healthz" in source
    assert 'pip install ".[dev]"' not in source
    assert "DEEPSEEK_API_KEY" not in source
    assert "FILE_AGENT_WEB_ACCESS_CODE" not in source
    assert "FILE_AGENT_WEB_PORT=" not in source
    assert "ARG " not in source
    assert "curl |" not in source


def test_dockerignore_excludes_sensitive_and_runtime_content_but_keeps_app() -> None:
    path = ROOT / ".dockerignore"
    patterns = _lines(path)

    assert path.is_file()
    for required in (
        ".git",
        ".env",
        ".env.*",
        ".venv",
        "runs",
        "runtime",
        "tests",
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        ".ruff_cache",
        "*.egg-info",
    ):
        assert required in patterns
    assert "!.env.example" in patterns
    for included in ("workspace", "workspace/", "app", "app/", "web", "web/"):
        assert included not in patterns
    assert "web_server.py" not in patterns
