"""Local Uvicorn entry point for the isolated File Agent Web demo."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from app.env_loader import EnvFileError, load_project_env
from app.web.app import create_web_app
from app.web.config import WebConfigurationError, WebSettings

PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run the local File Agent Web demo.")


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    try:
        load_project_env(PROJECT_ROOT / ".env", override=False)
        settings = WebSettings.from_environment()
        app = create_web_app(settings=settings)
    except (EnvFileError, WebConfigurationError) as exc:
        sys.stderr.write(f"Web configuration error: {exc}\n")
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
