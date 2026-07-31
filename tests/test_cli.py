from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AGENT_ENTRYPOINT = REPOSITORY_ROOT / "agent.py"


def test_agent_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, str(AGENT_ENTRYPOINT), "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    for option in (
        "--workspace",
        "--task",
        "--task-file",
        "--trace",
        "--max-turns",
        "--model",
        "--timeout",
        "--max-total-tokens",
        "--json",
    ):
        assert option in completed.stdout
    assert "api-key" not in completed.stdout.lower()
