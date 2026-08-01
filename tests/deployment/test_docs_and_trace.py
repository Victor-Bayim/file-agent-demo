from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECRET_PATTERN = re.compile(r"(?:sk-|Bearer\s+)[A-Za-z0-9._-]{20,}")
WINDOWS_ABSOLUTE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[/\\]")


def test_readme_has_required_public_repository_sections_without_secrets() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for heading in (
        "# File Agent Runtime",
        "## Online demo",
        "## Features",
        "## Architecture",
        "## Agent loop",
        "## Security model",
        "## Local CLI",
        "## Local Web demo",
        "## Docker",
        "## Render deployment",
        "## Verification",
        "## Acceptance history",
        "## Known limitations",
        "## Repository structure",
    ):
        assert heading in readme
    assert "DEPLOYED_DEMO_URL" not in readme
    assert "https://file-agent-demo.onrender.com" in readme
    assert "Gate 2's first real attempt" in readme
    assert "T2's first exact-format attempt failed" in readme
    assert "sync: false" in readme
    assert "X-Forwarded-For" in readme
    assert SECRET_PATTERN.search(readme) is None
    assert WINDOWS_ABSOLUTE.search(readme) is None


def test_notes_is_half_page_and_contains_only_four_required_topics() -> None:
    notes = (ROOT / "NOTES.md").read_text(encoding="utf-8")
    words = re.findall(r"\b[\w'-]+\b", notes)
    headings = [line for line in notes.splitlines() if line.startswith("## ")]

    assert 200 <= len(words) <= 300
    assert headings == [
        "## Loop and termination",
        "## Context selection",
        "## Key trade-off",
        "## Known next step",
    ]
    for term in ("Tool Result", "Token budget", "chain-of-thought", "one mutating", "Redis"):
        assert term in notes
    assert SECRET_PATTERN.search(notes) is None
    assert WINDOWS_ABSOLUTE.search(notes) is None


def test_sample_trace_is_valid_sanitized_jsonl_with_required_tools() -> None:
    path = ROOT / "examples" / "sample-trace.jsonl"
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            assert raw_line.endswith("\n"), line_number
            records.append(json.loads(raw_line))

    tools = [record.get("tool") for record in records]
    assert "list_directory" in tools
    assert "read_file" in tools or "search_text" in tools
    assert "write_file" in tools
    write = next(record for record in records if record.get("tool") == "write_file")
    content = write["args"]["content"]
    assert isinstance(content, dict)
    assert set(content) == {"characters", "utf8_bytes", "sha256"}
    serialized = path.read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY" not in serialized
    assert "FILE_AGENT_WEB_ACCESS_CODE" not in serialized
    assert "Authorization" not in serialized
    assert WINDOWS_ABSOLUTE.search(serialized) is None
    assert "/tmp/" not in serialized
