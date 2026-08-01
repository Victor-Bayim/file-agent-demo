from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_native_frontend_assets_exist_without_external_dependencies() -> None:
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    styles = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")

    assert index and script and styles
    assert "http://" not in index
    assert "https://" not in index
    assert "cdn" not in index.casefold()


def test_frontend_uses_safe_dynamic_text_and_required_controls() -> None:
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

    assert "textContent" in script
    assert "innerHTML" not in script
    assert "eval(" not in script
    assert "new Function" not in script
    assert "EventSource" in script
    assert "localStorage" not in script
    for value in (
        "reset-button",
        "run-button",
        "cancel-button",
        "file-tree",
        "stat-model-calls",
        "stat-total-tokens",
        "trace-list",
    ):
        assert value in index
    for forbidden in ("DEEPSEEK_API_KEY", "Authorization: Bearer"):
        assert forbidden not in index
        assert forbidden not in script
