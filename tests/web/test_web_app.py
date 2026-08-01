from __future__ import annotations

import time
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.web.app import create_web_app
from app.web.config import WebConfigurationError, WebSettings
from tests.fake_model import FakeModelClient
from tests.web.conftest import authenticate, model_response, tool_call


def wait_for_run(client: TestClient, run_id: str) -> dict[str, object]:
    for _attempt in range(200):
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] != "running":
            return payload
        time.sleep(0.005)
    raise AssertionError("Fake Web run did not finish")


def test_home_static_api_404_and_security_headers(web_client: TestClient) -> None:
    home = web_client.get("/")
    script = web_client.get("/static/app.js")
    missing_api = web_client.get("/api/does-not-exist")

    assert home.status_code == 200
    assert "File Agent" in home.text
    assert script.status_code == 200
    assert missing_api.status_code == 404
    assert missing_api.headers["content-type"].startswith("application/json")
    for response in (home, script, missing_api):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["x-robots-tag"] == "noindex, nofollow"
        assert "default-src 'self'" in response.headers["content-security-policy"]
        assert "access-control-allow-origin" not in response.headers


def test_fake_model_end_to_end_sse_file_and_reset(web_settings: WebSettings) -> None:
    def factory() -> FakeModelClient:
        return FakeModelClient(
            [
                model_response(
                    calls=[
                        tool_call(
                            "write_file",
                            {"path": "demo-output.txt", "content": "offline result\n"},
                            "write",
                        )
                    ]
                ),
                model_response(calls=[tool_call("read_file", {"path": "demo-output.txt"}, "read")]),
                model_response("The offline output was verified."),
            ]
        )

    app = create_web_app(settings=web_settings, model_client_factory=factory)
    with TestClient(app) as client:
        csrf = authenticate(client)
        initial_tree = client.get("/api/workspace/tree").json()
        assert initial_tree["count"] == 3

        started = client.post(
            "/api/runs",
            json={"task": "Create and verify a generic demo output."},
            headers={"X-CSRF-Token": csrf},
        )
        assert started.status_code == 202
        assert started.json()["status"] == "running"
        run_id = started.json()["run_id"]
        result = wait_for_run(client, run_id)
        assert result["status"] == "completed"
        assert result["answer"] == "The offline output was verified."
        assert result["model_calls"] == 3
        assert result["tool_calls"] == 2
        assert result["changed_mutations"] == 1
        assert result["failed_mutations"] == 0
        assert result["trace_available"] is True

        events = client.get(f"/api/runs/{run_id}/events")
        assert events.status_code == 200
        assert events.headers["content-type"].startswith("text/event-stream")
        assert events.headers["cache-control"] == "no-cache"
        assert events.headers["x-accel-buffering"] == "no"
        assert "event: tool_completed" in events.text
        assert "event: run_finished" in events.text
        assert "offline result" not in events.text

        output = client.get("/api/workspace/file", params={"path": "demo-output.txt"})
        assert output.status_code == 200
        assert output.json()["content"] == "offline result\n"

        reset = client.post(
            "/api/workspace/reset",
            headers={"X-CSRF-Token": csrf},
        )
        assert reset.status_code == 200
        assert reset.json()["workspace_revision"] == 1
        assert (
            client.get("/api/workspace/file", params={"path": "demo-output.txt"}).status_code == 400
        )
        final_tree = client.get("/api/workspace/tree").json()
        assert final_tree == initial_tree


def test_task_validation_and_rate_retry(
    web_settings: WebSettings,
    simple_model_factory: Callable[[], FakeModelClient],
) -> None:
    settings = web_settings.model_copy(update={"max_task_chars": 5, "max_runs_per_session_hour": 1})
    app = create_web_app(settings=settings, model_client_factory=simple_model_factory)
    with TestClient(app) as client:
        csrf = authenticate(client)
        assert (
            client.post("/api/runs", json={"task": ""}, headers={"X-CSRF-Token": csrf}).status_code
            == 422
        )
        assert (
            client.post(
                "/api/runs", json={"task": "123456"}, headers={"X-CSRF-Token": csrf}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/runs", json={"task": "a\x00b"}, headers={"X-CSRF-Token": csrf}
            ).status_code
            == 400
        )
        started = client.post("/api/runs", json={"task": "first"}, headers={"X-CSRF-Token": csrf})
        run_id = started.json()["run_id"]
        wait_for_run(client, run_id)
        limited = client.post("/api/runs", json={"task": "again"}, headers={"X-CSRF-Token": csrf})
        assert limited.status_code == 429
        assert int(limited.headers["retry-after"]) > 0
        assert limited.json()["error"]["retry_after_seconds"] > 0
        assert client.get("/api/runs/not-owned").status_code == 404


def test_last_event_id_avoids_duplicate_replay(web_client: TestClient) -> None:
    csrf = authenticate(web_client)
    started = web_client.post(
        "/api/runs",
        json={"task": "Finish offline."},
        headers={"X-CSRF-Token": csrf},
    )
    run_id = started.json()["run_id"]
    wait_for_run(web_client, run_id)

    replay = web_client.get(
        f"/api/runs/{run_id}/events",
        headers={"Last-Event-ID": "1"},
    )

    assert replay.status_code == 200
    assert "id: 1\n" not in replay.text
    assert "run_finished" in replay.text
    assert (
        web_client.get(
            f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "invalid"}
        ).status_code
        == 400
    )


def test_production_model_requires_credentials_without_network(
    web_settings: WebSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_THINKING",
        "DEEPSEEK_TEMPERATURE",
        "DEEPSEEK_MAX_OUTPUT_TOKENS",
        "DEEPSEEK_TIMEOUT_SECONDS",
        "DEEPSEEK_MAX_RETRIES",
    ):
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(WebConfigurationError, match="credentials are required"):
        create_web_app(settings=web_settings)


def test_reset_returns_409_while_session_run_is_active(web_client: TestClient) -> None:
    csrf = authenticate(web_client)
    sessions = web_client.app.state.session_manager.sessions
    session = next(iter(sessions.values()))
    session.active_run_id = "active-run"
    try:
        response = web_client.post(
            "/api/workspace/reset",
            headers={"X-CSRF-Token": csrf},
        )
    finally:
        session.active_run_id = None

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ACTIVE_RUN"


def test_healthz_is_minimal_and_has_no_session_run_or_model_side_effect(
    web_settings: WebSettings,
) -> None:
    factory_calls = 0

    def factory() -> FakeModelClient:
        nonlocal factory_calls
        factory_calls += 1
        return FakeModelClient([model_response("Unexpected.")])

    app = create_web_app(settings=web_settings, model_client_factory=factory)
    with TestClient(app) as client:
        sessions_before = list(web_settings.session_root.iterdir())
        runs_before = list(web_settings.web_runs_root.iterdir())

        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert response.text == '{"status":"ok"}'
        assert app.state.session_manager.sessions == {}
        assert app.state.run_manager.active_count == 0
        assert list(web_settings.session_root.iterdir()) == sessions_before
        assert list(web_settings.web_runs_root.iterdir()) == runs_before
        assert factory_calls == 0


def test_public_mode_disables_documentation_and_sets_secure_cookie(
    web_settings: WebSettings,
    simple_model_factory: Callable[[], FakeModelClient],
) -> None:
    settings = web_settings.model_copy(update={"public_mode": True, "cookie_secure": True})
    app = create_web_app(settings=settings, model_client_factory=simple_model_factory)

    with TestClient(app) as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404
        response = client.post(
            "/api/auth",
            json={"access_code": settings.access_code.get_secret_value()},
        )
        assert response.status_code == 200
        assert "Secure" in response.headers["set-cookie"]


def test_public_mode_still_requires_model_credentials_without_injected_fake(
    web_settings: WebSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = web_settings.model_copy(update={"public_mode": True, "cookie_secure": True})
    for variable in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_THINKING",
        "DEEPSEEK_TEMPERATURE",
        "DEEPSEEK_MAX_OUTPUT_TOKENS",
        "DEEPSEEK_TIMEOUT_SECONDS",
        "DEEPSEEK_MAX_RETRIES",
    ):
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(WebConfigurationError, match="credentials are required"):
        create_web_app(settings=settings)
