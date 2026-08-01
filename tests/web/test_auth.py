from __future__ import annotations

from fastapi.testclient import TestClient

from app.web.auth import SESSION_COOKIE_NAME
from app.web.config import WebSettings
from tests.web.conftest import TEST_ACCESS_CODE, authenticate


def test_correct_code_sets_opaque_secure_attributes(web_client: TestClient) -> None:
    response = web_client.post("/api/auth", json={"access_code": TEST_ACCESS_CODE})

    assert response.status_code == 200
    payload = response.json()
    assert payload["authenticated"] is True
    assert payload["csrf_token"]
    assert TEST_ACCESS_CODE not in response.text
    cookie = response.headers["set-cookie"]
    assert f"{SESSION_COOKIE_NAME}=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Path=/" in cookie
    assert TEST_ACCESS_CODE not in cookie
    assert "workspace" not in cookie


def test_wrong_code_is_safe_401(web_client: TestClient) -> None:
    supplied = "definitely-wrong"
    response = web_client.post("/api/auth", json={"access_code": supplied})

    assert response.status_code == 401
    assert supplied not in response.text
    assert TEST_ACCESS_CODE not in response.text


def test_unauthenticated_and_csrf_enforcement(web_client: TestClient) -> None:
    assert web_client.get("/api/workspace/tree").status_code == 401
    csrf = authenticate(web_client)

    missing = web_client.post("/api/workspace/reset")
    wrong = web_client.post("/api/workspace/reset", headers={"X-CSRF-Token": "wrong"})
    correct = web_client.post("/api/workspace/reset", headers={"X-CSRF-Token": csrf})

    assert missing.status_code == 403
    assert wrong.status_code == 403
    assert correct.status_code == 200


def test_logout_invalidates_and_removes_session(
    web_client: TestClient,
    web_settings: WebSettings,
) -> None:
    csrf = authenticate(web_client)
    assert len(list(web_settings.session_root.iterdir())) == 1

    response = web_client.post("/api/logout", headers={"X-CSRF-Token": csrf})

    assert response.status_code == 200
    assert web_client.get("/api/session").status_code == 401
    assert list(web_settings.session_root.iterdir()) == []
    assert list(web_settings.web_runs_root.iterdir()) == []
