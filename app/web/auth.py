"""Authentication primitives for opaque server-side Web sessions."""

from __future__ import annotations

import hmac

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from starlette.responses import Response

SESSION_COOKIE_NAME = "file_agent_session"


class AuthRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    access_code: SecretStr = Field(min_length=1, max_length=1024)


def access_code_matches(expected: SecretStr, supplied: SecretStr) -> bool:
    """Compare without early-exit string equality."""
    return hmac.compare_digest(expected.get_secret_value(), supplied.get_secret_value())


def csrf_matches(expected: str, supplied: str | None) -> bool:
    return supplied is not None and hmac.compare_digest(expected, supplied)


def set_session_cookie(
    response: Response,
    session_id: str,
    *,
    secure: bool,
    max_age: int,
) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
        max_age=max_age,
    )


def clear_session_cookie(response: Response, *, secure: bool) -> None:
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )
