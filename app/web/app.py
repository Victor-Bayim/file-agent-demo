"""FastAPI composition root for the isolated local Web demo."""

from __future__ import annotations

import logging
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import ConfigurationError, DeepSeekConfig
from app.deepseek_client import DeepSeekClient
from app.env_loader import EnvFileError, load_project_env
from app.model_types import ModelClient
from app.web.auth import (
    SESSION_COOKIE_NAME,
    AuthRequest,
    access_code_matches,
    clear_session_cookie,
    csrf_matches,
    set_session_cookie,
)
from app.web.browser import (
    WEB_FILE_MAX_LINES,
    BrowserError,
    list_workspace_tree,
    read_workspace_file,
)
from app.web.config import WebConfigurationError, WebSettings
from app.web.runs import RunManager, RunManagerError, RunNotFoundError
from app.web.sessions import (
    SessionActiveRunError,
    SessionError,
    SessionManager,
    SessionRecord,
)
from app.web.sse import stream_events

LOGGER = logging.getLogger("file_agent.web")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"


class WebAPIError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.safe_message = message
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=100_000)


def _require_session(request: Request) -> SessionRecord:
    manager: SessionManager = request.app.state.session_manager
    manager.cleanup_expired()
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    record = manager.get(session_id)
    if record is None:
        raise WebAPIError(401, "AUTH_REQUIRED", "Authentication is required.")
    return record


SessionDependency = Annotated[SessionRecord, Depends(_require_session)]
CsrfHeader = Annotated[str | None, Header(alias="X-CSRF-Token")]


def _require_csrf(
    session: SessionDependency,
    csrf_token: CsrfHeader = None,
) -> SessionRecord:
    if not csrf_matches(session.csrf_token, csrf_token):
        raise WebAPIError(403, "CSRF_REJECTED", "CSRF validation failed.")
    return session


CsrfSessionDependency = Annotated[SessionRecord, Depends(_require_csrf)]
FilePathParameter = Annotated[str, Query(min_length=1, max_length=1000)]
StartLineParameter = Annotated[int, Query(ge=1)]
MaxLinesParameter = Annotated[int, Query(ge=1, le=WEB_FILE_MAX_LINES)]
LastEventIdHeader = Annotated[str | None, Header(alias="Last-Event-ID")]


def _client_ip(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


def _error_response(error: WebAPIError, request_id: str | None = None) -> JSONResponse:
    payload: dict[str, Any] = {
        "error": {
            "code": error.code,
            "message": error.safe_message,
        }
    }
    if request_id is not None:
        payload["request_id"] = request_id
    headers = {}
    if error.retry_after_seconds is not None:
        payload["error"]["retry_after_seconds"] = error.retry_after_seconds
        headers["Retry-After"] = str(error.retry_after_seconds)
    return JSONResponse(content=payload, status_code=error.status_code, headers=headers)


def _production_components() -> tuple[WebSettings, Callable[[], ModelClient]]:
    try:
        load_project_env(PROJECT_ROOT / ".env", override=False)
        settings = WebSettings.from_environment()
        deepseek_config = DeepSeekConfig.from_environment()
    except (ConfigurationError, EnvFileError, WebConfigurationError) as exc:
        raise WebConfigurationError(str(exc)) from None
    if deepseek_config.api_key is None:
        raise WebConfigurationError("DeepSeek credentials are required for the Web app.")
    return settings, lambda: DeepSeekClient(deepseek_config)


def create_web_app(
    *,
    settings: WebSettings | None = None,
    model_client_factory: Callable[[], ModelClient] | None = None,
) -> FastAPI:
    """Construct the Web app without a second Agent loop or import-time side effects."""
    if settings is None:
        production_settings, production_factory = _production_components()
        settings = production_settings
        if model_client_factory is None:
            model_client_factory = production_factory
    elif model_client_factory is None:
        try:
            deepseek_config = DeepSeekConfig.from_environment()
        except ConfigurationError as exc:
            raise WebConfigurationError(str(exc)) from None
        if deepseek_config.api_key is None:
            raise WebConfigurationError("DeepSeek credentials are required for the Web app.")

        def default_model_client_factory() -> ModelClient:
            return DeepSeekClient(deepseek_config)

        model_client_factory = default_model_client_factory

    assert model_client_factory is not None
    session_manager = SessionManager(settings)
    run_manager = RunManager(settings, model_client_factory)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        del application
        session_manager.start()
        try:
            yield
        finally:
            await run_manager.shutdown()
            session_manager.shutdown()

    app = FastAPI(title="File Agent Web Demo", debug=False, lifespan=lifespan)
    app.state.web_settings = settings
    app.state.session_manager = session_manager
    app.state.run_manager = run_manager

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        return response

    @app.exception_handler(WebAPIError)
    async def handle_web_error(request: Request, exc: WebAPIError) -> JSONResponse:
        del request
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request, exc
        return _error_response(WebAPIError(422, "INVALID_REQUEST", "Request validation failed."))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        del request
        message = "Resource not found." if exc.status_code == 404 else "Request failed."
        return _error_response(WebAPIError(exc.status_code, "HTTP_ERROR", message))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = secrets.token_urlsafe(12)
        LOGGER.error("web_request_failed category=%s request_id=%s", type(exc).__name__, request_id)
        del request
        return _error_response(
            WebAPIError(500, "INTERNAL_ERROR", "The request failed safely."),
            request_id,
        )

    api = APIRouter(prefix="/api")

    @api.post("/auth")
    async def authenticate(body: AuthRequest, request: Request) -> JSONResponse:
        session_manager.cleanup_expired()
        if not access_code_matches(settings.access_code, body.access_code):
            raise WebAPIError(401, "AUTH_FAILED", "Authentication failed.")
        try:
            session = session_manager.create_session(client_ip=_client_ip(request))
        except SessionError as exc:
            raise WebAPIError(503, exc.code, exc.safe_message) from None
        payload = session.public_status()
        payload["csrf_token"] = session.csrf_token
        response = JSONResponse(payload)
        set_session_cookie(
            response,
            session.session_id,
            secure=settings.cookie_secure,
            max_age=settings.session_ttl_seconds,
        )
        return response

    @api.get("/session")
    async def session_status(
        session: SessionDependency,
    ) -> dict[str, object]:
        payload = session.public_status()
        payload["csrf_token"] = session.csrf_token
        return payload

    @api.post("/logout")
    async def logout(
        session: CsrfSessionDependency,
    ) -> JSONResponse:
        try:
            session_manager.remove_session(session.session_id)
        except SessionActiveRunError as exc:
            raise WebAPIError(409, exc.code, exc.safe_message) from None
        run_manager.forget_session(session.session_id)
        response = JSONResponse({"authenticated": False})
        clear_session_cookie(response, secure=settings.cookie_secure)
        return response

    @api.post("/workspace/reset")
    async def reset_workspace(
        session: CsrfSessionDependency,
    ) -> dict[str, object]:
        try:
            revision = session_manager.reset_workspace(session)
        except SessionActiveRunError as exc:
            raise WebAPIError(409, exc.code, exc.safe_message) from None
        except SessionError as exc:
            raise WebAPIError(500, exc.code, exc.safe_message) from None
        return {
            "reset": True,
            "workspace_revision": revision,
            "reset_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

    @api.get("/workspace/tree")
    async def workspace_tree(
        session: SessionDependency,
    ) -> dict[str, object]:
        try:
            entries = list_workspace_tree(session.workspace_path)
        except BrowserError as exc:
            raise WebAPIError(400, exc.code, exc.safe_message) from None
        return {"entries": entries, "count": len(entries)}

    @api.get("/workspace/file")
    async def workspace_file(
        path: FilePathParameter,
        session: SessionDependency,
        start_line: StartLineParameter = 1,
        max_lines: MaxLinesParameter = WEB_FILE_MAX_LINES,
    ) -> dict[str, object]:
        try:
            return read_workspace_file(
                session.workspace_path,
                path=path,
                start_line=start_line,
                max_lines=max_lines,
            )
        except BrowserError as exc:
            raise WebAPIError(400, exc.code, exc.safe_message) from None

    @api.post("/runs", status_code=202)
    async def start_run(
        body: TaskRequest,
        request: Request,
        session: CsrfSessionDependency,
    ) -> dict[str, object]:
        try:
            record = run_manager.start(session, task=body.task, client_ip=_client_ip(request))
        except RunManagerError as exc:
            raise WebAPIError(
                exc.status_code,
                exc.code,
                exc.safe_message,
                retry_after_seconds=exc.retry_after_seconds,
            ) from None
        return {"run_id": record.run_id, "status": "running"}

    @api.get("/runs/{run_id}")
    async def get_run(
        run_id: str,
        session: SessionDependency,
    ) -> dict[str, Any]:
        try:
            return run_manager.get(run_id, session.session_id).public_result()
        except RunNotFoundError as exc:
            raise WebAPIError(404, exc.code, exc.safe_message) from None

    @api.post("/runs/{run_id}/cancel")
    async def cancel_run(
        run_id: str,
        session: CsrfSessionDependency,
    ) -> dict[str, object]:
        try:
            record = run_manager.cancel(run_id, session.session_id)
        except RunNotFoundError as exc:
            raise WebAPIError(404, exc.code, exc.safe_message) from None
        status = "cancelling" if record.status == "running" else record.status
        return {"run_id": record.run_id, "status": status, "rollback": False}

    @api.get("/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        request: Request,
        session: SessionDependency,
        last_event_id_header: LastEventIdHeader = None,
    ) -> StreamingResponse:
        try:
            record = run_manager.get(run_id, session.session_id)
        except RunNotFoundError as exc:
            raise WebAPIError(404, exc.code, exc.safe_message) from None
        try:
            last_event_id = int(last_event_id_header or "0")
        except ValueError:
            raise WebAPIError(
                400,
                "INVALID_EVENT_ID",
                "Last-Event-ID must be an integer.",
            ) from None
        if last_event_id < 0:
            raise WebAPIError(400, "INVALID_EVENT_ID", "Last-Event-ID must not be negative.")
        generator = stream_events(
            record.backlog,
            request,
            last_event_id=last_event_id,
            keepalive_seconds=settings.sse_keepalive_seconds,
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @api.api_route(
        "/{unmatched_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def unknown_api(unmatched_path: str) -> None:
        del unmatched_path
        raise WebAPIError(404, "API_NOT_FOUND", "API endpoint not found.")

    app.include_router(api)

    @app.get("/", include_in_schema=False)
    async def frontend_index() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")
    return app
