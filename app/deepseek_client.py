"""DeepSeek Chat Completions adapter with provider-neutral outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

from app.config import DeepSeekConfig
from app.model_types import (
    ModelClient,
    ModelFinishReason,
    ModelMessage,
    ModelResponse,
    ModelRole,
    ModelToolCall,
    ModelUsage,
)


class _CompletionsClient(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _ChatClient(Protocol):
    completions: _CompletionsClient


class AsyncOpenAICompatibleClient(Protocol):
    chat: _ChatClient


class DeepSeekClientError(Exception):
    """Safe provider error containing metadata but never request content."""

    category = "client"

    def __init__(
        self,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        self.status_code = status_code
        self.request_id = request_id
        self.retryable = retryable
        suffix = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"DeepSeek {self.category} error{suffix}")


class DeepSeekConfigurationError(DeepSeekClientError):
    category = "configuration"


class DeepSeekProtocolError(DeepSeekClientError):
    category = "protocol"


class DeepSeekAuthenticationError(DeepSeekClientError):
    category = "authentication"


class DeepSeekRateLimitError(DeepSeekClientError):
    category = "rate_limit"


class DeepSeekTimeoutError(DeepSeekClientError, TimeoutError):
    category = "timeout"


class DeepSeekConnectionError(DeepSeekClientError):
    category = "connection"


class DeepSeekServerError(DeepSeekClientError):
    category = "server"


_FINISH_REASONS = {
    "stop": ModelFinishReason.STOP,
    "tool_calls": ModelFinishReason.TOOL_CALLS,
    "length": ModelFinishReason.LENGTH,
    "content_filter": ModelFinishReason.CONTENT_FILTER,
    "insufficient_system_resource": ModelFinishReason.PROVIDER_RESOURCE,
}


def to_deepseek_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """Convert without mutating messages or rewriting raw tool arguments."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role is ModelRole.TOOL:
            if not isinstance(message.content, str):
                raise DeepSeekConfigurationError()
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
            continue

        item: dict[str, Any] = {
            "role": message.role.value,
            "content": message.content,
        }
        if message.role is ModelRole.ASSISTANT and message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments_json,
                    },
                }
                for call in message.tool_calls
            ]
        converted.append(item)
    return converted


def prepare_deepseek_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate function tool envelopes and return independent plain JSON values."""
    try:
        prepared = json.loads(json.dumps(tools, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        raise DeepSeekConfigurationError() from None
    if not isinstance(prepared, list):
        raise DeepSeekConfigurationError()
    for tool in prepared:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise DeepSeekConfigurationError()
        function = tool.get("function")
        if not isinstance(function, dict) or "strict" in function:
            raise DeepSeekConfigurationError()
        if not isinstance(function.get("name"), str) or not function["name"].strip():
            raise DeepSeekConfigurationError()
        if not isinstance(function.get("description"), str):
            raise DeepSeekConfigurationError()
        if not isinstance(function.get("parameters"), dict):
            raise DeepSeekConfigurationError()
    return prepared


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _token_value(usage: Any, name: str) -> int | None:
    value = _value(usage, name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeepSeekProtocolError()
    return value


def _normalize_usage(raw_usage: Any) -> ModelUsage:
    if raw_usage is None:
        return ModelUsage(exact=False)
    input_tokens = _token_value(raw_usage, "prompt_tokens")
    output_tokens = _token_value(raw_usage, "completion_tokens")
    total_tokens = _token_value(raw_usage, "total_tokens")
    supplied = (input_tokens, output_tokens, total_tokens)
    inconsistent = (
        input_tokens is not None
        and output_tokens is not None
        and total_tokens is not None
        and total_tokens != input_tokens + output_tokens
    )
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        exact=any(value is not None for value in supplied) and not inconsistent,
    )


def normalize_deepseek_response(response: Any) -> ModelResponse:
    """Normalize a completion, deterministically selecting the first choice."""
    choices = _value(response, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise DeepSeekProtocolError()
    choice = choices[0]
    raw_message = _value(choice, "message")
    if raw_message is None:
        raise DeepSeekProtocolError()
    content = _value(raw_message, "content")
    if content is not None and not isinstance(content, str):
        raise DeepSeekProtocolError()

    normalized_calls: list[ModelToolCall] = []
    raw_calls = _value(raw_message, "tool_calls", []) or []
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        raise DeepSeekProtocolError()
    for raw_call in raw_calls:
        if _value(raw_call, "type") != "function":
            raise DeepSeekProtocolError()
        function = _value(raw_call, "function")
        call_id = _value(raw_call, "id")
        name = _value(function, "name")
        arguments = _value(function, "arguments")
        if not all(isinstance(value, str) for value in (call_id, name, arguments)):
            raise DeepSeekProtocolError()
        try:
            normalized_calls.append(ModelToolCall(id=call_id, name=name, arguments_json=arguments))
        except ValueError:
            raise DeepSeekProtocolError() from None

    raw_finish = _value(choice, "finish_reason")
    raw_finish_reason = raw_finish if isinstance(raw_finish, str) else None
    request_id = _value(response, "_request_id")
    provider_model = _value(response, "model")
    return ModelResponse(
        message=ModelMessage(
            role=ModelRole.ASSISTANT,
            content=content,
            tool_calls=normalized_calls,
        ),
        usage=_normalize_usage(_value(response, "usage")),
        finish_reason=_FINISH_REASONS.get(raw_finish_reason, ModelFinishReason.UNKNOWN),
        raw_finish_reason=raw_finish_reason,
        provider_request_id=request_id if isinstance(request_id, str) else None,
        provider_model=provider_model if isinstance(provider_model, str) else None,
    )


def normalize_deepseek_error(exc: BaseException) -> DeepSeekClientError:
    """Map SDK failures without copying their potentially sensitive messages or bodies."""
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    safe_status = status_code if isinstance(status_code, int) else None
    safe_request_id = request_id if isinstance(request_id, str) else None
    metadata = {"status_code": safe_status, "request_id": safe_request_id}
    if isinstance(exc, AuthenticationError):
        return DeepSeekAuthenticationError(**metadata)
    if isinstance(exc, RateLimitError):
        return DeepSeekRateLimitError(**metadata, retryable=True)
    if isinstance(exc, APITimeoutError):
        return DeepSeekTimeoutError(**metadata, retryable=True)
    if isinstance(exc, APIConnectionError):
        return DeepSeekConnectionError(**metadata, retryable=True)
    if isinstance(exc, InternalServerError) or (
        isinstance(exc, APIStatusError) and safe_status is not None and safe_status >= 500
    ):
        return DeepSeekServerError(**metadata, retryable=True)
    return DeepSeekClientError(**metadata)


class DeepSeekClient(ModelClient):
    """One logical ``complete`` call; SDK HTTP retries remain internal to that call."""

    def __init__(
        self,
        config: DeepSeekConfig,
        client: AsyncOpenAICompatibleClient | None = None,
    ) -> None:
        self._config = config.model_copy(deep=True)
        if client is None:
            if config.api_key is None:
                raise DeepSeekConfigurationError()
            client = AsyncOpenAI(
                api_key=config.api_key.get_secret_value(),
                base_url=config.base_url,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
            )
        self._client = client

    async def complete(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        request_messages = to_deepseek_messages(messages)
        request_tools = prepare_deepseek_tools(tools)
        normalized_error: DeepSeekClientError | None = None
        try:
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=request_messages,
                tools=request_tools,
                tool_choice="auto",
                temperature=self._config.temperature,
                max_tokens=self._config.max_output_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except DeepSeekClientError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK exceptions normalize here
            normalized_error = normalize_deepseek_error(exc)
            response = None
        if normalized_error is not None:
            raise normalized_error
        return normalize_deepseek_response(response)
