from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

import app.deepseek_client as deepseek_module
from app.config import DeepSeekConfig
from app.deepseek_client import (
    DeepSeekAuthenticationError,
    DeepSeekClient,
    DeepSeekClientError,
    DeepSeekConfigurationError,
    DeepSeekConnectionError,
    DeepSeekProtocolError,
    DeepSeekRateLimitError,
    DeepSeekServerError,
    DeepSeekTimeoutError,
    normalize_deepseek_response,
    prepare_deepseek_tools,
    to_deepseek_messages,
)
from app.model_types import ModelFinishReason, ModelMessage, ModelRole, ModelToolCall


def function_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "sample_tool",
            "description": "Perform a sample operation.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }


def sdk_tool_call(
    call_id: str = "call-1",
    name: str = "sample_tool",
    arguments: str = '{"value":"x"}',
    call_type: str = "function",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type=call_type,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def sdk_response(
    *,
    content: str | None = "done",
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = "stop",
    usage: Any = None,
    choices: list[Any] | None = None,
    request_id: str | None = "request-1",
    model: str | None = "provider-model",
) -> SimpleNamespace:
    if choices is None:
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
    response = SimpleNamespace(
        choices=choices,
        usage=usage,
        model=model,
        request_id="must-not-be-used",
    )
    response._request_id = request_id
    return response


def fake_sdk(response: Any = None, *, failure: BaseException | None = None) -> Any:
    create = AsyncMock(return_value=response, side_effect=failure)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        create_mock=create,
    )


def test_message_conversion_covers_all_roles_and_preserves_raw_arguments() -> None:
    raw_arguments = '{ "z": 2, "a": [1,  3] '
    calls = [
        ModelToolCall(id="call-1", name="sample_tool", arguments_json=raw_arguments),
        ModelToolCall(id="call-2", name="sample_tool", arguments_json="{}"),
    ]
    messages = [
        ModelMessage(role=ModelRole.SYSTEM, content="system"),
        ModelMessage(role=ModelRole.USER, content="user"),
        ModelMessage(role=ModelRole.ASSISTANT, content="assistant"),
        ModelMessage(role=ModelRole.ASSISTANT, content=None, tool_calls=calls),
        ModelMessage(role=ModelRole.TOOL, tool_call_id="call-1", content='{"ok":true}'),
    ]
    before = [message.model_copy(deep=True) for message in messages]

    converted = to_deepseek_messages(messages)

    assert converted == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": "assistant"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "sample_tool", "arguments": raw_arguments},
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "sample_tool", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"ok":true}'},
    ]
    assert messages == before
    assert all(isinstance(item, dict) for item in converted)


def test_tool_message_content_must_be_a_string() -> None:
    message = ModelMessage(role=ModelRole.TOOL, tool_call_id="call-1", content=None)

    with pytest.raises(DeepSeekConfigurationError):
        to_deepseek_messages([message])


def test_tool_preparation_preserves_schema_and_returns_independent_json() -> None:
    original = function_tool()
    expected = deepcopy(original)

    prepared = prepare_deepseek_tools([original])
    prepared[0]["function"]["parameters"]["properties"]["value"]["type"] = "number"

    assert original == expected
    assert expected["function"]["parameters"]["additionalProperties"] is False
    assert "strict" not in expected["function"]


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "custom", "function": {}},
        {"type": "function", "function": {"name": "x", "parameters": {}}},
        {
            "type": "function",
            "function": {"name": "x", "description": "x", "parameters": []},
        },
        {
            "type": "function",
            "function": {
                "name": "x",
                "description": "x",
                "parameters": {},
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {"name": "x", "description": "x", "parameters": {"bad": {1}}},
        },
    ],
)
def test_tool_preparation_rejects_invalid_or_non_json_schemas(tool: dict[str, Any]) -> None:
    with pytest.raises(DeepSeekConfigurationError):
        prepare_deepseek_tools([tool])


def test_request_uses_only_required_chat_completion_parameters() -> None:
    response = sdk_response(
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    )
    sdk = fake_sdk(response)
    config = DeepSeekConfig(
        model="deepseek-v4-pro",
        temperature=0.25,
        max_output_tokens=1234,
    )
    client = DeepSeekClient(config, sdk)
    messages = [ModelMessage(role=ModelRole.USER, content="hello")]
    tools = [function_tool()]

    normalized = asyncio.run(client.complete(messages, tools))

    sdk.create_mock.assert_awaited_once()
    kwargs = sdk.create_mock.await_args.kwargs
    assert kwargs == {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.25,
        "max_tokens": 1234,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    assert "stream" not in kwargs
    assert "strict" not in str(kwargs)
    assert "reasoning_content" not in str(kwargs)
    assert normalized.message.content == "done"


def test_default_sdk_client_receives_transport_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    injected = fake_sdk()

    def factory(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return injected

    monkeypatch.setattr(deepseek_module, "AsyncOpenAI", factory)
    config = DeepSeekConfig(
        api_key="constructor-secret",
        base_url="https://example.test/v1",
        timeout_seconds=9,
        max_retries=3,
    )

    DeepSeekClient(config)

    assert captured == {
        "api_key": "constructor-secret",
        "base_url": "https://example.test/v1",
        "timeout": 9.0,
        "max_retries": 3,
    }


def test_default_sdk_client_requires_key_but_injected_client_does_not() -> None:
    with pytest.raises(DeepSeekConfigurationError):
        DeepSeekClient(DeepSeekConfig())

    assert DeepSeekClient(DeepSeekConfig(), fake_sdk())


def test_response_normalizes_text_usage_and_public_metadata() -> None:
    response = sdk_response(
        content="answer",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        request_id="public-request-id",
        model="deepseek-v4-flash-202607",
    )

    normalized = normalize_deepseek_response(response)

    assert normalized.message.content == "answer"
    assert normalized.message.tool_calls == []
    assert normalized.usage.model_dump() == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "exact": True,
    }
    assert normalized.provider_request_id == "public-request-id"
    assert normalized.provider_model == "deepseek-v4-flash-202607"


def test_response_preserves_content_and_multiple_raw_tool_calls() -> None:
    malformed = "{not-json"
    response = sdk_response(
        content="preface",
        tool_calls=[
            sdk_tool_call(arguments=malformed),
            sdk_tool_call(call_id="call-2", arguments='{ "value": 2 }'),
        ],
        finish_reason="tool_calls",
    )

    normalized = normalize_deepseek_response(response)

    assert normalized.message.content == "preface"
    assert [call.id for call in normalized.message.tool_calls] == ["call-1", "call-2"]
    assert normalized.message.tool_calls[0].arguments_json == malformed
    assert normalized.message.tool_calls[1].arguments_json == '{ "value": 2 }'
    assert normalized.finish_reason is ModelFinishReason.TOOL_CALLS


def test_multiple_choices_deterministically_select_the_first() -> None:
    choices = [
        SimpleNamespace(
            message=SimpleNamespace(content="first", tool_calls=None), finish_reason="stop"
        ),
        SimpleNamespace(
            message=SimpleNamespace(content="second", tool_calls=None), finish_reason="stop"
        ),
    ]

    normalized = normalize_deepseek_response(sdk_response(choices=choices))

    assert normalized.message.content == "first"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", ModelFinishReason.STOP),
        ("tool_calls", ModelFinishReason.TOOL_CALLS),
        ("length", ModelFinishReason.LENGTH),
        ("content_filter", ModelFinishReason.CONTENT_FILTER),
        ("insufficient_system_resource", ModelFinishReason.PROVIDER_RESOURCE),
        ("future_reason", ModelFinishReason.UNKNOWN),
        (None, ModelFinishReason.UNKNOWN),
    ],
)
def test_all_finish_reasons_are_normalized(
    raw: str | None,
    expected: ModelFinishReason,
) -> None:
    normalized = normalize_deepseek_response(sdk_response(finish_reason=raw))

    assert normalized.finish_reason is expected
    assert normalized.raw_finish_reason == raw


def test_missing_usage_remains_none_instead_of_becoming_zero() -> None:
    normalized = normalize_deepseek_response(sdk_response(usage=None))

    assert normalized.usage.input_tokens is None
    assert normalized.usage.output_tokens is None
    assert normalized.usage.total_tokens is None
    assert normalized.usage.exact is False


def test_inconsistent_provider_total_is_preserved_and_marked_inexact() -> None:
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=99)

    normalized = normalize_deepseek_response(sdk_response(usage=usage))

    assert normalized.usage.total_tokens == 99
    assert normalized.usage.exact is False


@pytest.mark.parametrize(
    "response",
    [
        sdk_response(choices=[]),
        sdk_response(content=123),
        sdk_response(tool_calls=[sdk_tool_call(call_type="custom")]),
        sdk_response(tool_calls=[sdk_tool_call(arguments=123)]),
    ],
)
def test_malformed_provider_responses_raise_safe_protocol_error(response: Any) -> None:
    with pytest.raises(DeepSeekProtocolError) as captured:
        normalize_deepseek_response(response)

    assert str(captured.value) == "DeepSeek protocol error"


def test_protocol_validation_does_not_attach_raw_tool_arguments() -> None:
    sensitive_arguments = '{"token":"unit-test-sensitive-value"}'
    response = sdk_response(
        tool_calls=[sdk_tool_call(name="", arguments=sensitive_arguments)],
        finish_reason="tool_calls",
    )

    with pytest.raises(DeepSeekProtocolError) as captured:
        normalize_deepseek_response(response)

    assert sensitive_arguments not in str(captured.value)
    assert captured.value.__cause__ is None


def sdk_status_error(error_type: type[Exception], status: int) -> Exception:
    request = httpx.Request("POST", "https://example.test/chat/completions")
    response = httpx.Response(status, request=request, headers={"x-request-id": "safe-id"})
    return error_type(
        "Authorization Bearer unit-test-secret full request body",
        response=response,
        body={"api_key": "unit-test-secret"},
    )


@pytest.mark.parametrize(
    ("failure", "expected_type", "retryable"),
    [
        (sdk_status_error(AuthenticationError, 401), DeepSeekAuthenticationError, False),
        (sdk_status_error(RateLimitError, 429), DeepSeekRateLimitError, True),
        (
            APITimeoutError(httpx.Request("POST", "https://example.test/chat/completions")),
            DeepSeekTimeoutError,
            True,
        ),
        (
            APIConnectionError(
                request=httpx.Request("POST", "https://example.test/chat/completions")
            ),
            DeepSeekConnectionError,
            True,
        ),
        (sdk_status_error(InternalServerError, 503), DeepSeekServerError, True),
        (RuntimeError("unit-test-secret full request body"), DeepSeekClientError, False),
    ],
)
def test_sdk_errors_are_safely_normalized(
    failure: BaseException,
    expected_type: type[DeepSeekClientError],
    retryable: bool,
) -> None:
    client = DeepSeekClient(DeepSeekConfig(), fake_sdk(failure=failure))

    with pytest.raises(expected_type) as captured:
        asyncio.run(client.complete([ModelMessage(role=ModelRole.USER, content="task")], []))

    error = captured.value
    assert error.retryable is retryable
    assert "unit-test-secret" not in str(error)
    assert "full request body" not in str(error)
    assert error.__context__ is None
    if isinstance(failure, (AuthenticationError, RateLimitError, InternalServerError)):
        assert error.request_id == "safe-id"


def test_normalized_timeout_is_recognizable_by_the_agent_boundary() -> None:
    assert issubclass(DeepSeekTimeoutError, TimeoutError)
