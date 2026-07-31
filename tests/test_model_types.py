from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from app.model_types import (
    ModelFinishReason,
    ModelMessage,
    ModelResponse,
    ModelRole,
    ModelToolCall,
    ModelUsage,
)
from tests.fake_model import (
    FakeModelClient,
    FakeModelExhaustedError,
    implements_model_client,
)


def make_response(content: str) -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(role=ModelRole.ASSISTANT, content=content),
        usage=ModelUsage(
            input_tokens=2,
            output_tokens=1,
            total_tokens=3,
            exact=True,
        ),
        finish_reason=ModelFinishReason.STOP,
        raw_finish_reason="stop",
        provider_request_id=None,
        provider_model=None,
    )


def test_tool_message_requires_tool_call_id() -> None:
    with pytest.raises(ValidationError, match="must contain tool_call_id"):
        ModelMessage(role=ModelRole.TOOL, content="result")


def test_non_assistant_message_rejects_tool_calls() -> None:
    call = ModelToolCall.from_arguments(id="call-1", name="sample", arguments={})

    with pytest.raises(ValidationError, match="only assistant"):
        ModelMessage(role=ModelRole.USER, content="hello", tool_calls=[call])


def test_assistant_message_allows_empty_content_with_tool_calls() -> None:
    call = ModelToolCall.from_arguments(
        id="call-1",
        name="sample",
        arguments={"value": 1},
    )

    message = ModelMessage(
        role=ModelRole.ASSISTANT,
        content=None,
        tool_calls=[call],
    )

    assert message.content is None
    assert message.tool_calls == [call]


def test_model_tool_call_preserves_invalid_json() -> None:
    call = ModelToolCall(id="call-1", name="sample", arguments_json="{not-json")

    assert call.arguments_json == "{not-json"


def test_model_tool_call_from_arguments_uses_stable_json() -> None:
    call = ModelToolCall.from_arguments(
        id="call-1",
        name="sample",
        arguments={"z": "中文", "a": 1},
    )

    assert call.arguments_json == '{"a":1,"z":"中文"}'


def test_only_tool_messages_may_have_tool_call_id() -> None:
    with pytest.raises(ValidationError, match="only tool messages"):
        ModelMessage(
            role=ModelRole.ASSISTANT,
            content="done",
            tool_call_id="call-1",
        )


def test_model_usage_missing_values_remain_explicitly_unavailable() -> None:
    usage = ModelUsage(exact=True).to_usage_stats()

    assert usage.available is False
    assert usage.breakdown_available is False
    assert usage.exact is False
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0
    assert usage.total_available is False


def test_model_usage_total_only_does_not_fake_a_breakdown() -> None:
    usage = ModelUsage(total_tokens=17, exact=True).to_usage_stats()

    assert usage.available is True
    assert usage.breakdown_available is False
    assert usage.total_tokens == 17
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.exact is True
    assert usage.total_available is True


def test_model_usage_can_derive_total_from_complete_breakdown() -> None:
    usage = ModelUsage(input_tokens=5, output_tokens=2, exact=True).to_usage_stats()

    assert usage.total_tokens == 7
    assert usage.breakdown_available is True
    assert usage.exact is True
    assert usage.total_available is True


def test_inconsistent_provider_total_is_preserved_when_marked_inexact() -> None:
    normalized = ModelUsage(
        input_tokens=5,
        output_tokens=2,
        total_tokens=99,
        exact=False,
    )

    assert normalized.total_tokens == 99
    assert normalized.to_usage_stats().exact is False


def test_fake_model_returns_responses_in_order_and_records_inputs() -> None:
    first = make_response("first")
    second = make_response("second")
    client = FakeModelClient([first, second])
    messages = [ModelMessage(role=ModelRole.USER, content="hello")]
    tools = [{"type": "function", "function": {"name": "sample"}}]

    first_result = asyncio.run(client.complete(messages, tools))
    messages[0] = ModelMessage(role=ModelRole.USER, content="changed")
    tools[0]["function"]["name"] = "changed"
    second_result = asyncio.run(client.complete([], []))

    assert first_result.message.content == "first"
    assert second_result.message.content == "second"
    assert client.calls[0].messages[0].content == "hello"
    assert client.calls[0].tools[0]["function"]["name"] == "sample"
    assert implements_model_client(client) is True


def test_fake_model_recording_preserves_raw_invalid_tool_json() -> None:
    invalid_call = ModelToolCall(
        id="call-invalid",
        name="sample",
        arguments_json="{invalid-json",
    )
    client = FakeModelClient([make_response("done")])
    messages = [
        ModelMessage(
            role=ModelRole.ASSISTANT,
            content=None,
            tool_calls=[invalid_call],
        )
    ]

    asyncio.run(client.complete(messages, []))

    assert client.calls[0].messages[0].tool_calls[0].arguments_json == "{invalid-json"


def test_fake_model_exhaustion_is_explicit() -> None:
    client = FakeModelClient([])

    with pytest.raises(FakeModelExhaustedError, match="exhausted"):
        asyncio.run(client.complete([], []))


def test_fake_model_can_raise_on_a_configured_call() -> None:
    client = FakeModelClient(
        [make_response("first"), make_response("second")],
        failures={2: TimeoutError("configured timeout")},
    )

    assert asyncio.run(client.complete([], [])).message.content == "first"
    with pytest.raises(TimeoutError, match="configured timeout"):
        asyncio.run(client.complete([], []))
    assert asyncio.run(client.complete([], [])).message.content == "second"
    assert len(client.calls) == 3
