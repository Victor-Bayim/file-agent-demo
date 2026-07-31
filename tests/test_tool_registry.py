from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from app.tools import DuplicateToolError, ToolRegistry, ToolSpec, UnknownToolError


class SampleArgs(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1)


def make_spec(name: str = "sample_tool") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Perform a deterministic sample operation.",
        args_model=SampleArgs,
        is_mutating=False,
    )


@pytest.mark.parametrize("name", ["Uppercase", "has-dash", "has space", ""])
def test_tool_spec_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValidationError, match="lowercase"):
        make_spec(name)


def test_tool_spec_rejects_empty_description() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        ToolSpec(
            name="sample",
            description="   ",
            args_model=SampleArgs,
            is_mutating=False,
        )


def test_tool_spec_generates_stable_provider_neutral_schema() -> None:
    spec = make_spec()

    first = spec.model_schema()
    second = spec.model_schema()

    assert first == second
    assert list(first) == ["type", "function"]
    assert first["type"] == "function"
    assert first["function"]["name"] == "sample_tool"
    assert first["function"]["description"] == spec.description
    assert first["function"]["parameters"] == SampleArgs.model_json_schema()
    assert first["function"]["parameters"]["required"] == ["query"]


def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.register(make_spec())

    with pytest.raises(DuplicateToolError, match="already registered"):
        registry.register(make_spec())


def test_registry_rejects_unknown_names() -> None:
    registry = ToolRegistry()

    with pytest.raises(UnknownToolError, match="Unknown tool"):
        registry.get("missing")


def test_registry_preserves_order_and_does_not_expose_internal_list() -> None:
    registry = ToolRegistry()
    first = make_spec("first")
    second = make_spec("second")
    registry.register(first)
    registry.register(second)

    returned = registry.list_specs()
    returned.clear()

    assert registry.list_specs() == [first, second]
    assert registry.get("first") is first
    assert [schema["function"]["name"] for schema in registry.model_schemas()] == [
        "first",
        "second",
    ]


def test_model_schema_callers_cannot_mutate_future_results() -> None:
    registry = ToolRegistry()
    registry.register(make_spec())
    schemas = registry.model_schemas()
    schemas[0]["function"]["name"] = "changed"

    assert registry.model_schemas()[0]["function"]["name"] == "sample_tool"


def test_new_registry_contains_no_real_tools() -> None:
    assert ToolRegistry().list_specs() == []
