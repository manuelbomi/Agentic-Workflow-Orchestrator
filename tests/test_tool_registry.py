"""Tests for the MCP-style tool registry and its permission enforcement."""

from __future__ import annotations

import pytest

from orchestrator.tool_registry import (
    ToolInputError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolRegistry,
    ToolSpec,
)


def _add_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="add",
        description="Adds two integers",
        input_schema={
            "required": ["a", "b"],
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
        handler=lambda a, b: a + b,
    )


def test_registered_tool_is_callable_by_granted_caller() -> None:
    registry = ToolRegistry()
    registry.register(_add_tool_spec())
    registry.grant("researcher_agent", ["add"])

    result = registry.call("researcher_agent", "add", a=2, b=3)

    assert result == 5


def test_unauthorized_tool_call_is_rejected() -> None:
    """An agent that was never granted a tool must not be able to call it."""
    registry = ToolRegistry()
    registry.register(_add_tool_spec())
    registry.grant("researcher_agent", ["add"])

    with pytest.raises(ToolPermissionError):
        registry.call("writer_agent", "add", a=1, b=1)


def test_granting_one_tool_does_not_grant_others() -> None:
    registry = ToolRegistry()
    registry.register(_add_tool_spec())
    registry.register(
        ToolSpec(
            name="subtract",
            description="Subtracts b from a",
            input_schema={"required": ["a", "b"], "properties": {}},
            handler=lambda a, b: a - b,
        )
    )
    registry.grant("analyst_agent", ["add"])

    assert registry.call("analyst_agent", "add", a=5, b=2) == 7
    with pytest.raises(ToolPermissionError):
        registry.call("analyst_agent", "subtract", a=5, b=2)


def test_calling_unregistered_tool_raises_not_found() -> None:
    registry = ToolRegistry()
    registry.grant("researcher_agent", ["does_not_exist"])

    with pytest.raises(ToolNotFoundError):
        registry.call("researcher_agent", "does_not_exist")


def test_missing_required_argument_raises_input_error() -> None:
    registry = ToolRegistry()
    registry.register(_add_tool_spec())
    registry.grant("researcher_agent", ["add"])

    with pytest.raises(ToolInputError):
        registry.call("researcher_agent", "add", a=1)


def test_wrong_argument_type_raises_input_error() -> None:
    registry = ToolRegistry()
    registry.register(_add_tool_spec())
    registry.grant("researcher_agent", ["add"])

    with pytest.raises(ToolInputError):
        registry.call("researcher_agent", "add", a="not-an-int", b=2)


def test_allowed_tools_reflects_grants() -> None:
    registry = ToolRegistry()
    registry.register(_add_tool_spec())
    registry.grant("researcher_agent", ["add"])

    assert registry.allowed_tools("researcher_agent") == frozenset({"add"})
    assert registry.allowed_tools("unknown_agent") == frozenset()


def test_successful_calls_are_logged() -> None:
    registry = ToolRegistry()
    registry.register(_add_tool_spec())
    registry.grant("researcher_agent", ["add"])

    registry.call("researcher_agent", "add", a=1, b=1)

    assert len(registry.call_log) == 1
    assert registry.call_log[0].caller == "researcher_agent"
    assert registry.call_log[0].tool_name == "add"
