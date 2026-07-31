"""An MCP-style tool registry with explicit per-agent permission scopes.

This module is a deliberately small stand-in for the kind of "tool /
function calling gateway" that shows up in modern AI integration protocols
(Anthropic tool use, OpenAI function calling, and the Model Context Protocol
all converge on the same shape: a named tool, a JSON-schema-like input spec,
and a handler). The addition here that is easy to skip in a toy demo -- and
that matters a great deal in a real advisory-firm deployment -- is an
explicit allow-list per caller. An agent may only invoke tools it has been
granted, and any attempt to call an unauthorized tool raises a
``PermissionError`` instead of silently succeeding.

This is the "AI service interface / gateway design pattern" referenced in
the project README: a single, auditable choke point through which every
tool call must pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


class ToolNotFoundError(KeyError):
    """Raised when a tool name is not registered with the registry."""


class ToolPermissionError(PermissionError):
    """Raised when a caller attempts to invoke a tool outside its allow-list."""


class ToolInputError(ValueError):
    """Raised when a tool is called with inputs that violate its input spec."""


@dataclass(frozen=True)
class ToolSpec:
    """Describes a single registered tool.

    Attributes:
        name: Unique tool identifier, e.g. ``"document_parser"``.
        description: Human-readable description of what the tool does.
        input_schema: A JSON-schema-like mapping describing accepted keyword
            arguments. Only the ``"required"`` (list[str]) and
            ``"properties"`` (mapping of field name -> {"type": str}) keys
            are interpreted by the lightweight validator in this module;
            the shape intentionally mirrors real JSON Schema / MCP tool
            definitions so it reads naturally to anyone familiar with them.
        handler: The callable that actually performs the tool's work.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: Callable[..., Any]


_PY_TYPE_BY_JSON_TYPE = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _validate_against_schema(tool_name: str, schema: Mapping[str, Any], kwargs: Mapping[str, Any]) -> None:
    """Validates ``kwargs`` against a minimal JSON-schema-like ``schema``.

    Only checks required fields are present and, when a ``"type"`` is given
    for a property, that the supplied value has a compatible Python type.
    This is intentionally lightweight -- it is not a full JSON Schema
    implementation -- but it is enough to catch the common integration bugs
    (missing argument, wrong type) that a real tool gateway must guard
    against.
    """
    required = schema.get("required", [])
    missing = [field_name for field_name in required if field_name not in kwargs]
    if missing:
        raise ToolInputError(
            f"Tool '{tool_name}' call is missing required argument(s): {missing}"
        )

    properties = schema.get("properties", {})
    for field_name, value in kwargs.items():
        field_spec = properties.get(field_name)
        if not field_spec:
            continue
        expected_type = field_spec.get("type")
        py_type = _PY_TYPE_BY_JSON_TYPE.get(expected_type)
        if py_type is not None and not isinstance(value, py_type):
            raise ToolInputError(
                f"Tool '{tool_name}' argument '{field_name}' expected type "
                f"'{expected_type}' but got {type(value).__name__}"
            )


@dataclass
class ToolCallRecord:
    """An audit-log entry for a single tool invocation."""

    caller: str
    tool_name: str
    kwargs: Mapping[str, Any]
    result_repr: str


@dataclass
class ToolRegistry:
    """Central registry of tools plus per-caller permission enforcement.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(ToolSpec(
        ...     name="add",
        ...     description="Add two numbers",
        ...     input_schema={"required": ["a", "b"], "properties": {}},
        ...     handler=lambda a, b: a + b,
        ... ))
        >>> registry.grant("researcher_agent", ["add"])
        >>> registry.call("researcher_agent", "add", a=1, b=2)
        3
    """

    _tools: dict[str, ToolSpec] = field(default_factory=dict)
    _grants: dict[str, set[str]] = field(default_factory=dict)
    call_log: list[ToolCallRecord] = field(default_factory=list)

    def register(self, spec: ToolSpec) -> None:
        """Registers a new tool. Overwrites any existing tool of the same name."""
        self._tools[spec.name] = spec

    def grant(self, caller: str, tool_names: Sequence[str]) -> None:
        """Grants ``caller`` permission to invoke each name in ``tool_names``."""
        self._grants.setdefault(caller, set()).update(tool_names)

    def allowed_tools(self, caller: str) -> frozenset[str]:
        """Returns the frozen set of tool names ``caller`` is permitted to call."""
        return frozenset(self._grants.get(caller, set()))

    def list_tools(self) -> list[ToolSpec]:
        """Returns all registered tool specs, sorted by name."""
        return [self._tools[name] for name in sorted(self._tools)]

    def call(self, caller: str, tool_name: str, **kwargs: Any) -> Any:
        """Invokes ``tool_name`` on behalf of ``caller``, enforcing permissions.

        Raises:
            ToolPermissionError: If ``caller`` has not been granted ``tool_name``.
            ToolNotFoundError: If ``tool_name`` is not registered.
            ToolInputError: If ``kwargs`` fail the tool's input schema check.
        """
        if tool_name not in self._grants.get(caller, set()):
            raise ToolPermissionError(
                f"Caller '{caller}' is not permitted to call tool '{tool_name}'. "
                f"Allowed tools: {sorted(self._grants.get(caller, set()))}"
            )
        if tool_name not in self._tools:
            raise ToolNotFoundError(f"No tool registered with name '{tool_name}'")

        spec = self._tools[tool_name]
        _validate_against_schema(tool_name, spec.input_schema, kwargs)
        result = spec.handler(**kwargs)
        self.call_log.append(
            ToolCallRecord(
                caller=caller,
                tool_name=tool_name,
                kwargs=dict(kwargs),
                result_repr=repr(result)[:200],
            )
        )
        return result
