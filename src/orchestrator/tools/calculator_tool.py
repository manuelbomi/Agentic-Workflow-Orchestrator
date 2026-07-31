"""A safe arithmetic expression evaluator tool.

Agents occasionally need to compute a derived statistic mentioned in an
analysis (e.g. a growth rate or a ratio). Rather than handing an agent
Python's ``eval()`` against arbitrary text -- a classic and dangerous
shortcut -- this tool parses and evaluates a restricted arithmetic grammar
using Python's ``ast`` module, allow-listing only numeric literals and the
operators below. Anything else (attribute access, function calls, name
lookups, string literals, etc.) is rejected before it can be evaluated.
"""

from __future__ import annotations

import ast
import operator
from typing import Callable

from orchestrator.tool_registry import ToolSpec

_BINARY_OPERATORS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS: dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains a construct outside the allow-list."""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise UnsafeExpressionError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise UnsafeExpressionError(f"Unsupported binary operator: {node.op!r}")
        return op_fn(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise UnsafeExpressionError(f"Unsupported unary operator: {node.op!r}")
        return op_fn(_eval_node(node.operand))
    raise UnsafeExpressionError(f"Unsupported expression element: {type(node).__name__}")


def safe_calculate(expression: str) -> float:
    """Safely evaluates a simple arithmetic expression string.

    Supports ``+ - * / ** %``, unary +/-, parentheses, and int/float
    literals only. Does not use ``eval()``; instead it parses the
    expression into a Python AST and walks a small allow-listed subset of
    node types.

    Args:
        expression: An arithmetic expression, e.g. ``"(120 - 100) / 100 * 100"``.

    Returns:
        The numeric result of evaluating ``expression``.

    Raises:
        UnsafeExpressionError: If the expression contains any construct
            outside the allow-listed grammar (names, calls, strings,
            comparisons, etc.), or is not syntactically valid.
    """
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Could not parse expression: {expression!r}") from exc
    return _eval_node(parsed)


def build_spec() -> ToolSpec:
    """Builds the ``ToolSpec`` for registering this tool with a ``ToolRegistry``."""
    return ToolSpec(
        name="calculator",
        description="Safely evaluates a restricted arithmetic expression (no eval()).",
        input_schema={
            "required": ["expression"],
            "properties": {"expression": {"type": "string"}},
        },
        handler=safe_calculate,
    )
