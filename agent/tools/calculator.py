"""A calculator that can never run code: it only walks a parsed expression."""
from __future__ import annotations

import ast
import operator

from .base import Tool, ToolError

# Only these operations are allowed. Anything else in the expression is rejected (no eval).
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENT = 100  # stops expressions like 9**9**9 from freezing the program
_MAX_LENGTH = 200


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("only numbers and + - * / // % ** ( ) are allowed")


def calculator(expression: str) -> str:
    """Evaluate a basic math expression safely."""
    if len(expression) > _MAX_LENGTH:
        raise ToolError("Calculation failed: expression too long")
    try:
        value = _evaluate(ast.parse(expression.strip(), mode="eval"))
    except ZeroDivisionError:
        raise ToolError("Calculation failed: division by zero") from None
    except (SyntaxError, ValueError, TypeError, OverflowError) as error:
        raise ToolError(f"Calculation failed: {error}") from None
    if isinstance(value, float):
        value = round(value, 10)  # hides float noise such as 0.30000000000000004
        if value.is_integer():
            value = int(value)
    return str(value)


CALCULATOR = Tool(
    name="calculator",
    description=(
        "Evaluate a math expression exactly. Use it for every calculation (differences, ratios, sums) "
        "instead of computing in your head. Supports + - * / // % ** and brackets."
    ),
    func=calculator,
    parameters={"expression": {"type": "string", "description": "A math expression, e.g. '(31.2 - 17.4) / 17.4 * 100'."}},
    required=("expression",),
)
