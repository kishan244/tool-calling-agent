"""Tools the agent can call. Each one is a plain Python function."""

import ast
import operator

from ddgs import DDGS


def web_search(query: str) -> str:
    """Return a short summary of the top search results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return "No search results found."

        return "\n".join(f"- {r['title']}: {r['body']}" for r in results)

    except Exception as e:
        return f"Search failed: {e}"


# Only these operations are allowed. Anything else in the expression is rejected,
# so the calculator can never run arbitrary code (no eval).
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
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_evaluate(tree))
    except ZeroDivisionError:
        return "Calculation failed: division by zero"
    except (SyntaxError, ValueError, TypeError) as e:
        return f"Calculation failed: {e}"


AVAILABLE_TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
}
