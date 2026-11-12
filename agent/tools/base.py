"""What a tool is, and how a tool call is checked and run safely."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

MAX_RESULT_CHARS = 4000  # a long tool result costs tokens on every later step

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


class ToolError(Exception):
    """A tool raises this for a problem the model can fix (bad input, no data)."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: Callable[..., str]
    parameters: dict[str, dict] = field(default_factory=dict)  # name -> {"type": ..., "description": ...}
    required: tuple[str, ...] = ()

    def schema(self) -> dict:
        """The tool in the format the Anthropic API expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {"type": "object", "properties": self.parameters, "required": list(self.required)},
        }

    def prepare(self, arguments: dict) -> tuple[dict, str | None]:
        """Clean the arguments. Returns (arguments, problem); problem is None when they are fine."""
        if not isinstance(arguments, dict):
            return {}, "arguments must be an object"
        missing = [name for name in self.required if name not in arguments]
        if missing:
            return arguments, f"missing required argument(s): {', '.join(missing)}"
        unknown = [name for name in arguments if name not in self.parameters]
        if unknown:
            return arguments, f"unknown argument(s): {', '.join(unknown)}. Allowed: {', '.join(self.parameters)}"
        clean = {}
        for name, value in arguments.items():
            expected = self.parameters[name].get("type")
            # Models sometimes send a number as text ("2021"). That is harmless, so accept it.
            if expected == "integer" and isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
                value = int(value)
            python_type = _JSON_TYPES.get(expected)
            if python_type and (isinstance(value, bool) != (expected == "boolean") or not isinstance(value, python_type)):
                return arguments, f"argument '{name}' must be of type {expected}"
            clean[name] = value
        return clean, None


def run_tool(tools: dict[str, Tool], name: str, arguments: dict) -> tuple[str, bool]:
    """Run one tool call. Returns (text for the model, is_error). Never raises.

    Every failure becomes a clear message, so the model can correct itself
    (other tool, other arguments) instead of the program crashing.
    """
    tool = tools.get(name)
    if tool is None:
        return f"Unknown tool '{name}'. Available tools: {', '.join(tools)}", True
    arguments, problem = tool.prepare(arguments)
    if problem:
        return f"Invalid arguments for {name}: {problem}", True
    try:
        result = str(tool.func(**arguments))
    except ToolError as error:
        return str(error), True
    except Exception as error:  # a bug or a network problem inside the tool
        return f"{name} failed: {type(error).__name__}: {error}", True
    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + "\n[result cut off]"
    return result, False
