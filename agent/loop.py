"""The reason -> act -> observe loop (ReAct), written without a framework.

1. The question and the tool descriptions go to the model.
2. The model either answers, or asks for tools with arguments.
3. This code runs the real Python functions and sends the results back.
4. Repeat until the model answers or the step limit is reached.

The model never executes anything itself. It plans; this code runs.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

from .tools import ANALYST_TOOLS, Tool, run_tool

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-4-5-20251001")
MAX_STEPS = 8  # safety stop: the loop can never run forever
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a careful analyst with tools.

Rules:
- For any fact about a company's risk factors (shares, rankings, keyword counts, quotes), call the filing tools. Never answer these from memory.
- Use the calculator for every calculation, even an easy one.
- If a tool says the data does not cover a company or a year, say that clearly. Do not guess and do not switch to another source for it.
- General knowledge questions can be answered directly, without tools.
- Keep the final answer short and include the numbers you used."""


@dataclass
class ToolCall:
    step: int
    name: str
    arguments: dict
    result: str
    is_error: bool
    seconds: float


@dataclass
class Trace:
    """Everything that happened while answering one question."""

    question: str
    model: str = ""
    answer: str = ""
    stopped: str = "answered"  # answered | step_limit
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def make_client():
    """The Anthropic client. It reads ANTHROPIC_API_KEY from the environment."""
    from anthropic import Anthropic

    return Anthropic(max_retries=4)  # the SDK retries rate limits and server errors with backoff


def run_agent(
    question: str,
    client=None,
    tools: list[Tool] | None = None,
    system: str = SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    max_steps: int = MAX_STEPS,
    on_tool_call=None,
) -> Trace:
    """Answer one question. Returns the full trace, not only the text."""
    client = client or make_client()
    tools = ANALYST_TOOLS if tools is None else tools
    by_name = {tool.name: tool for tool in tools}
    request = {"model": model, "max_tokens": MAX_TOKENS, "system": system}
    if tools:
        request["tools"] = [tool.schema() for tool in tools]

    trace = Trace(question=question, model=model)
    messages = [{"role": "user", "content": question}]
    started = time.perf_counter()

    for step in range(1, max_steps + 1):
        response = client.messages.create(messages=messages, **request)
        trace.model_calls += 1
        trace.input_tokens += response.usage.input_tokens
        trace.output_tokens += response.usage.output_tokens

        if response.stop_reason != "tool_use":
            trace.answer = "".join(block.text for block in response.content if block.type == "text").strip()
            break

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:  # one reply can ask for several tools
            if block.type != "tool_use":
                continue
            tool_started = time.perf_counter()
            result, is_error = run_tool(by_name, block.name, block.input)
            call = ToolCall(step, block.name, dict(block.input or {}), result, is_error, round(time.perf_counter() - tool_started, 3))
            trace.tool_calls.append(call)
            if on_tool_call:
                on_tool_call(call)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": result, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    else:
        trace.stopped = "step_limit"
        trace.answer = f"Stopped: no final answer after {max_steps} steps."

    trace.seconds = round(time.perf_counter() - started, 2)
    return trace
