"""Tool-calling agent using the ReAct loop, written without a framework."""

import os
from dotenv import load_dotenv
from anthropic import Anthropic
from tools import AVAILABLE_TOOLS

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-haiku-4-5-20251001"
MAX_STEPS = 8  # safety stop: the loop can never run forever

TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use for recent events, "
            "prices, live data, or anything that may have changed after training."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a math expression exactly. Use for any arithmetic "
            "instead of computing it yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '847 * 392'.",
                }
            },
            "required": ["expression"],
        },
    },
]


def run_agent(user_question: str) -> str:
    """Run the reason/act/observe loop until the model returns a text answer."""
    messages = [{"role": "user", "content": user_question}]

    for _ in range(MAX_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        messages.append({"role": "assistant", "content": response.content})

        # One reply can request several tool calls.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [calling {block.name}({block.input})]")
                tool = AVAILABLE_TOOLS.get(block.name)
                if tool is None:
                    result, failed = f"Unknown tool: {block.name}", True
                else:
                    result = tool(**block.input)
                    failed = result.startswith(("Search failed", "Calculation failed"))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                        "is_error": failed,
                    }
                )

        messages.append({"role": "user", "content": tool_results})

    return "Stopped: the agent needed more than MAX_STEPS tool calls."


if __name__ == "__main__":
    print("Tool-calling agent ready. Type a question (or 'quit' to exit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        print(f"\nAgent: {run_agent(question)}\n")
