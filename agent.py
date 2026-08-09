"""
A minimal tool-calling agent, built by hand (no LangChain/LangGraph yet).

WHY build it by hand first?
Frameworks like LangGraph do exactly what this file does, just with more
features and more abstraction layered on top. If you understand this loop,
LangGraph will make sense as "the same thing, with extra tooling." If you
skip straight to the framework, you'll be able to copy-paste working code
without being able to explain *why* it works in an interview.

THE CORE IDEA (this is the "ReAct" pattern - Reason, Act, Observe):
1. Send the user's question to the model, along with a list of tools it's
   allowed to use.
2. The model replies with either:
   a) a direct answer, or
   b) a request to call a specific tool with specific arguments
3. If (b): your code actually runs that tool, gets a real result, and
   sends that result BACK to the model as new information.
4. The model now has the tool's real output and can either answer, or
   request another tool call. Repeat until it gives a final answer.

This loop is literally what "agentic" means: the model isn't just
answering once, it's taking multiple steps and reacting to real results.
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic
from tools import AVAILABLE_TOOLS

load_dotenv()  # reads your .env file and loads ANTHROPIC_API_KEY

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-haiku-4-5-20251001"  # cheap + fast, plenty for this project

# This is how we describe each tool to the model: name, when to use it,
# and what input it expects. The model reads this and decides on its own
# whether a given question needs a tool, and if so, which one.
TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use this for questions "
            "about recent events, current prices, live data, or anything "
            "that could have changed after your training — not for general "
            "knowledge you already know."
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
            "Evaluate a math expression exactly. Use this for any arithmetic "
            "instead of computing it yourself — you are unreliable at math, "
            "this tool is not."
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
    """
    Runs the full reason -> act -> observe loop until the model gives a
    final text answer. Returns that final answer.
    """
    messages = [{"role": "user", "content": user_question}]

    while True:
        # Step 1: ask the model what it wants to do next.
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Step 2: check whether the model wants to call a tool, or is
        # giving a final answer.
        if response.stop_reason == "tool_use":
            # The model's reply gets added to the conversation history so
            # it remembers what it just decided to do.
            messages.append({"role": "assistant", "content": response.content})

            # A single reply can contain multiple tool calls. Handle each.
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    print(f"  [agent is calling tool: {tool_name}({tool_input})]")

                    # Step 3: actually run the real Python function.
                    tool_function = AVAILABLE_TOOLS[tool_name]
                    result = tool_function(**tool_input)

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            # Step 4: send the real tool result back to the model as a new
            # "user" turn, so it can use that real information to continue.
            messages.append({"role": "user", "content": tool_results})
            # Loop back to step 1 - the model now reasons again with the
            # new information available.

        else:
            # No more tools needed - this is the final answer.
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return final_text


if __name__ == "__main__":
    print("Tool-calling agent ready. Type a question (or 'quit' to exit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer = run_agent(question)
        print(f"\nAgent: {answer}\n")
