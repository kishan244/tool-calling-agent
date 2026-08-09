"""
The agent's tools.

A "tool" is just a normal Python function. What makes it usable by the
agent is that we also describe it (name, purpose, expected inputs) in a
format the model understands — see TOOL_DEFINITIONS in agent.py.

The model never runs this code itself. It only ever says, in effect:
"I want to call calculator with expression='847*392'". Your Python code
is the one that actually runs it and hands the result back. This
separation is important — the model plans, your code executes.
"""

from ddgs import DDGS


def web_search(query: str) -> str:
    """
    Search the web and return a short summary of the top results.
    Used when the agent needs current/live information it can't know
    from training data alone (news, prices, recent events, etc).
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return "No search results found."

        formatted = []
        for r in results:
            formatted.append(f"- {r['title']}: {r['body']}")

        return "\n".join(formatted)

    except Exception as e:
        return f"Search failed: {e}"


def calculator(expression: str) -> str:
    """
    Safely evaluate a basic math expression.
    Used when the agent needs an exact numeric answer instead of
    guessing — language models are unreliable at arithmetic, so we
    always want them to delegate real math to actual code.
    """
    # Only allow digits, basic operators, parentheses, decimal points,
    # and spaces. This is a simple safety guard so the "calculator"
    # can't be tricked into running arbitrary code.
    allowed_chars = set("0123456789+-*/(). ")
    if not all(char in allowed_chars for char in expression):
        return "Error: expression contains characters that aren't allowed."

    try:
        result = eval(expression)  # safe here because of the character check above
        return str(result)
    except Exception as e:
        return f"Calculation failed: {e}"


# This is the "menu" the agent sees: a mapping from tool name to the
# actual Python function, so agent.py can call the right one by name.
AVAILABLE_TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
}
