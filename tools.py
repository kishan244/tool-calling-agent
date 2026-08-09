"""Tools the agent can call. Each one is a plain Python function."""

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


def calculator(expression: str) -> str:
    """Evaluate a basic math expression."""
    # Character allowlist keeps eval from running anything but arithmetic.
    allowed_chars = set("0123456789+-*/(). ")
    if not all(char in allowed_chars for char in expression):
        return "Error: expression contains characters that aren't allowed."

    try:
        return str(eval(expression))
    except Exception as e:
        return f"Calculation failed: {e}"


AVAILABLE_TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
}
