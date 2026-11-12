"""Web search, for questions the local data cannot answer."""
from __future__ import annotations

from .base import Tool, ToolError


def web_search(query: str) -> str:
    """Return a short summary of the top search results."""
    from ddgs import DDGS  # imported here so the rest of the agent works without it

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as error:
        raise ToolError(f"Search failed: {error}") from None
    if not results:
        return "No search results found."
    return "\n".join(f"- {r['title']}: {r['body']}" for r in results)


WEB_SEARCH = Tool(
    name="web_search",
    description=(
        "Search the web for current information (news, prices, recent events). "
        "Do NOT use it for the companies' risk-factor data: the filing tools are exact, the web is not."
    ),
    func=web_search,
    parameters={"query": {"type": "string", "description": "The search query to run."}},
    required=("query",),
)
