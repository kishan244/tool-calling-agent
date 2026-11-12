"""The tools the agent can call. Each one is a plain Python function plus a description."""
from .base import Tool, ToolError, run_tool
from .calculator import CALCULATOR
from .filings import FILING_TOOLS
from .web import WEB_SEARCH

ORIGINAL_TOOLS = [WEB_SEARCH, CALCULATOR]  # what the first version of this project had
ANALYST_TOOLS = [*FILING_TOOLS, CALCULATOR, WEB_SEARCH]

__all__ = ["ANALYST_TOOLS", "CALCULATOR", "FILING_TOOLS", "ORIGINAL_TOOLS", "WEB_SEARCH", "Tool", "ToolError", "run_tool"]
