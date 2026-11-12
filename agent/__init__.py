"""A tool-calling agent in plain Python."""
from .loop import DEFAULT_MODEL, MAX_STEPS, SYSTEM_PROMPT, Trace, run_agent

__all__ = ["DEFAULT_MODEL", "MAX_STEPS", "SYSTEM_PROMPT", "Trace", "run_agent"]
