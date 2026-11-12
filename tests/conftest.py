"""A fake model for the tests: it replies with a script, so no API key and no cost."""
from types import SimpleNamespace

import pytest


def text(value):
    return SimpleNamespace(type="text", text=value)


def tool_use(name, arguments, call_id="call_1"):
    return SimpleNamespace(type="tool_use", name=name, input=arguments, id=call_id)


def reply(*blocks, tokens=(100, 20)):
    stop = "tool_use" if any(block.type == "tool_use" for block in blocks) else "end_turn"
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]))


class FakeClient:
    """Stands in for the Anthropic client. Records every request it gets."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = self

    def create(self, **request):
        # copy the messages: the loop keeps appending to the same list afterwards
        self.requests.append({**request, "messages": list(request["messages"])})
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return item(request) if callable(item) else item


@pytest.fixture
def fake():
    return FakeClient
