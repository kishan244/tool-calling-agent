from agent.loop import run_agent
from agent.tools import ANALYST_TOOLS

from .conftest import reply, text, tool_use


def test_direct_answer_needs_no_tool(fake):
    client = fake([reply(text("Paris."))])
    trace = run_agent("Capital of France?", client=client)
    assert trace.answer == "Paris." and trace.tool_calls == [] and trace.model_calls == 1
    assert {tool["name"] for tool in client.requests[0]["tools"]} == {tool.name for tool in ANALYST_TOOLS}


def test_tool_result_goes_back_to_the_model(fake):
    client = fake([reply(tool_use("calculator", {"expression": "847 * 293"}, "abc")), reply(text("It is 248,171."))])
    trace = run_agent("847 * 293?", client=client)
    assert trace.answer == "It is 248,171." and trace.model_calls == 2
    assert [(c.name, c.result, c.is_error) for c in trace.tool_calls] == [("calculator", "248171", False)]
    sent_back = client.requests[1]["messages"][-1]
    assert sent_back["role"] == "user"
    assert sent_back["content"] == [{"type": "tool_result", "tool_use_id": "abc", "content": "248171", "is_error": False}]


def test_several_tool_calls_in_one_reply(fake):
    both = reply(tool_use("get_risk_mix", {"company": "AAPL", "year": 2020}, "a"), tool_use("get_risk_mix", {"company": "AAPL", "year": 2021}, "b"))
    client = fake([both, reply(text("done"))])
    trace = run_agent("compare", client=client)
    assert len(trace.tool_calls) == 2 and all(call.step == 1 for call in trace.tool_calls)
    assert [r["tool_use_id"] for r in client.requests[1]["messages"][-1]["content"]] == ["a", "b"]


def test_errors_are_reported_to_the_model_and_the_loop_goes_on(fake):
    client = fake([
        reply(tool_use("no_such_tool", {}, "a")),
        reply(tool_use("get_risk_mix", {"company": "Netflix", "year": 2022}, "b")),
        reply(text("The data does not cover Netflix.")),
    ])
    trace = run_agent("Netflix?", client=client)
    assert [call.is_error for call in trace.tool_calls] == [True, True]
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True
    assert "only covers" in trace.tool_calls[1].result and trace.stopped == "answered"


def test_step_limit_stops_a_model_that_never_answers(fake):
    client = fake([reply(tool_use("calculator", {"expression": "1 + 1"}))])
    trace = run_agent("loop forever", client=client, max_steps=3)
    assert trace.stopped == "step_limit" and trace.model_calls == 3 and len(trace.tool_calls) == 3


def test_tokens_are_added_up(fake):
    client = fake([reply(tool_use("calculator", {"expression": "2 * 2"}), tokens=(500, 30)), reply(text("4"), tokens=(600, 5))])
    trace = run_agent("2*2", client=client)
    assert (trace.input_tokens, trace.output_tokens) == (1100, 35)


def test_no_tools_setup_sends_no_tool_list(fake):
    client = fake([reply(text("42"))])
    run_agent("anything", client=client, tools=[], system="plain")
    assert "tools" not in client.requests[0] and client.requests[0]["system"] == "plain"
