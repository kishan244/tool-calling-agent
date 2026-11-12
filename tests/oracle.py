"""A scripted "perfect model" for tests: it solves each kind of question with the right tool calls.

It proves two things without any API cost: every question can be answered with the tools,
and the whole chain (loop -> tools -> grading -> summary) gives 100% when the model does its part.
"""
import re

from .conftest import reply, text, tool_use


def _share(result: str, category: str) -> float:
    return float(re.search(re.escape(f"- {category}: ") + r"([\d.]+)%", result).group(1))


def oracle(request):
    messages = request["messages"]
    question = messages[0]["content"]
    results = [block["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], list) for block in m["content"]]
    errors = [block["is_error"] for m in messages if m["role"] == "user" and isinstance(m["content"], list) for block in m["content"]]
    quoted = re.search(r"(?:did the|is about|section to|mention) '([^']+)'", question)
    category = quoted.group(1) if quoted else ""
    years = [int(y) for y in re.findall(r"fiscal year (\d{4})(?: to (\d{4}))?", question)[0] if y] if "fiscal year" in question else []

    if question.startswith("What is "):
        return reply(text(f"ANSWER: {results[0]}")) if results else reply(tool_use("calculator", {"expression": question[8:-1]}))
    if question.startswith("How many passages"):
        if results:
            return reply(text(f"ANSWER: {results[0].split()[0]}"))
        company = question.split("passages of ")[1].split("'s fiscal")[0]
        return reply(tool_use("count_keyword", {"keyword": category, "company": company, "year": years[0]}))
    if question.startswith("Which of the companies"):
        if results:
            return reply(text("ANSWER: " + results[0].splitlines()[1].split(". ", 1)[1].split(" (")[0]))
        return reply(tool_use("rank_companies", {"risk_type": category, "year": years[0]}))
    if question.startswith("In fiscal year"):
        first, second = question.split("': ")[1].rstrip("?").split(" or ")
        if results:
            return reply(text(f"ANSWER: {first if _share(results[0], category) > _share(results[1], category) else second}"))
        return reply(tool_use("get_risk_mix", {"company": first, "year": years[0]}, "a"), tool_use("get_risk_mix", {"company": second, "year": years[0]}, "b"))
    if question.startswith("By how many"):
        company = question.split("share of ")[1].split("'s Risk")[0]
        if len(results) == 3:
            return reply(text(f"ANSWER: {results[2]}"))
        if len(results) == 2:
            return reply(tool_use("calculator", {"expression": f"{_share(results[1], category)} - {_share(results[0], category)}"}))
        return reply(tool_use("get_risk_mix", {"company": company, "year": years[0]}, "a"), tool_use("get_risk_mix", {"company": company, "year": years[1]}, "b"))
    # "What share of X's fiscal year Y ... is about 'Z'?"
    company = question.split("share of ")[1].split("'s fiscal")[0]
    if results:
        return reply(text("ANSWER: NOT_AVAILABLE" if errors[0] else f"ANSWER: {_share(results[0], category)}"))
    return reply(tool_use("get_risk_mix", {"company": company, "year": years[0]}))
