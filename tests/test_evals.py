import json

from evals.build_questions import OUT, build
from evals.grade import final_line, grade, parse_number, wilson_interval
from evals.run import CONFIGS, run_config, summary_markdown

from .conftest import reply, text, tool_use
from .oracle import oracle

QUESTIONS = json.loads(OUT.read_text())
NUMBER = {"answer_type": "number", "answer": 13.0, "tolerance": 0.7}


def test_saved_questions_match_the_data():
    """If the data snapshot changes, the question file must be rebuilt, or the answers are stale."""
    assert build() == QUESTIONS and len(QUESTIONS) == 44
    assert len({q["id"] for q in QUESTIONS}) == 44


def test_final_line_and_numbers():
    assert final_line("Some work.\nANSWER: 12.5") == ("12.5", True)
    assert final_line("first ANSWER: 1\nthen\nANSWER: **2**") == ("2", True)
    assert final_line("no marker here\n7") == ("7", False)
    assert parse_number("1,234.5 passages") == 1234.5 and parse_number("−3.8") == -3.8 and parse_number("none") is None


def test_number_grading_uses_the_tolerance_and_the_sign():
    assert grade(NUMBER, "ANSWER: 13.4")["correct"]
    assert not grade(NUMBER, "ANSWER: 14.0")["correct"]
    assert not grade(NUMBER, "ANSWER: -13.0")["correct"]
    assert not grade(NUMBER, "ANSWER: NOT_AVAILABLE")["correct"]
    assert not grade(NUMBER, "")["correct"]


def test_choice_grading_rejects_naming_both_companies():
    question = {"answer_type": "choice", "answer": "Amazon", "wrong": "Alphabet"}
    assert grade(question, "ANSWER: Amazon")["correct"]
    assert grade(question, "ANSWER: amazon (AMZN)")["correct"]
    assert not grade(question, "ANSWER: Alphabet")["correct"]
    assert not grade(question, "ANSWER: Amazon or Alphabet")["correct"]


def test_not_available_grading_flags_made_up_answers():
    question = {"answer_type": "not_available", "answer": "NOT_AVAILABLE"}
    assert grade(question, "ANSWER: NOT_AVAILABLE") == {"correct": True, "given": "NOT_AVAILABLE", "format_ok": True, "made_up": False}
    assert grade(question, "ANSWER: 12.4")["made_up"]


def test_wilson_interval():
    low, high = wilson_interval(33, 44)
    assert 0.60 < low < 0.75 < high < 0.87
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_a_perfect_model_scores_100_percent(fake):
    """Every question can be answered with the tools, and loop -> tools -> grading works end to end."""
    result = run_config("analyst", QUESTIONS, fake([oracle]), model="scripted", log=lambda *_: None)
    wrong = [(row["id"], row["given"], row["answer"]) for row in result["questions"] if not row["correct"]]
    assert wrong == [] and result["summary"]["accuracy"] == 1.0
    assert result["summary"]["made_up_answers"] == [0, 5] and result["summary"]["used_expected_tool"] == 44


def test_a_model_without_tools_cannot_pass_by_refusing(fake):
    """Always answering NOT_AVAILABLE must only earn the 5 questions where that is true."""
    result = run_config("no_tools", QUESTIONS, fake([reply(text("ANSWER: NOT_AVAILABLE"))]), model="scripted", log=lambda *_: None)
    assert result["summary"]["correct"] == 5


def test_run_config_grades_and_summarises(fake):
    questions = [q for q in QUESTIONS if q["id"] in ("arithmetic-1", "not_covered-1")]

    def model(request):
        last = request["messages"][-1]["content"]
        if isinstance(last, str) and "847" in last:
            return reply(tool_use("calculator", {"expression": "847 * 293"}))
        if isinstance(last, str):
            return reply(text("I think it is about 12%.\nANSWER: 12"))  # a made-up answer
        return reply(text(f"ANSWER: {last[0]['content']}"))

    result = run_config("analyst", questions, fake([model]), model="fake-model", log=lambda *_: None)
    summary = result["summary"]
    assert (summary["correct"], summary["questions"]) == (1, 2)
    assert summary["made_up_answers"] == [1, 1] and summary["by_kind"] == {"arithmetic": [1, 1], "not_covered": [0, 1]}
    arithmetic = next(row for row in result["questions"] if row["id"] == "arithmetic-1")
    assert arithmetic["used_expected_tool"] and arithmetic["trace"]["tool_calls"][0]["result"] == "248171"
    assert "**50.0%**" in summary_markdown([result]) and set(CONFIGS) == {"no_tools", "original", "analyst"}


def test_an_api_error_does_not_lose_the_run(fake):
    def broken(request):
        raise RuntimeError("rate limited")

    result = run_config("no_tools", QUESTIONS[:2], fake([broken]), model="fake-model", log=lambda *_: None)
    assert result["summary"]["api_errors"] == 2 and result["summary"]["correct"] == 0
