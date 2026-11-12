import re

import pandas as pd
import pytest

from agent.tools import ANALYST_TOOLS, Tool, ToolError, run_tool
from agent.tools.base import MAX_RESULT_CHARS
from agent.tools.calculator import calculator
from agent.tools.filings import DATA_DIR, count_keyword, get_risk_mix, list_companies, rank_companies, search_passages

TOOLS = {tool.name: tool for tool in ANALYST_TOOLS}


# ---- calculator
def test_basic_arithmetic():
    assert calculator("847 * 293") == "248171"
    assert calculator("(2 + 3) * 4") == "20"
    assert calculator("-5 + 2") == "-3"


def test_float_noise_is_hidden():
    assert calculator("0.1 + 0.2") == "0.3"
    assert calculator("10 / 4") == "2.5"
    assert calculator("10 / 5") == "2"


def test_division_by_zero_is_handled():
    with pytest.raises(ToolError, match="division by zero"):
        calculator("1 / 0")


def test_code_is_rejected():
    for bad in ["__import__('os').system('ls')", "open('x')", "a + 1", "'a' * 3", "1 if 1 else 2"]:
        with pytest.raises(ToolError):
            calculator(bad)


def test_huge_inputs_are_blocked():
    with pytest.raises(ToolError, match="too large"):
        calculator("9 ** 9 ** 9")
    with pytest.raises(ToolError, match="too long"):
        calculator("1+" * 200 + "1")


# ---- running a tool call safely
def test_every_tool_has_a_valid_schema():
    for tool in ANALYST_TOOLS:
        schema = tool.schema()
        assert schema["name"] and len(schema["description"]) > 30
        assert set(schema["input_schema"]["required"]) <= set(schema["input_schema"]["properties"])


def test_unknown_tool_is_an_error_not_a_crash():
    result, is_error = run_tool(TOOLS, "delete_everything", {})
    assert is_error and "Unknown tool" in result and "calculator" in result


def test_missing_unknown_and_wrong_type_arguments():
    assert "missing required" in run_tool(TOOLS, "get_risk_mix", {"company": "Apple"})[0]
    assert "unknown argument" in run_tool(TOOLS, "get_risk_mix", {"company": "Apple", "year": 2021, "x": 1})[0]
    assert "must be of type integer" in run_tool(TOOLS, "get_risk_mix", {"company": "Apple", "year": "last year"})[0]
    assert "must be of type string" in run_tool(TOOLS, "calculator", {"expression": 5})[0]


def test_a_number_sent_as_text_is_accepted():
    result, is_error = run_tool(TOOLS, "get_risk_mix", {"company": "Apple", "year": "2021"})
    assert not is_error and "fiscal year 2021" in result


def test_a_bug_inside_a_tool_becomes_a_message():
    broken = {"boom": Tool("boom", "always fails", lambda: 1 / 0)}
    result, is_error = run_tool(broken, "boom", {})
    assert is_error and "ZeroDivisionError" in result


def test_long_results_are_cut():
    noisy = {"noisy": Tool("noisy", "returns a lot", lambda: "x" * 50_000)}
    result, is_error = run_tool(noisy, "noisy", {})
    assert not is_error and len(result) < MAX_RESULT_CHARS + 50 and result.endswith("[result cut off]")


# ---- filing tools, checked against the raw data
def test_company_can_be_a_ticker_or_a_name():
    assert get_risk_mix("AAPL", 2021) == get_risk_mix("apple", 2021) == get_risk_mix("Apple", 2021)
    assert "JPMorgan Chase (JPM)" in get_risk_mix("jpmorgan", 2022)


def test_unknown_company_and_year_say_what_is_available():
    with pytest.raises(ToolError, match="only covers.*AAPL"):
        get_risk_mix("Netflix", 2022)
    with pytest.raises(ToolError, match="Available years: 2019"):
        get_risk_mix("Apple", 2012)


def test_risk_mix_matches_the_data_and_adds_up():
    text = get_risk_mix("TSLA", 2023)
    shares = [float(x) for x in re.findall(r": (\d+\.\d)% \(", text)]
    assert len(shares) == 13 and abs(sum(shares) - 100) < 0.5
    raw = pd.read_csv(DATA_DIR / "category_share.csv").query("ticker == 'TSLA' and year == 2023 and category == 'Supply Chain & Manufacturing'")
    assert f"Supply Chain & Manufacturing: {raw.share.iloc[0] * 100:.1f}%" in text


def test_ranking_puts_the_largest_share_first():
    raw = pd.read_csv(DATA_DIR / "category_share.csv").query("year == 2025 and category == 'Climate & ESG'")
    top = raw.sort_values("share", ascending=False).iloc[0]
    lines = rank_companies("climate", 2025).splitlines()
    assert lines[1].startswith(f"1. {top.company}") and len(lines) == 21
    with pytest.raises(ToolError, match="Unknown risk type"):
        rank_companies("weather", 2025)


def test_keyword_count_matches_a_direct_count():
    passages = pd.read_parquet(DATA_DIR / "passages.parquet").query("ticker == 'TSLA' and year == 2025")
    expected = int(passages.text.str.contains(r"\btariff", case=False, regex=True).sum())
    assert count_keyword("tariff", "Tesla", 2025).startswith(f"{expected} of {len(passages)} passages")


def test_search_returns_matching_passages():
    result = search_passages("ransomware attack", "MSFT", 2025, 2)
    assert result.count("- Microsoft 2025 [") == 2 and "ransomware" in result.lower()
    assert "No passage" in search_passages("zzzzqqq")


def test_list_companies_has_all_twenty():
    assert len(list_companies().splitlines()) == 21
