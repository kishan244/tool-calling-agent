"""Build the evaluation set: questions whose correct answer is computed from the data.

Run:  python -m evals.build_questions

The answers are worked out here with pandas, straight from the data files, and not through
the agent's tools. So a bug in a tool shows up as a wrong answer instead of hiding.
The random seed is fixed, so the same questions come out every time.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "questions.json"
SEED = 7
YEARS = list(range(2019, 2026))  # years in which all 20 companies have a report
KEYWORDS = ["tariff", "inflation", "pandemic", "cyber", "climate", "artificial intelligence", "shortage", "sanctions"]
NOT_COVERED = ["Netflix", "Intel", "Ford", "Bank of America", "Starbucks"]


def build() -> list[dict]:
    rng = random.Random(SEED)
    shares = pd.read_csv(ROOT / "data" / "category_share.csv")
    shares = shares[shares.year.isin(YEARS)]
    passages = pd.read_parquet(ROOT / "data" / "passages.parquet")
    companies = shares[["ticker", "company"]].drop_duplicates().to_records(index=False).tolist()
    categories = sorted(shares.category.unique())
    pct = {(r.ticker, r.year, r.category): r.share * 100 for r in shares.itertuples()}
    questions: list[dict] = []

    def add(kind, question, answer_type, answer, tools, **extra):
        questions.append({"id": f"{kind}-{sum(q['kind'] == kind for q in questions) + 1}", "kind": kind, "question": question,
                          "answer_type": answer_type, "answer": answer, "expected_tools": tools, **extra})

    # 1. lookup: one tool call, read one number
    while sum(q["kind"] == "lookup" for q in questions) < 8:
        (ticker, company), year, category = rng.choice(companies), rng.choice(YEARS), rng.choice(categories)
        value = pct[ticker, year, category]
        if value >= 3:
            add("lookup", f"What share of {company}'s fiscal year {year} Risk Factors section is about '{category}'? Answer in percent.",
                "number", round(value, 1), ["get_risk_mix"], tolerance=0.6)

    # 2. compare: two companies, same year (only clear gaps, so rounding cannot flip the answer)
    while sum(q["kind"] == "compare" for q in questions) < 8:
        (t1, c1), (t2, c2) = rng.sample(companies, 2)
        year, category = rng.choice(YEARS), rng.choice(categories)
        a, b = pct[t1, year, category], pct[t2, year, category]
        if abs(a - b) >= 3:
            add("compare", f"In fiscal year {year}, which company gave a larger share of its Risk Factors section to '{category}': {c1} or {c2}?",
                "choice", c1 if a > b else c2, ["get_risk_mix", "rank_companies"], wrong=c2 if a > b else c1)

    # 3. change: two tool calls and a subtraction
    while sum(q["kind"] == "change" for q in questions) < 8:
        (ticker, company), category = rng.choice(companies), rng.choice(categories)
        y1, y2 = sorted(rng.sample(YEARS, 2))
        diff = pct[ticker, y2, category] - pct[ticker, y1, category]
        if abs(diff) >= 3:
            add("change", f"By how many percentage points did the '{category}' share of {company}'s Risk Factors section change from fiscal year {y1} to {y2}? "
                "Give a signed number (negative if it fell).", "number", round(diff, 1), ["get_risk_mix", "calculator"], tolerance=0.7)

    # 4. rank: one company out of twenty
    while sum(q["kind"] == "rank" for q in questions) < 6:
        year, category = rng.choice(YEARS), rng.choice(categories)
        table = shares[(shares.year == year) & (shares.category == category)].sort_values("share", ascending=False)
        top, second = table.iloc[0], table.iloc[1]
        if (top.share - second.share) * 100 >= 1.5:
            add("rank", f"Which of the companies in the data gave the largest share of its fiscal year {year} Risk Factors section to '{category}'?",
                "choice", top.company, ["rank_companies"], wrong="")

    # 5. keyword: exact count
    while sum(q["kind"] == "keyword" for q in questions) < 6:
        (ticker, company), year, keyword = rng.choice(companies), rng.choice(YEARS), rng.choice(KEYWORDS)
        texts = passages[(passages.ticker == ticker) & (passages.year == year)].text
        hits = int(texts.str.contains(re.compile(r"\b" + re.escape(keyword), re.I)).sum())
        if hits >= 2:
            add("keyword", f"How many passages of {company}'s fiscal year {year} Risk Factors section mention '{keyword}'?",
                "number", hits, ["count_keyword"], tolerance=0)

    # 6. not covered: the honest answer is "the data does not have this"
    for company in NOT_COVERED[:3]:
        year, category = rng.choice(YEARS), rng.choice(categories)
        add("not_covered", f"What share of {company}'s fiscal year {year} Risk Factors section is about '{category}'? Answer in percent.",
            "not_available", "NOT_AVAILABLE", ["get_risk_mix", "list_companies"])
    for year in (2012, 2016):
        (ticker, company), category = rng.choice(companies), rng.choice(categories)
        add("not_covered", f"What share of {company}'s fiscal year {year} Risk Factors section is about '{category}'? Answer in percent.",
            "not_available", "NOT_AVAILABLE", ["get_risk_mix", "list_companies"])

    # 7. arithmetic: the calculator, with nothing to look up
    for expression, value in [("847 * 293", 248171), ("(1234 + 5678) * 91", 628992), ("987654 - 123456 * 7", 123462)]:
        add("arithmetic", f"What is {expression}?", "number", value, ["calculator"], tolerance=0)

    return questions


if __name__ == "__main__":
    questions = build()
    OUT.write_text(json.dumps(questions, indent=1, ensure_ascii=False) + "\n")
    kinds = pd.Series([q["kind"] for q in questions]).value_counts().to_dict()
    print(f"{len(questions)} questions -> {OUT.relative_to(ROOT)}  {kinds}")
