"""Tools over a snapshot of annual-report (10-K) risk factors: 20 US companies, 2019 onwards.

The data comes from the Risk Radar project (github.com/kishan244/risk-narrative-nlp):
every passage of each company's "Risk Factors" section, sorted into 13 risk types.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .base import Tool, ToolError

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@lru_cache(maxsize=1)
def _shares() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "category_share.csv")


@lru_cache(maxsize=1)
def _reports() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "reports.csv")


@lru_cache(maxsize=1)
def _passages() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "passages.parquet")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _company(name: str) -> tuple[str, str]:
    """Accept a ticker or a company name. Returns (ticker, company)."""
    table = _reports()[["ticker", "company"]].drop_duplicates()
    key = _norm(name)
    for row in table.itertuples():
        if key in (_norm(row.ticker), _norm(row.company)):
            return row.ticker, row.company
    partial = [row for row in table.itertuples() if key and len(key) >= 4 and key in _norm(row.company)]
    if len(partial) == 1:
        return partial[0].ticker, partial[0].company
    known = ", ".join(f"{row.ticker} ({row.company})" for row in table.itertuples())
    raise ToolError(f"No data for company '{name}'. The data only covers: {known}.")


def _check_year(ticker: str, company: str, year: int) -> None:
    years = sorted(_reports().loc[_reports().ticker == ticker, "year"])
    if year not in years:
        raise ToolError(f"No report for {company} for fiscal year {year}. Available years: {years[0]} to {years[-1]}.")


def _category(name: str) -> str:
    categories = sorted(_shares().category.unique())
    key = _norm(name)
    exact = [c for c in categories if _norm(c) == key]
    if exact:
        return exact[0]
    partial = [c for c in categories if key and (key in _norm(c) or any(_norm(part) == key for part in c.split("&")))]
    if len(partial) == 1:
        return partial[0]
    raise ToolError(f"Unknown risk type '{name}'. Use one of: {'; '.join(categories)}.")


def keyword_pattern(keyword: str) -> re.Pattern:
    """Case-insensitive match for words that start with the keyword ('tariff' also finds 'tariffs')."""
    return re.compile(r"\b" + re.escape(keyword.strip()), re.I)


# --------------------------------------------------------------------------- tools


def list_companies() -> str:
    reports = _reports()
    lines = ["ticker | company | sector | fiscal years with a report"]
    for (ticker, company, sector), group in reports.groupby(["ticker", "company", "sector"], sort=False):
        lines.append(f"{ticker} | {company} | {sector} | {group.year.min()}-{group.year.max()}")
    return "\n".join(lines)


def get_risk_mix(company: str, year: int) -> str:
    ticker, name = _company(company)
    _check_year(ticker, name, year)
    report = _reports().query("ticker == @ticker and year == @year").iloc[0]
    rows = _shares().query("ticker == @ticker and year == @year").sort_values("share", ascending=False)
    lines = [
        f"{name} ({ticker}), fiscal year {year}. Risk Factors section: {int(report.words):,} words, "
        f"{int(report.passages)} passages.",
        "Share of the section per risk type:",
    ]
    lines += [f"- {row.category}: {row.share * 100:.1f}% ({int(row.passages)} passages)" for row in rows.itertuples()]
    if not math.isnan(report.reused_share):
        lines.append(f"Sentences repeated word for word from the previous year: {report.reused_share * 100:.1f}%")
    return "\n".join(lines)


def rank_companies(risk_type: str, year: int) -> str:
    category = _category(risk_type)
    rows = _shares().query("category == @category and year == @year").sort_values("share", ascending=False)
    if rows.empty:
        years = sorted(_shares().year.unique())
        raise ToolError(f"No reports for fiscal year {year}. Available years: {years[0]} to {years[-1]}.")
    lines = [f"Share of the Risk Factors section about '{category}', fiscal year {year}, highest first:"]
    lines += [f"{i}. {row.company} ({row.ticker}, {row.sector}): {row.share * 100:.1f}%" for i, row in enumerate(rows.itertuples(), 1)]
    return "\n".join(lines)


def count_keyword(keyword: str, company: str, year: int) -> str:
    if not keyword.strip():
        raise ToolError("keyword must not be empty")
    ticker, name = _company(company)
    _check_year(ticker, name, year)
    texts = _passages().query("ticker == @ticker and year == @year").text
    hits = int(texts.str.contains(keyword_pattern(keyword)).sum())
    return (
        f"{hits} of {len(texts)} passages ({hits / len(texts) * 100:.1f}%) in {name}'s fiscal year {year} "
        f"Risk Factors section mention '{keyword.strip()}'."
    )


def search_passages(query: str, company: str = "", year: int = 0, limit: int = 3) -> str:
    """Rank passages by how many of the query's words they contain; rare words count more."""
    passages = _passages()
    scope = "all companies"
    if company:
        ticker, name = _company(company)
        passages, scope = passages[passages.ticker == ticker], name
    if year:
        passages = passages[passages.year == year]
        scope += f", fiscal year {year}"
    if passages.empty:
        raise ToolError(f"No passages for {scope}.")
    words = {w for w in re.findall(r"[a-z0-9]{2,}", query.lower())}
    if not words:
        raise ToolError("query needs at least one word")
    lowered = passages.text.str.lower()
    score = pd.Series(0.0, index=passages.index)
    for word in words:
        contains = lowered.str.contains(r"\b" + re.escape(word), regex=True)
        if contains.any():
            score += contains * math.log(len(passages) / contains.sum() + 1)
    best = score[score > 0].sort_values(ascending=False).head(max(1, min(int(limit), 5)))
    if best.empty:
        return f"No passage in {scope} contains any of the words in '{query}'."
    lines = [f"Top {len(best)} passages for '{query}' in {scope}:"]
    for index in best.index:
        row = passages.loc[index]
        snippet = re.sub(r"\s+", " ", row.text)[:450]
        lines.append(f"- {row.company} {row.year} [{row.category}]: {snippet}...")
    return "\n".join(lines)


_COMPANY = {"type": "string", "description": "Ticker or company name, e.g. 'AAPL' or 'Apple'."}
_YEAR = {"type": "integer", "description": "Fiscal year, e.g. 2023."}

FILING_TOOLS = [
    Tool(
        name="list_companies",
        description="List the companies, sectors and fiscal years the risk-factor data covers. Call it when unsure whether a company or year is covered.",
        func=list_companies,
    ),
    Tool(
        name="get_risk_mix",
        description=(
            "For ONE company and ONE fiscal year: how its annual-report Risk Factors section splits across the 13 risk types "
            "(share in percent and number of passages), plus section length and how much text was repeated from the year before. "
            "To compare two years or two companies, call it once for each."
        ),
        func=get_risk_mix,
        parameters={"company": _COMPANY, "year": _YEAR},
        required=("company", "year"),
    ),
    Tool(
        name="rank_companies",
        description="For ONE risk type and ONE fiscal year: all companies ranked by the share of their Risk Factors section about that risk type. Use it for 'which company has the highest/lowest ...' questions.",
        func=rank_companies,
        parameters={
            "risk_type": {"type": "string", "description": "One of the 13 risk types, e.g. 'Climate & ESG' or 'Regulatory & Legal'."},
            "year": _YEAR,
        },
        required=("risk_type", "year"),
    ),
    Tool(
        name="count_keyword",
        description="Count how many passages of ONE company's Risk Factors section in ONE fiscal year mention a keyword (case-insensitive; 'tariff' also matches 'tariffs').",
        func=count_keyword,
        parameters={"keyword": {"type": "string", "description": "A word or short phrase, e.g. 'tariff'."}, "company": _COMPANY, "year": _YEAR},
        required=("keyword", "company", "year"),
    ),
    Tool(
        name="search_passages",
        description="Find the passages of the Risk Factors sections that best match some words, to quote what a company actually wrote. Optional filters: company, year.",
        func=search_passages,
        parameters={
            "query": {"type": "string", "description": "Words to look for, e.g. 'generative AI competition'."},
            "company": _COMPANY,
            "year": _YEAR,
            "limit": {"type": "integer", "description": "How many passages to return (1-5, default 3)."},
        },
        required=("query",),
    ),
]
