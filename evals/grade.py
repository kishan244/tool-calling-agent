"""Grade one answer against the expected answer. No model is involved: the rules are fixed."""
from __future__ import annotations

import math
import re

ANSWER_FORMAT = (
    "\n\nFinish with one last line in exactly this form: `ANSWER: <value>`. "
    "<value> is a plain number (no units, no % sign), or a company name, "
    "or NOT_AVAILABLE if you cannot find the answer out reliably."
)

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_line(answer: str) -> tuple[str, bool]:
    """The text after the last 'ANSWER:'. Second value: did the model follow the format?"""
    matches = re.findall(r"ANSWER:\s*(.+)", answer or "")
    if matches:
        return matches[-1].strip().strip("`*_ "), True
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    return (lines[-1] if lines else ""), False


def parse_number(text: str) -> float | None:
    match = _NUMBER.search(text.replace("−", "-").replace("–", "-"))
    return float(match.group().replace(",", "")) if match else None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def says_not_available(text: str) -> bool:
    return "notavailable" in _norm(text)


def grade(question: dict, answer: str) -> dict:
    """Returns {"correct": bool, "given": str, "format_ok": bool, "made_up": bool}."""
    given, format_ok = final_line(answer)
    kind = question["answer_type"]
    correct = False
    if kind == "not_available":
        correct = says_not_available(given)
    elif says_not_available(given):
        correct = False
    elif kind == "number":
        value = parse_number(given)
        correct = value is not None and math.isclose(value, question["answer"], abs_tol=question.get("tolerance", 0) + 1e-9)
    elif kind == "choice":
        wrong = question.get("wrong", "")
        correct = _norm(question["answer"]) in _norm(given) and not (wrong and _norm(wrong) in _norm(given))
    # "made up": the data cannot answer the question, but the model gave an answer anyway
    made_up = question["answer_type"] == "not_available" and not correct
    return {"correct": correct, "given": given, "format_ok": format_ok, "made_up": made_up}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% range for a success rate measured on n questions."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)
