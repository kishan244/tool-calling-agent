"""Run the evaluation set against one or more agent setups and write the results.

    python -m evals.run                       # all setups
    python -m evals.run --config analyst      # one setup
    python -m evals.run --limit 5 --tag smoke # a cheap trial run

Setups (same questions, same model, same grading):
    no_tools  the model alone
    original  the first version of this project: web search + calculator
    analyst   the current agent: filing tools + calculator + web search
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agent.loop import DEFAULT_MODEL, MAX_STEPS, SYSTEM_PROMPT, Trace, make_client, run_agent
from agent.tools import ANALYST_TOOLS, ORIGINAL_TOOLS

from .grade import ANSWER_FORMAT, grade, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals" / "questions.json"
RESULTS = ROOT / "evals" / "results"

CONFIGS = {
    "no_tools": {"label": "Model alone", "tools": [], "system": "You are a careful analyst."},
    "original": {"label": "First version (web search + calculator)", "tools": ORIGINAL_TOOLS, "system": "You are a careful analyst with tools."},
    "analyst": {"label": "Analyst agent (filing tools)", "tools": ANALYST_TOOLS, "system": SYSTEM_PROMPT},
}


def run_config(name: str, questions: list[dict], client, model: str, max_steps: int = MAX_STEPS, log=print) -> dict:
    config = CONFIGS[name]
    rows = []
    for number, question in enumerate(questions, 1):
        try:
            trace = run_agent(question["question"], client=client, tools=config["tools"],
                              system=config["system"] + ANSWER_FORMAT, model=model, max_steps=max_steps)
            error = ""
        except Exception as exception:  # an API problem must not lose the whole run
            trace, error = Trace(question=question["question"], model=model, stopped="error"), f"{type(exception).__name__}: {exception}"
        result = grade(question, trace.answer)
        called = [call.name for call in trace.tool_calls]
        rows.append({
            **{key: question[key] for key in ("id", "kind", "question", "answer")},
            **result,
            "error": error,
            "used_expected_tool": any(tool in called for tool in question["expected_tools"]),
            "tool_errors": sum(call.is_error for call in trace.tool_calls),
            "trace": trace.to_dict(),
        })
        log(f"[{name}] {number}/{len(questions)} {question['id']}: {'ok ' if result['correct'] else 'WRONG'} given={result['given'][:40]!r} expected={question['answer']!r}")
    return {"config": name, "label": config["label"], "model": model, "tools": [tool.name for tool in config["tools"]],
            "summary": summarise(rows), "questions": rows}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(row["correct"] for row in rows)
    low, high = wilson_interval(correct, n)
    kinds = sorted({row["kind"] for row in rows})
    not_covered = [row for row in rows if row["kind"] == "not_covered"]
    return {
        "questions": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0,
        "accuracy_ci95": [round(low, 3), round(high, 3)],
        "by_kind": {kind: [sum(r["correct"] for r in rows if r["kind"] == kind), sum(r["kind"] == kind for r in rows)] for kind in kinds},
        "made_up_answers": [sum(row["made_up"] for row in not_covered), len(not_covered)],
        "used_expected_tool": sum(row["used_expected_tool"] for row in rows),
        "format_ok": sum(row["format_ok"] for row in rows),
        "step_limit_stops": sum(row["trace"]["stopped"] == "step_limit" for row in rows),
        "api_errors": sum(bool(row["error"]) for row in rows),
        "avg_model_calls": round(sum(row["trace"]["model_calls"] for row in rows) / n, 2) if n else 0,
        "avg_tool_calls": round(sum(len(row["trace"]["tool_calls"]) for row in rows) / n, 2) if n else 0,
        "tool_errors": sum(row["tool_errors"] for row in rows),
        "input_tokens": sum(row["trace"]["input_tokens"] for row in rows),
        "output_tokens": sum(row["trace"]["output_tokens"] for row in rows),
        "avg_seconds": round(sum(row["trace"]["seconds"] for row in rows) / n, 2) if n else 0,
    }


def summary_markdown(results: list[dict]) -> str:
    kinds = list(results[0]["summary"]["by_kind"])
    head = ["Setup", "Correct", "Accuracy (95% range)", *kinds, "Made-up answers", "Avg model calls", "Tokens in / out"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for result in results:
        s = result["summary"]
        low, high = s["accuracy_ci95"]
        cells = [result["label"], f"{s['correct']}/{s['questions']}", f"**{s['accuracy'] * 100:.1f}%** ({low * 100:.0f}–{high * 100:.0f}%)",
                 *[f"{s['by_kind'][k][0]}/{s['by_kind'][k][1]}" for k in kinds],
                 f"{s['made_up_answers'][0]}/{s['made_up_answers'][1]}", f"{s['avg_model_calls']}", f"{s['input_tokens']:,} / {s['output_tokens']:,}"]
        lines.append("| " + " | ".join(cells) + " |")
    return f"Model: `{results[0]['model']}` · {results[0]['summary']['questions']} questions\n\n" + "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=[*CONFIGS, "all"], default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions of each kind (0 = all)")
    parser.add_argument("--tag", default="latest", help="folder name under evals/results/")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    questions = json.loads(QUESTIONS.read_text())
    if args.limit:
        seen: dict[str, int] = {}
        questions = [q for q in questions if (seen.__setitem__(q["kind"], seen.get(q["kind"], 0) + 1) or seen[q["kind"]] <= args.limit)]
    client = make_client()
    out = RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.time()
    for name in (CONFIGS if args.config == "all" else [args.config]):
        result = run_config(name, questions, client, args.model)
        (out / f"{name}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
        results.append(result)
    markdown = summary_markdown(results)
    (out / "summary.md").write_text(markdown)
    print(f"\n{markdown}\nDone in {time.time() - started:.0f}s -> {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
