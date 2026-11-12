"""Web app: see how the agent did on the evaluation set, step by step, and ask it a question."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent.loop import DEFAULT_MODEL, run_agent
from agent.tools import ANALYST_TOOLS

RESULTS = Path(os.environ.get("AGENT_RESULTS_DIR") or Path(__file__).parent / "evals" / "results")
BLUE, GREY, INK, MUTED = "#2a78d6", "#b8bcc4", "#1f2328", "#667085"
KIND_LABELS = {
    "lookup": "Look up one number",
    "compare": "Compare two companies",
    "change": "Change between two years",
    "rank": "Find the top company",
    "keyword": "Count a keyword",
    "not_covered": "Not in the data (must say so)",
    "arithmetic": "Arithmetic",
}
SESSION_LIMIT, DAILY_LIMIT = 5, 40  # live questions: per visitor, and for everybody together per day

st.set_page_config(page_title="Analyst Agent", page_icon="🧭", layout="wide")


@st.cache_data
def load_runs(results_dir: str) -> dict[str, list[dict]]:
    """{run name: [result per setup]} for every folder under evals/results/."""
    runs = {}
    for folder in sorted(path for path in Path(results_dir).glob("*") if path.is_dir()):
        results = [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))]
        if results:
            order = {"no_tools": 0, "original": 1, "analyst": 2}
            runs[folder.name] = sorted(results, key=lambda r: order.get(r["config"], 9))
    return runs


@st.cache_resource
def daily_counter() -> dict:
    return {"day": None, "count": 0}


def api_key() -> str:
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    except Exception:  # no secrets file
        return os.environ.get("ANTHROPIC_API_KEY", "")


def show_steps(tool_calls: list[dict], answer: str) -> None:
    if not tool_calls:
        st.caption("The model answered directly, without a tool.")
    for call in tool_calls:
        arguments = ", ".join(f"{key}={value!r}" for key, value in call["arguments"].items())
        icon = "⚠️" if call["is_error"] else "🔧"
        with st.expander(f"{icon} Step {call['step']} · {call['name']}({arguments})", expanded=len(tool_calls) <= 3):
            st.code(call["result"], language=None, wrap_lines=True)
    st.markdown("**Final answer**")
    st.info(answer or "(empty)")


st.title("Analyst Agent")
st.markdown(
    "A tool-calling AI agent written in plain Python, **without a framework**. It answers questions about the risk factors "
    "in the annual reports of 20 US companies by calling tools on real data. Because the correct answers can be computed "
    "from the data, every answer of the agent is **checked automatically**."
)

tab_results, tab_inside, tab_live, tab_how = st.tabs(["Results", "Look inside a run", "Ask the agent", "How it works"])
runs = load_runs(str(RESULTS))

with tab_results:
    if not runs:
        st.warning("No evaluation results in this copy yet. Run `python -m evals.run` (needs an Anthropic API key).")
    else:
        run_name = st.selectbox("Evaluation run", list(runs), index=len(runs) - 1) if len(runs) > 1 else next(iter(runs))
        results = runs[run_name]
        best = results[-1]["summary"]
        low, high = best["accuracy_ci95"]
        a, b, c, d = st.columns(4)
        a.metric(f"Correct answers · {results[-1]['label']}", f"{best['accuracy'] * 100:.1f}%", f"{best['correct']} of {best['questions']} · 95% range {low * 100:.0f}–{high * 100:.0f}%", delta_color="off", delta_arrow="off")
        b.metric("Made-up answers", f"{best['made_up_answers'][0]} of {best['made_up_answers'][1]}", "questions the data cannot answer", delta_color="off", delta_arrow="off")
        c.metric("Model calls per question", f"{best['avg_model_calls']}", f"{best['avg_seconds']} s on average", delta_color="off", delta_arrow="off")
        d.metric("Tokens per question", f"{(best['input_tokens'] + best['output_tokens']) / best['questions']:,.0f}", f"model: {results[-1]['model']}", delta_color="off", delta_arrow="off")

        left, right = st.columns([2, 3])
        with left:
            st.markdown("**Same questions, three setups**")
            figure = go.Figure(go.Bar(
                x=[r["summary"]["accuracy"] * 100 for r in results], y=[r["label"] for r in results], orientation="h",
                marker_color=[GREY] * (len(results) - 1) + [BLUE], marker_cornerradius=4,
                text=[f"{r['summary']['accuracy'] * 100:.0f}%" for r in results], textposition="outside", textfont_color=INK,
                hovertemplate="%{y}: %{x:.1f}% correct<extra></extra>",
            ))
            figure.update_layout(height=230, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[0, 125], tickvals=[0, 25, 50, 75, 100], ticksuffix="%", gridcolor="#eceef1"),
                                 yaxis=dict(autorange="reversed"), plot_bgcolor="white", font_color=MUTED)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        with right:
            st.markdown("**Correct answers per kind of question**")
            kinds = list(best["by_kind"])
            table = pd.DataFrame({r["label"]: [f"{r['summary']['by_kind'][k][0]} / {r['summary']['by_kind'][k][1]}" for k in kinds] for r in results},
                                 index=[KIND_LABELS.get(k, k) for k in kinds])
            st.dataframe(table, width="stretch")
        st.caption("The answers are graded by fixed rules (number within a small tolerance, right company name, or NOT_AVAILABLE), not by another model.")

with tab_inside:
    if not runs:
        st.info("Nothing to show yet.")
    else:
        results = runs[list(runs)[-1]]
        setup = st.radio("Setup", [r["label"] for r in results], index=len(results) - 1, horizontal=True)
        rows = next(r for r in results if r["label"] == setup)["questions"]
        only_wrong = st.toggle("Only wrong answers", value=False)
        shown = [row for row in rows if not (only_wrong and row["correct"])]
        if not shown:
            st.success("No wrong answers in this setup.")
        else:
            row = st.selectbox("Question", shown, format_func=lambda r: f"{'✅' if r['correct'] else '❌'} {r['question']}")
            x, y, z = st.columns(3)
            x.metric("Expected", str(row["answer"]))
            y.metric("Agent said", row["given"][:28] or "(nothing)")
            z.metric("Result", "Correct" if row["correct"] else "Wrong")
            if row["error"]:
                st.error(row["error"])
            show_steps(row["trace"]["tool_calls"], row["trace"]["answer"])

with tab_live:
    st.markdown("Ask about the risk factors of the 20 companies, for example: *Which company wrote most about climate risk in 2025, and how much more than the runner-up?*")
    counter = daily_counter()
    today = datetime.date.today()
    if counter["day"] != today:
        counter.update(day=today, count=0)
    asked = st.session_state.get("asked", 0)
    if not api_key():
        st.info("Live questions are switched off in this copy (no API key set). The other tabs show real recorded runs.")
    elif asked >= SESSION_LIMIT or counter["count"] >= DAILY_LIMIT:
        st.info("The limit for live questions is reached (each one costs a little money). The recorded runs are in the other tabs.")
    else:
        with st.form("ask"):
            question = st.text_input("Your question", max_chars=300)
            go_button = st.form_submit_button("Ask", type="primary")
        if go_button and question.strip():
            os.environ["ANTHROPIC_API_KEY"] = api_key()
            st.session_state["asked"] = asked + 1
            counter["count"] += 1
            with st.status("Working…", expanded=True) as status:
                try:
                    trace = run_agent(question.strip(), max_steps=6, on_tool_call=lambda call: st.write(f"Step {call.step}: `{call.name}`"))
                    status.update(label=f"Done in {trace.seconds} s · {trace.model_calls} model calls", state="complete", expanded=False)
                    st.session_state["live_trace"] = trace.to_dict()
                except Exception as error:
                    status.update(label="The model could not be reached", state="error")
                    st.error(f"{type(error).__name__}: {error}")
        if "live_trace" in st.session_state:
            show_steps(st.session_state["live_trace"]["tool_calls"], st.session_state["live_trace"]["answer"])
        st.caption(f"{SESSION_LIMIT - st.session_state.get('asked', 0)} live questions left in this visit.")

with tab_how:
    st.markdown("#### The loop")
    st.graphviz_chart("""
    digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4fc", color="#2a78d6", fontname="Helvetica", fontsize=11];
      edge [color="#667085", fontname="Helvetica", fontsize=10];
      q [label="Question"]; m [label="Model decides:\\nanswer or call a tool?"]; t [label="Python runs the tool\\n(arguments checked first)"]; a [label="Final answer"];
      q -> m; m -> t [label="tool call"]; t -> m [label="result or error message"]; m -> a [label="answer"]; }
    """)
    st.markdown(
        f"The model never runs anything itself. It plans; the code runs. The loop stops after 8 steps at the latest. Default model: `{DEFAULT_MODEL}`."
    )
    st.markdown("#### The tools")
    st.dataframe(pd.DataFrame([{"Tool": t.name, "What it is for": t.description, "Arguments": ", ".join(t.parameters) or "none"} for t in ANALYST_TOOLS]),
                 hide_index=True, width="stretch")
    st.markdown(
        "#### Why the test is fair\n"
        "- The correct answers are **computed from the data by code**, not written by hand and not judged by another model.\n"
        "- Some questions ask about companies or years that are **not in the data**. The only correct answer there is to say so.\n"
        "- All setups get the same questions, the same model and the same grading rules.\n\n"
        "Code: [github.com/kishan244/tool-calling-agent](https://github.com/kishan244/tool-calling-agent) · "
        "Data: [Risk Radar](https://github.com/kishan244/risk-narrative-nlp)"
    )
