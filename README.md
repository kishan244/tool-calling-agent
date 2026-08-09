# Project 1: Tool-Calling Agent (Week 1-2)

This is your first agentic AI project. The goal: build an agent that can
decide *when* it needs a tool, call that tool, and use the result to answer
a question — instead of just guessing from memory.

This pattern (reason → act → observe → repeat) is called **ReAct**, and it's
the foundation every agent framework (LangGraph, CrewAI, AutoGen) is built
on top of. Learn this by hand first, in plain Python, before using a
framework — it's much easier to debug a framework later if you understand
what it's doing underneath.

## What this agent can do

- Answer normal questions directly (no tool needed)
- Decide when a question needs live web info, and search for it
- Decide when a question needs math, and calculate it precisely instead of
  guessing (LLMs are bad at arithmetic — this is a real, common failure
  mode you're fixing)

## Setup (do this once)

### 1. Get an Anthropic API key

1. Go to https://console.anthropic.com and sign up (separate from your
   Claude Pro subscription — this is billed separately, pay-as-you-go)
2. Add a small amount of credit — $5 is plenty to build and test this
   entire project
3. Go to "API Keys" in the console, create a new key, copy it
   (starts with `sk-ant-...`)

### 2. Set up your Python environment

Open this folder (`agent-project-01`) in VS Code. Open a terminal inside
VS Code (`` Ctrl+` `` or `` Cmd+` ``) and run:

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API key

1. Copy `.env.example` to a new file called `.env`
2. Open `.env` and paste your API key after `ANTHROPIC_API_KEY=`
3. **Never commit `.env` to GitHub** — it's already in `.gitignore` for you.
   This is a real security habit recruiters and engineers care about.

### 4. Run it

```bash
python agent.py
```

## Files

- `agent.py` — the agent itself, heavily commented
- `tools.py` — the two tools the agent can use (web search, calculator)
- `requirements.txt` — Python packages needed
- `.env.example` — template for your API key (copy to `.env`)
- `.gitignore` — makes sure your API key never gets pushed to GitHub

## What to do after it runs

1. Read `agent.py` top to bottom — every step is commented so you
   understand *why*, not just *what*
2. Try asking it: "What's 847 * 392?" (should use the calculator)
3. Try asking it: "What's the latest news about the Bundesliga?" (should
   search)
4. Try asking it: "What is the capital of France?" (should answer directly,
   no tool)
5. Once it works, push this to a public GitHub repo with a good README —
   this becomes portfolio project #1

## Next steps (Week 3-4)

Once this works and you understand it, we'll add a vector database and
build a RAG (retrieval-augmented generation) agent that answers questions
from your own documents — e.g. your Master's thesis or the air-quality
dataset from your Flask dashboard project.
