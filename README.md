# Tool-Calling Agent

![tests](https://github.com/kishan244/tool-calling-agent/actions/workflows/tests.yml/badge.svg)

A small AI agent in plain Python. It decides when it needs a tool, calls it,
and uses the real result to answer — instead of guessing.

No framework. The reason → act → observe loop (ReAct) is written out directly,
which is what LangGraph and similar libraries wrap.

## What it does

- Answers general questions directly, with no tool call
- Searches the web when a question needs current information
- Uses a calculator for arithmetic, rather than predicting the number

Example run:

```
You: What is 847 * 293?
  [calling calculator({'expression': '847 * 293'})]

Agent: 847 * 293 = 248,171
```

## How it works

1. The question and the tool definitions go to the model
2. The model either answers, or asks for a specific tool with specific arguments
3. If it asks for a tool, `agent.py` runs the real Python function
4. The result is sent back and the model continues from there

The model never executes anything itself. It plans; the code runs.

## Setup

Requires an Anthropic API key from [console.anthropic.com](https://console.anthropic.com).

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your key to .env
python agent.py
```

`.env` is gitignored and never committed.

## Files

| File | Purpose |
|---|---|
| `agent.py` | The ReAct loop and tool definitions |
| `tools.py` | `web_search` (ddgs) and a safe `calculator` |
| `tests/` | pytest tests for the calculator |
| `requirements.txt` | Dependencies |
| `.env.example` | Template for the API key |

## Safety

- The calculator never uses `eval`. It parses the expression into a syntax tree and only
  allows numbers and `+ - * / // % **` with brackets, so it can't run code. Huge powers
  like `9**9**9` are refused so the program can't freeze.
- The loop stops after `MAX_STEPS` tool calls, so a confused model can't loop forever.
- Unknown tools and failed tool calls are sent back to the model marked as errors,
  so it can try another way instead of crashing.

## Tests

```bash
pip install pytest
pytest -q
```

GitHub Actions runs the tests on every push.
