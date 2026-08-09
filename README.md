# Tool-Calling Agent

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
| `tools.py` | `web_search` (ddgs) and `calculator` |
| `requirements.txt` | Dependencies |
| `.env.example` | Template for the API key |

## Notes

The calculator restricts input to digits, operators, parentheses and spaces
before evaluating, so it can't be used to run arbitrary code.

Web search uses `ddgs`, the current name of the package formerly published as
`duckduckgo-search`.

## Next

RAG version with a vector database, answering questions over a real document set.
