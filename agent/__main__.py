"""Chat with the agent in the terminal: python -m agent"""
from dotenv import load_dotenv

from .loop import run_agent


def main() -> None:
    load_dotenv()
    print("Analyst agent ready. Ask about the risk factors of 20 US companies (or 'quit').\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        trace = run_agent(question, on_tool_call=lambda call: print(f"  [step {call.step}: {call.name}({call.arguments})]"))
        print(f"\nAgent: {trace.answer}\n  ({trace.model_calls} model calls, {trace.input_tokens + trace.output_tokens:,} tokens, {trace.seconds}s)\n")


if __name__ == "__main__":
    main()
