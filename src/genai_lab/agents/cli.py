"""Command-line entry point for the Module 3 tool-using agent."""

from __future__ import annotations

import argparse

from genai_lab.config.settings import get_settings

from .executor import ToolUsingResearchAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local tool-using research agent.")
    parser.add_argument("task", help="Research task or question.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    agent = ToolUsingResearchAgent(settings=get_settings())
    run = agent.run(args.task)
    print(run.response.model_dump_json(indent=2))
    if run.traces:
        print("\nTool trace:")
        for trace in run.traces:
            state = "ok" if trace.ok else "error"
            print(f"- {trace.tool_name} attempt={trace.attempt} {state}: {trace.output_summary}")


if __name__ == "__main__":
    main()

