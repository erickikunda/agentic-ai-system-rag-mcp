"""Command-line entry point for the Module 3 tool-using agent."""

from __future__ import annotations

import argparse

from genai_lab.config.settings import get_settings
from genai_lab.observability.tracing import JsonlTraceWriter, SpanTimer

from .executor import ToolUsingResearchAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local tool-using research agent.")
    parser.add_argument("task", help="Research task or question.")
    parser.add_argument("--trace", action="store_true", help="Write local JSONL trace events.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    timer = SpanTimer(name="agent.run", kind="agent")
    agent = ToolUsingResearchAgent(settings=settings)
    run = agent.run(args.task)
    print(run.response.model_dump_json(indent=2))
    if run.traces:
        print("\nTool trace:")
        for trace in run.traces:
            state = "ok" if trace.ok else "error"
            print(f"- {trace.tool_name} attempt={trace.attempt} {state}: {trace.output_summary}")
    if args.trace:
        writer = JsonlTraceWriter(settings.trace_log_dir)
        path = writer.write(
            timer.finish(
                status=run.response.status,
                inputs={"task": args.task},
                outputs={
                    "answer_preview": run.response.answer[:320],
                    "tools_used": run.response.tools_used,
                    "warnings": run.response.warnings,
                },
                metadata={
                    "tool_attempts": len(run.traces),
                    "observability_provider": settings.observability_provider.value,
                },
            )
        )
        print(f"\nTrace: {path}")


if __name__ == "__main__":
    main()
