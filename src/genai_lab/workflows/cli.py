"""CLI for the Module 4 LangGraph workflow."""

from __future__ import annotations

import argparse
from uuid import uuid4

from genai_lab.config.settings import get_settings
from genai_lab.workflows.graph import build_research_workflow, build_workflow_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the research LangGraph workflow.")
    parser.add_argument("task", help="Research task or question.")
    parser.add_argument("--thread-id", default=None, help="Checkpoint thread id.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    graph = build_research_workflow(settings)
    thread_id = args.thread_id or f"workflow-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"task": args.task}, config=config, context=build_workflow_context(settings))
    if "__interrupt__" in result:
        print("Workflow interrupted for human review:")
        print(result["__interrupt__"])
        print(f"Resume with the same thread id: {thread_id}")
        return
    print(result.get("final_answer", "No final answer produced."))
    print("\nAudit log:")
    for event in result.get("audit_log", []):
        print(f"- {event}")


if __name__ == "__main__":
    main()
