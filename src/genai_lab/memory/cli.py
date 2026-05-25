"""CLI demo for Module 5 memory."""

from __future__ import annotations

import argparse

from genai_lab.config.settings import get_settings
from genai_lab.memory.manager import MemoryManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local memory demo.")
    parser.add_argument("query", help="Query used to retrieve memory.")
    parser.add_argument("--user-id", default="demo-user")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manager = MemoryManager.create(get_settings(), user_id=args.user_id)
    manager.remember_fact("The user is studying RAG, agents, memory, LangGraph, and MCP.")
    manager.remember_fact("The user wants clear explanations of embedding drift and retrieval quality.")
    manager.add_turn(
        "Please remember that I prefer staff-engineer trade-off explanations.",
        "Noted. I will emphasize architecture trade-offs and production failure modes.",
    )
    snapshot = manager.build_snapshot(args.query)
    print("Short-term context:")
    print(snapshot.short_term_context)
    print("\nLong-term memory:")
    for result in snapshot.long_term_context:
        print(f"- {result.kind.value} score={result.score:.2f}: {result.text}")
    print("\nEntity memory:")
    for result in snapshot.entity_context:
        print(f"- {result.metadata.get('entity', result.text)}")


if __name__ == "__main__":
    main()
