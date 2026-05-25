"""Command-line RAG query interface."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from genai_lab.config.settings import get_settings
from genai_lab.observability.tracing import trace_rag_answer
from genai_lab.rag.engine import RagQueryEngine, build_rag_config
from genai_lab.rag.schema import QueryTransform, RetrievalStrategy


def build_parser() -> argparse.ArgumentParser:
    """Create the RAG CLI parser."""

    parser = argparse.ArgumentParser(description="Ask the research assistant RAG engine.")
    parser.add_argument("question", help="Question to answer from the research corpus.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Local corpus directory.")
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in RetrievalStrategy],
        default=RetrievalStrategy.SIMILARITY.value,
        help="Retrieval strategy.",
    )
    parser.add_argument(
        "--transform",
        choices=[transform.value for transform in QueryTransform],
        default=QueryTransform.NONE.value,
        help="Query transformation.",
    )
    parser.add_argument(
        "--use-vector-store",
        action="store_true",
        help="Query PGVector instead of the local dry-run corpus.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Write a local JSONL trace event for this RAG query.",
    )
    return parser


def main() -> None:
    """Run a RAG query from the command line."""

    settings = get_settings()
    args = build_parser().parse_args()
    config = build_rag_config(settings, use_vector_store=args.use_vector_store)
    config = replace(
        config,
        strategy=RetrievalStrategy(args.strategy),
        transform=QueryTransform(args.transform),
    )
    engine = RagQueryEngine(settings=settings, config=config, input_dir=args.input_dir)
    answer = engine.query(args.question)

    print(answer.answer)
    print("\nCitations:")
    for citation in answer.citations:
        flags = f" flags={','.join(citation.flags)}" if citation.flags else ""
        print(
            f"[{citation.source_id}] {citation.source_name} "
            f"score={citation.score}{flags} :: {citation.snippet}"
        )
    if answer.warnings:
        print("\nWarnings:")
        for warning in answer.warnings:
            print(f"- {warning}")
    if args.trace:
        trace_path = trace_rag_answer(
            settings=settings,
            question=answer.question,
            answer=answer.answer,
            citation_count=len(answer.citations),
            warnings=answer.warnings,
        )
        print(f"\nTrace: {trace_path}")


if __name__ == "__main__":
    main()
