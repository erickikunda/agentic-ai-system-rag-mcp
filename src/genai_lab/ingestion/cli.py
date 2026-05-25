"""Command-line entry point for Module 1 ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from genai_lab.config.settings import get_settings
from genai_lab.ingestion.chunking import ChunkingProfile
from genai_lab.ingestion.pipeline import run_ingestion


def build_parser() -> argparse.ArgumentParser:
    """Create the ingestion CLI parser."""

    parser = argparse.ArgumentParser(description="Ingest research documents with LlamaIndex.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of source documents.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Token chunk size.")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Token overlap.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse documents into nodes without embedding or writing to PGVector.",
    )
    return parser


def main() -> None:
    """Run ingestion from the command line."""

    settings = get_settings()
    args = build_parser().parse_args()
    input_dir = args.input_dir or Path(settings.ingestion_input_dir)
    chunking = ChunkingProfile(
        chunk_size_tokens=args.chunk_size or settings.chunk_size_tokens,
        chunk_overlap_tokens=args.chunk_overlap or settings.chunk_overlap_tokens,
    )

    result = run_ingestion(
        settings=settings,
        input_dir=input_dir,
        chunking=chunking,
        dry_run=args.dry_run,
    )
    mode = "dry run" if result.dry_run else "indexed"
    print(
        f"Ingestion {mode}: loaded {result.documents_loaded} documents, "
        f"created {result.nodes_created} nodes, vector_store={result.vector_store}"
    )


if __name__ == "__main__":
    main()

