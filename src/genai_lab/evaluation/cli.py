"""Command-line RAG evaluation runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from genai_lab.config.settings import get_settings

from .runner import result_to_dict, run_rag_evaluation, summarize_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local RAG evaluation suite.")
    parser.add_argument("--dataset", type=Path, default=None, help="JSONL evaluation dataset path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    results = run_rag_evaluation(settings=settings, dataset_path=args.dataset)
    payload = {
        "summary": summarize_results(results),
        "results": [result_to_dict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
