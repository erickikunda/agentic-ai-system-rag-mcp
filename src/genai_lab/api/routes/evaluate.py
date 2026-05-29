"""RAG evaluation endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from genai_lab.api.schemas import EvaluateRequest
from genai_lab.config.settings import get_settings
from genai_lab.evaluation.runner import result_to_dict, run_rag_evaluation, summarize_results

router = APIRouter()


@router.post("/evaluate")
def run_evaluation(req: EvaluateRequest) -> dict[str, object]:
    settings = get_settings()
    dataset_path = Path(req.dataset_path) if req.dataset_path else None
    results = run_rag_evaluation(settings=settings, dataset_path=dataset_path)
    return {
        "summary": summarize_results(results),
        "results": [result_to_dict(r) for r in results],
    }
