"""RAG query endpoint."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from fastapi import APIRouter

from genai_lab.api.schemas import RagRequest
from genai_lab.config.settings import get_settings
from genai_lab.rag.engine import RagQueryEngine, build_rag_config
from genai_lab.rag.schema import QueryTransform, RetrievalStrategy

router = APIRouter()


@router.post("/rag")
def query_rag(req: RagRequest) -> dict[str, object]:
    settings = get_settings()
    config = build_rag_config(settings, use_vector_store=req.use_vector_store)
    from dataclasses import replace
    config = replace(
        config,
        strategy=RetrievalStrategy(req.strategy),
        transform=QueryTransform(req.transform),
    )
    engine = RagQueryEngine(settings=settings, config=config)
    answer = engine.query(req.question)
    return dataclasses.asdict(answer)
