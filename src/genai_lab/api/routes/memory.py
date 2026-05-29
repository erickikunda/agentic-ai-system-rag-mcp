"""Memory snapshot endpoint."""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter

from genai_lab.api.schemas import MemoryQuery
from genai_lab.config.settings import get_settings
from genai_lab.memory.manager import MemoryManager

router = APIRouter()


@router.post("/memory")
def get_memory(req: MemoryQuery) -> dict[str, object]:
    settings = get_settings()
    manager = MemoryManager.create(settings, user_id=req.user_id)
    manager.remember_fact("The user is studying RAG, agents, memory, LangGraph, and MCP.")
    manager.remember_fact("The user wants clear explanations of embedding drift and retrieval quality.")
    snapshot = manager.build_snapshot(req.query)
    return {
        "short_term_context": snapshot.short_term_context,
        "summary": snapshot.summary,
        "long_term_context": [dataclasses.asdict(r) for r in snapshot.long_term_context],
        "entity_context": [dataclasses.asdict(r) for r in snapshot.entity_context],
    }
