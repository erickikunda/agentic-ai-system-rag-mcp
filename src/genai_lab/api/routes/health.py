"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from genai_lab.config.settings import get_settings

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "profile": settings.profile.value}
