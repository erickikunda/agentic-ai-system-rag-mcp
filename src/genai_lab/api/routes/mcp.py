"""MCP domain tool endpoints."""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter, HTTPException

from genai_lab.api.schemas import McpRequest
from genai_lab.config.settings import get_settings
from genai_lab.mcp.client import build_mcp_provider
from genai_lab.mcp.schema import McpToolCall

router = APIRouter()

_VALID_TOOLS = {"normalize_claim", "lookup_venue_metadata", "build_reading_plan"}


@router.post("/mcp/{tool}")
def call_mcp_tool(tool: str, req: McpRequest) -> dict[str, object]:
    if tool not in _VALID_TOOLS:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool}")
    settings = get_settings()
    provider = build_mcp_provider(settings)
    arguments: dict[str, str] = {
        "normalize_claim": {"claim": req.value},
        "lookup_venue_metadata": {"venue": req.value},
        "build_reading_plan": {"topic": req.value, "level": req.level},
    }[tool]
    result = provider.call_tool(McpToolCall(name=tool, arguments=arguments))
    return dataclasses.asdict(result)
