"""Expose MCP provider tools as LangChain tools."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from genai_lab.mcp.client import McpToolProvider
from genai_lab.mcp.schema import McpToolCall


class NormalizeClaimInput(BaseModel):
    claim: str = Field(description="Raw claim to normalize.")


class VenueLookupInput(BaseModel):
    venue: str = Field(description="Venue/source name to inspect.")


class ReadingPlanInput(BaseModel):
    topic: str = Field(description="Topic to study.")
    level: str = Field(default="senior CS student", description="Learner level.")


def build_mcp_langchain_tools(provider: McpToolProvider) -> list[BaseTool]:
    """Create LangChain tool wrappers for MCP tools."""

    @tool(
        "mcp_normalize_claim",
        args_schema=NormalizeClaimInput,
        description="Normalize a research claim through the external MCP tool provider.",
    )
    def normalize_claim(claim: str) -> str:
        result = provider.call_tool(McpToolCall(name="normalize_claim", arguments={"claim": claim}))
        return json.dumps(result.payload, sort_keys=True)

    @tool(
        "mcp_lookup_venue_metadata",
        args_schema=VenueLookupInput,
        description="Look up venue/source metadata through the external MCP tool provider.",
    )
    def lookup_venue_metadata(venue: str) -> str:
        result = provider.call_tool(
            McpToolCall(name="lookup_venue_metadata", arguments={"venue": venue})
        )
        return json.dumps(result.payload, sort_keys=True)

    @tool(
        "mcp_build_reading_plan",
        args_schema=ReadingPlanInput,
        description="Build a study reading plan through the external MCP tool provider.",
    )
    def build_reading_plan(topic: str, level: str = "senior CS student") -> str:
        result = provider.call_tool(
            McpToolCall(name="build_reading_plan", arguments={"topic": topic, "level": level})
        )
        return json.dumps(result.payload, sort_keys=True)

    return [normalize_claim, lookup_venue_metadata, build_reading_plan]

