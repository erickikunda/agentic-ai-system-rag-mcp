"""Python client boundary for MCP domain tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from genai_lab.config.settings import Settings
from genai_lab.mcp.schema import McpToolCall, McpToolResult


class McpToolProvider(Protocol):
    """Provider interface consumed by Python agents."""

    def call_tool(self, call: McpToolCall) -> McpToolResult:
        """Call a named MCP tool and normalize the result."""


@dataclass
class LocalMcpToolProvider:
    """Local deterministic provider with the same contract as the Java MCP server."""

    provider_name: str = "local-fallback"

    def call_tool(self, call: McpToolCall) -> McpToolResult:
        if call.name == "normalize_claim":
            claim = str(call.arguments["claim"]).strip()
            normalized = " ".join(claim.split())
            if not normalized.endswith("."):
                normalized = f"{normalized}."
            terms = []
            for token in normalized.replace(".", "").split():
                lowered = token.lower()
                if len(lowered) > 4 and lowered not in terms:
                    terms.append(lowered)
            return McpToolResult(
                name=call.name,
                provider=self.provider_name,
                payload={
                    "normalizedClaim": normalized,
                    "keyTerms": terms[:8],
                    "confidenceNote": "Local fallback; verify against Java MCP server in integration.",
                },
            )
        if call.name == "lookup_venue_metadata":
            venue = str(call.arguments["venue"])
            lowered = venue.lower()
            if "arxiv" in lowered:
                payload = {
                    "venue": venue,
                    "venueType": "preprint repository",
                    "reviewModel": "not peer reviewed by default",
                    "caution": "Treat as useful but not final; prefer corroborating sources.",
                }
            else:
                payload = {
                    "venue": venue,
                    "venueType": "unknown or local source",
                    "reviewModel": "unknown",
                    "caution": "Require stronger citation metadata before treating as authoritative.",
                }
            return McpToolResult(name=call.name, provider=self.provider_name, payload=payload)
        if call.name == "build_reading_plan":
            topic = str(call.arguments["topic"])
            level = str(call.arguments.get("level") or "senior CS student")
            return McpToolResult(
                name=call.name,
                provider=self.provider_name,
                payload={
                    "topic": topic,
                    "steps": [
                        "Skim architecture notes and identify the system boundary.",
                        "Read implementation details for failure modes and trade-offs.",
                        "Run one example and trace state changes.",
                        "Write production questions for latency, correctness, and recovery.",
                    ],
                    "expectedOutcome": (
                        f"A {level} should explain the concept, failure modes, and when not to use it."
                    ),
                },
            )
        raise ValueError(f"Unknown MCP tool: {call.name}")


@dataclass
class StreamableHttpMcpToolProvider:
    """MCP Streamable HTTP provider.

    This provider documents the production integration boundary. The local test
    path uses ``LocalMcpToolProvider`` because it does not require a running
    Spring Boot server.
    """

    settings: Settings

    async def call_tool_async(self, call: McpToolCall) -> McpToolResult:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(self.settings.mcp_server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(call.name, call.arguments)
        return McpToolResult(
            name=call.name,
            provider=self.settings.mcp_server_url,
            payload={"content": [str(item) for item in result.content]},
        )


def build_mcp_provider(settings: Settings, *, local_fallback: bool = True) -> McpToolProvider:
    """Build the MCP provider used by Python agents."""

    if local_fallback:
        return LocalMcpToolProvider()
    raise RuntimeError("Use StreamableHttpMcpToolProvider.call_tool_async for live MCP calls.")

