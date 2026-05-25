"""MCP tool data structures."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class McpToolCall:
    """A call to an external MCP tool."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class McpToolResult:
    """Normalized result returned by an MCP tool provider."""

    name: str
    payload: dict[str, Any]
    provider: str

