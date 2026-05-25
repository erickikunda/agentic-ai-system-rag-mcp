"""CLI for local MCP tool-provider calls."""

from __future__ import annotations

import argparse
import json

from genai_lab.config.settings import get_settings
from genai_lab.mcp.client import build_mcp_provider
from genai_lab.mcp.schema import McpToolCall


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call a genai-lab MCP domain tool.")
    parser.add_argument("tool", choices=["normalize_claim", "lookup_venue_metadata", "build_reading_plan"])
    parser.add_argument("value", help="Claim, venue, or topic value.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    provider = build_mcp_provider(get_settings())
    arguments = {
        "normalize_claim": {"claim": args.value},
        "lookup_venue_metadata": {"venue": args.value},
        "build_reading_plan": {"topic": args.value, "level": "senior CS student"},
    }[args.tool]
    result = provider.call_tool(McpToolCall(name=args.tool, arguments=arguments))
    print(json.dumps(result.payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

