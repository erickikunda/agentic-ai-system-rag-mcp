import json
import unittest

from genai_lab.mcp.client import LocalMcpToolProvider
from genai_lab.mcp.langchain_tools import build_mcp_langchain_tools
from genai_lab.mcp.schema import McpToolCall


class McpProviderTests(unittest.TestCase):
    def test_local_provider_normalizes_claim(self) -> None:
        provider = LocalMcpToolProvider()

        result = provider.call_tool(
            McpToolCall(
                name="normalize_claim",
                arguments={"claim": "  embedding drift hurts retrieval quality  "},
            )
        )

        self.assertEqual(result.payload["normalizedClaim"], "embedding drift hurts retrieval quality.")
        self.assertIn("embedding", result.payload["keyTerms"])

    def test_langchain_wrappers_call_provider(self) -> None:
        tools = {tool.name: tool for tool in build_mcp_langchain_tools(LocalMcpToolProvider())}

        payload = json.loads(
            tools["mcp_lookup_venue_metadata"].invoke({"venue": "arXiv"})
        )

        self.assertEqual(payload["venueType"], "preprint repository")

    def test_reading_plan_tool_shape(self) -> None:
        provider = LocalMcpToolProvider()

        result = provider.call_tool(
            McpToolCall(
                name="build_reading_plan",
                arguments={"topic": "MCP integration", "level": "Python engineer"},
            )
        )

        self.assertEqual(result.payload["topic"], "MCP integration")
        self.assertEqual(len(result.payload["steps"]), 4)


if __name__ == "__main__":
    unittest.main()

