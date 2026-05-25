import json
import unittest

from langchain_core.tools import tool

from genai_lab.agents.executor import ToolUsingResearchAgent
from genai_lab.config.settings import Settings


class AgentExecutorTests(unittest.TestCase):
    def test_agent_answers_with_rag_tool_and_structured_output(self) -> None:
        agent = ToolUsingResearchAgent(settings=Settings())

        run = agent.run("What is embedding drift?")

        self.assertEqual(run.response.status, "answered")
        self.assertIn("rag_search", run.response.tools_used)
        self.assertTrue(run.response.citations)
        self.assertTrue(any(trace.ok for trace in run.traces))

    def test_agent_routes_compare_tasks_to_compare_tool(self) -> None:
        agent = ToolUsingResearchAgent(settings=Settings())

        run = agent.run("Compare RAG failure modes versus agentic workflows")

        self.assertEqual(run.response.status, "answered")
        self.assertEqual(run.response.tools_used, ["compare_sources"])
        self.assertIn("rag_failure_modes", run.response.answer)

    def test_tool_retry_failure_becomes_needs_review(self) -> None:
        @tool("rag_search")
        def failing_tool(question: str, strategy: str = "hybrid") -> str:
            """Always fail for retry testing."""

            raise RuntimeError("temporary backend outage")

        agent = ToolUsingResearchAgent(settings=Settings(agent_tool_retry_attempts=2), tools=[failing_tool])

        run = agent.run("What is embedding drift?")

        self.assertEqual(run.response.status, "needs_review")
        self.assertEqual(len(run.traces), 2)
        self.assertTrue(all(not trace.ok for trace in run.traces))

    def test_tool_outputs_are_json_objects(self) -> None:
        agent = ToolUsingResearchAgent(settings=Settings())
        tool_obj = agent.tools["citation_lookup"]

        result = json.loads(tool_obj.invoke({"source_name": "rag_failure_modes"}))

        self.assertEqual(result["source_name"], "rag_failure_modes.md")
        self.assertIn("source_path", result)

    def test_agent_can_route_to_mcp_reading_plan_tool(self) -> None:
        agent = ToolUsingResearchAgent(settings=Settings())

        run = agent.run("Build a reading plan for MCP integration")

        self.assertEqual(run.response.status, "answered")
        self.assertEqual(run.response.tools_used, ["mcp_build_reading_plan"])
        self.assertIn("1.", run.response.answer)


if __name__ == "__main__":
    unittest.main()
