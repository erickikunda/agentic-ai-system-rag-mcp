import unittest

from langgraph.types import Command

from genai_lab.config.settings import Settings
from genai_lab.workflows.graph import build_research_workflow, build_workflow_context


class WorkflowGraphTests(unittest.TestCase):
    def test_workflow_produces_final_answer_with_audit_log(self) -> None:
        graph = build_research_workflow(Settings())
        context = build_workflow_context(Settings())
        result = graph.invoke(
            {"task": "What is embedding drift?"},
            config={"configurable": {"thread_id": "test-final"}},
            context=context,
        )

        self.assertIn("final_answer", result)
        self.assertTrue(result["citations"])
        self.assertTrue(any("planner:" in event for event in result["audit_log"]))
        self.assertTrue(any("evidence_worker:" in event for event in result["audit_log"]))

    def test_workflow_interrupts_and_resumes_for_review(self) -> None:
        graph = build_research_workflow(Settings())
        context = build_workflow_context(Settings())
        config = {"configurable": {"thread_id": "test-review"}}
        result = graph.invoke(
            {"task": "Guarantee this production answer is always correct."},
            config=config,
            context=context,
        )

        self.assertIn("__interrupt__", result)

        resumed = graph.invoke(Command(resume="approved"), config=config, context=context)

        self.assertEqual(resumed["human_decision"], "approved")
        self.assertIn("final_answer", resumed)

    def test_workflow_compiles_parallel_worker_edges(self) -> None:
        graph = build_research_workflow(Settings())
        graph_shape = graph.get_graph()
        node_names = {node.id for node in graph_shape.nodes.values()}

        self.assertIn("retrieve_evidence", node_names)
        self.assertIn("assess_risk", node_names)
        self.assertIn("human_review", node_names)


if __name__ == "__main__":
    unittest.main()
