from pathlib import Path
import json
import tempfile
import unittest

from genai_lab.config.settings import Settings
from genai_lab.evaluation.metrics import score_rag_answer
from genai_lab.evaluation.runner import EvaluationCase, run_rag_evaluation, summarize_results
from genai_lab.observability.prompt_registry import PromptRegistry
from genai_lab.observability.tracing import JsonlTraceWriter, SpanTimer, build_langsmith_environment
from genai_lab.rag.schema import Citation, RagAnswer, QueryTransform, RetrievalStrategy


class ObservabilityEvaluationTests(unittest.TestCase):
    def test_trace_writer_redacts_sensitive_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = JsonlTraceWriter(Path(temp_dir))
            event = SpanTimer(name="agent.tool", kind="tool", run_id="run-1").finish(
                status="ok",
                inputs={"api_key": "secret", "question": "What is RAG?"},
            )

            path = writer.write(event)
            payload = json.loads(path.read_text(encoding="utf-8").strip())

            self.assertEqual(payload["inputs"]["api_key"], "[redacted]")
            self.assertEqual(payload["inputs"]["question"], "What is RAG?")

    def test_prompt_registry_renders_versioned_prompt(self) -> None:
        prompt = PromptRegistry().render(
            "rag-answer-v1",
            question="What is retrieval?",
            context="Retrieval finds relevant context.",
        )

        self.assertIn("What is retrieval?", prompt)
        self.assertIn("citation ids", prompt)

    def test_langsmith_environment_uses_settings_without_mutating_env(self) -> None:
        environment = build_langsmith_environment(
            Settings(langsmith_tracing=True, langsmith_project="genai-lab-test")
        )

        self.assertEqual(environment["LANGSMITH_TRACING"], "true")
        self.assertEqual(environment["LANGSMITH_PROJECT"], "genai-lab-test")

    def test_rag_metric_scores_grounded_answer(self) -> None:
        answer = RagAnswer(
            question="What is embedding drift?",
            transformed_queries=("What is embedding drift?",),
            answer="Embedding drift happens when the embedding model changes. [1]",
            citations=(
                Citation(
                    source_id=1,
                    source_name="note",
                    source_path="note.md",
                    score=0.9,
                    snippet="Embedding drift happens when the embedding model or preprocessing changes.",
                ),
            ),
            strategy=RetrievalStrategy.SIMILARITY,
            transform=QueryTransform.NONE,
        )

        score = score_rag_answer(
            answer=answer,
            reference_answer="Embedding drift happens when the embedding model changes.",
            required_terms=("embedding", "drift", "model"),
        )

        self.assertGreaterEqual(score.faithfulness, 0.7)
        self.assertEqual(score.citation_coverage, 1.0)

    def test_evaluation_runner_returns_summary(self) -> None:
        results = run_rag_evaluation(
            settings=Settings(),
            cases=(
                EvaluationCase(
                    case_id="drift",
                    question="What is embedding drift?",
                    reference_answer="Embedding drift happens when model changes make vectors inconsistent.",
                    required_terms=("embedding", "drift", "model"),
                ),
            ),
        )

        summary = summarize_results(results)

        self.assertEqual(len(results), 1)
        self.assertIn("overall", summary)
        self.assertGreater(summary["overall"], 0.0)


if __name__ == "__main__":
    unittest.main()
