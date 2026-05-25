import unittest

from genai_lab.config.settings import Settings
from genai_lab.memory.entity import extract_entities
from genai_lab.memory.manager import MemoryManager
from genai_lab.memory.short_term import ConversationBuffer


class MemoryManagerTests(unittest.TestCase):
    def test_short_term_memory_summarizes_old_turns(self) -> None:
        buffer = ConversationBuffer(max_turns=10, summary_trigger_tokens=20)
        buffer.add_turn("Explain LangGraph checkpointing in detail", "Checkpointing stores graph state.")
        buffer.add_turn("Now explain PGVector memory", "PGVector stores embedded long-term memories.")

        self.assertIsNotNone(buffer.rolling_summary)
        self.assertEqual(len(buffer.turns), 1)
        self.assertIn("PGVector", buffer.render())

    def test_memory_manager_retrieves_semantic_memory(self) -> None:
        manager = MemoryManager.create(Settings(memory_retrieval_limit=3), user_id="alice")
        manager.remember_fact("Alice prefers detailed explanations about embedding drift.")

        snapshot = manager.build_snapshot("embedding drift preferences")

        self.assertTrue(snapshot.long_term_context)
        self.assertIn("embedding drift", snapshot.long_term_context[0].text)

    def test_turn_persists_episodic_and_entity_memory(self) -> None:
        manager = MemoryManager.create(Settings(memory_retrieval_limit=5), user_id="bob")
        manager.add_turn("Compare LangGraph and LangChain memory", "LangGraph uses checkpointed state.")

        snapshot = manager.build_snapshot("LangGraph memory")

        self.assertTrue(any(result.kind.value == "episodic" for result in snapshot.long_term_context))
        self.assertTrue(any("LangGraph" in result.text for result in snapshot.entity_context))

    def test_entity_extraction_keeps_key_tech_terms(self) -> None:
        entities = extract_entities("We use LangGraph, PGVector, and MCP for Eric's RAG lab.")

        self.assertIn("LangGraph", entities)
        self.assertIn("PGVector", entities)
        self.assertIn("MCP", entities)


if __name__ == "__main__":
    unittest.main()

