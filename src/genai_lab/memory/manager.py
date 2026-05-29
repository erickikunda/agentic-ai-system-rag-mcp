"""Memory manager that combines short-term, long-term, and entity memory."""

from __future__ import annotations

from dataclasses import dataclass

from genai_lab.config.settings import RuntimeProfile, Settings
from genai_lab.memory.schema import ConversationTurn, MemoryKind, MemorySnapshot
from genai_lab.memory.short_term import ConversationBuffer
from genai_lab.memory.store import BaseMemoryStore, LongTermMemoryStore, MemoryNamespace


@dataclass
class MemoryManager:
    """Facade for writing and retrieving scoped memory."""

    settings: Settings
    namespace: MemoryNamespace
    long_term: BaseMemoryStore
    short_term: ConversationBuffer

    @classmethod
    def create(cls, settings: Settings, *, user_id: str) -> "MemoryManager":
        long_term: BaseMemoryStore
        if settings.profile == RuntimeProfile.PROD:
            from genai_lab.memory.store import PgLongTermMemoryStore
            long_term = PgLongTermMemoryStore(
                dsn=settings.postgres_dsn.get_secret_value(),
                embedding_dimension=settings.embedding_dimension,
            )
        else:
            long_term = LongTermMemoryStore()

        return cls(
            settings=settings,
            namespace=MemoryNamespace(user_id=user_id),
            long_term=long_term,
            short_term=ConversationBuffer(
                max_turns=settings.memory_buffer_max_turns,
                summary_trigger_tokens=settings.memory_summary_trigger_tokens,
            ),
        )

    def add_turn(self, user: str, assistant: str) -> None:
        """Write a turn to short-term and episodic/entity long-term memory."""
        self.short_term.add_turn(user, assistant)
        self.long_term.persist_turn(self.namespace, ConversationTurn(user=user, assistant=assistant))

    def remember_fact(self, fact: str, *, source: str = "user") -> None:
        self.long_term.remember_semantic_fact(self.namespace, fact, source=source)

    def build_snapshot(self, query: str) -> MemorySnapshot:
        """Retrieve prompt-ready memory for a new task."""
        semantic_and_episodic = self.long_term.search(
            self.namespace.user,
            query,
            kinds=(MemoryKind.SEMANTIC, MemoryKind.EPISODIC),
            limit=self.settings.memory_retrieval_limit,
        )
        entities = self.long_term.search(
            self.namespace.user,
            query,
            kinds=(MemoryKind.ENTITY,),
            limit=self.settings.memory_retrieval_limit,
        )
        return MemorySnapshot(
            short_term_context=self.short_term.render(),
            long_term_context=semantic_and_episodic,
            entity_context=entities,
            summary=self.short_term.rolling_summary,
        )
