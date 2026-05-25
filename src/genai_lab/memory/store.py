"""Long-term memory store abstractions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable

from langgraph.store.memory import InMemoryStore

from genai_lab.memory.entity import extract_entities
from genai_lab.memory.schema import ConversationTurn, MemoryKind, MemoryRecord, MemorySearchResult


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in TOKEN_RE.findall(text))


def _lexical_score(query: str, text: str) -> float:
    query_terms = Counter(_tokens(query))
    text_terms = Counter(_tokens(text))
    if not query_terms or not text_terms:
        return 0.0
    overlap = sum(min(text_terms[term], count) for term, count in query_terms.items())
    return overlap / max(1, sum(query_terms.values()))


@dataclass(frozen=True)
class MemoryNamespace:
    """Memory scope selector."""

    user_id: str
    agent_id: str = "research-assistant"

    @property
    def user(self) -> tuple[str, ...]:
        return ("memory", "user", self.user_id)

    @property
    def agent(self) -> tuple[str, ...]:
        return ("memory", "agent", self.agent_id)


class LongTermMemoryStore:
    """LangGraph-store-backed long-term memory.

    In production this boundary can be backed by PGVector for semantic search.
    Module 5 uses LangGraph's store API plus deterministic lexical ranking so
    tests remain local and stable.
    """

    def __init__(self, store: InMemoryStore | None = None) -> None:
        self.store = store or InMemoryStore()

    def put(self, record: MemoryRecord) -> None:
        self.store.put(record.namespace, record.key, record.as_store_value())

    def search(
        self,
        namespace: tuple[str, ...],
        query: str,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int = 5,
    ) -> tuple[MemorySearchResult, ...]:
        allowed = set(kinds) if kinds else None
        raw_items = self.store.search(namespace, query=query, limit=100)
        results: list[MemorySearchResult] = []
        for item in raw_items:
            kind = MemoryKind(str(item.value["kind"]))
            if allowed and kind not in allowed:
                continue
            text = str(item.value["text"])
            score = _lexical_score(query, text)
            results.append(
                MemorySearchResult(
                    key=item.key,
                    text=text,
                    kind=kind,
                    score=score,
                    metadata={str(k): str(v) for k, v in dict(item.value["metadata"]).items()},
                )
            )
        results.sort(key=lambda result: result.score, reverse=True)
        return tuple(results[:limit])

    def persist_turn(self, namespace: MemoryNamespace, turn: ConversationTurn) -> None:
        """Write episodic and entity memories from a conversation turn."""

        self.put(
            MemoryRecord(
                text=turn.text,
                kind=MemoryKind.EPISODIC,
                namespace=namespace.user,
                metadata={"source": "conversation_turn"},
            )
        )
        for entity in extract_entities(turn.text):
            self.put(
                MemoryRecord(
                    text=f"Entity mentioned by user {namespace.user_id}: {entity}",
                    kind=MemoryKind.ENTITY,
                    namespace=namespace.user,
                    metadata={"entity": entity},
                )
            )

    def remember_semantic_fact(
        self,
        namespace: MemoryNamespace,
        fact: str,
        *,
        source: str = "user",
    ) -> None:
        """Store a durable semantic memory."""

        self.put(
            MemoryRecord(
                text=fact,
                kind=MemoryKind.SEMANTIC,
                namespace=namespace.user,
                metadata={"source": source},
            )
        )
