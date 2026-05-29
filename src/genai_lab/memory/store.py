"""Long-term memory store — dev (InMemory) and prod (PostgreSQL) backends."""

from __future__ import annotations

import abc
import json
from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Iterable

import psycopg
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


class BaseMemoryStore(abc.ABC):
    """Shared interface for all long-term memory backends."""

    @abc.abstractmethod
    def put(self, record: MemoryRecord) -> None: ...

    @abc.abstractmethod
    def search(
        self,
        namespace: tuple[str, ...],
        query: str,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int = 5,
    ) -> tuple[MemorySearchResult, ...]: ...

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


class LongTermMemoryStore(BaseMemoryStore):
    """LangGraph InMemoryStore backend — used in dev and test profiles."""

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


class PgLongTermMemoryStore(BaseMemoryStore):
    """PostgreSQL+pgvector-backed long-term memory — used in prod profile.

    Search uses lexical ranking for now. Replace _lexical_score with a
    pgvector cosine similarity query once an embedding adapter is wired
    (deployment Phase 4).
    """

    def __init__(self, dsn: str, *, embedding_dimension: int = 768) -> None:
        self._conn: psycopg.Connection[Any] = psycopg.connect(dsn)
        self._ensure_table(embedding_dimension)

    def _ensure_table(self, embedding_dimension: int) -> None:
        from genai_lab.memory.pgvector_design import memory_table_sql
        with self._conn.cursor() as cur:
            cur.execute(memory_table_sql(embedding_dimension))
        self._conn.commit()

    def put(self, record: MemoryRecord) -> None:
        user_id = record.namespace[2] if len(record.namespace) >= 3 else record.namespace[-1]
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_memory (id, user_id, agent_id, kind, text, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (id) DO UPDATE
                  SET text     = EXCLUDED.text,
                      metadata = EXCLUDED.metadata
                """,
                (
                    record.key,
                    user_id,
                    "research-assistant",
                    record.kind.value,
                    record.text,
                    json.dumps(record.metadata),
                    record.created_at,
                ),
            )
        self._conn.commit()

    def search(
        self,
        namespace: tuple[str, ...],
        query: str,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int = 5,
    ) -> tuple[MemorySearchResult, ...]:
        user_id = namespace[2] if len(namespace) >= 3 else namespace[-1]
        allowed = list(kinds) if kinds else None

        sql = "SELECT id, kind, text, metadata FROM agent_memory WHERE user_id = %s"
        params: list[object] = [user_id]
        if allowed:
            sql += " AND kind = ANY(%s)"
            params.append([k.value for k in allowed])

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[MemorySearchResult] = []
        for row in rows:
            key, kind_str, text, metadata = row
            results.append(
                MemorySearchResult(
                    key=str(key),
                    text=str(text),
                    kind=MemoryKind(str(kind_str)),
                    score=_lexical_score(query, str(text)),
                    metadata={str(k): str(v) for k, v in (dict(metadata) if metadata else {}).items()},
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return tuple(results[:limit])

    def close(self) -> None:
        self._conn.close()
