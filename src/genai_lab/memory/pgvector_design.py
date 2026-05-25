"""PGVector memory design helpers.

Module 5 keeps tests on LangGraph's in-memory store. This module records the
SQL shape used when swapping the long-term memory store to PGVector.
"""


MEMORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_memory (
  id UUID PRIMARY KEY,
  user_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  embedding VECTOR(:embedding_dimension),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_accessed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS agent_memory_scope_idx
  ON agent_memory (user_id, agent_id, kind);

CREATE INDEX IF NOT EXISTS agent_memory_embedding_idx
  ON agent_memory USING hnsw (embedding vector_cosine_ops);
"""

