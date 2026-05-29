"""PGVector memory table DDL.

Returns the CREATE TABLE / index SQL for the agent_memory table used by
PgLongTermMemoryStore. The embedding dimension is injected at runtime so the
column type matches the active embedding model.
"""


def memory_table_sql(embedding_dimension: int) -> str:
    """Return DDL for the agent_memory table and supporting indexes."""
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS agent_memory (
  id               UUID         PRIMARY KEY,
  user_id          TEXT         NOT NULL,
  agent_id         TEXT         NOT NULL DEFAULT 'research-assistant',
  kind             TEXT         NOT NULL,
  text             TEXT         NOT NULL,
  metadata         JSONB        NOT NULL DEFAULT '{{}}'::jsonb,
  embedding        VECTOR({embedding_dimension}),
  created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
  last_accessed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS agent_memory_scope_idx
  ON agent_memory (user_id, agent_id, kind);

CREATE INDEX IF NOT EXISTS agent_memory_embedding_idx
  ON agent_memory USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
"""
