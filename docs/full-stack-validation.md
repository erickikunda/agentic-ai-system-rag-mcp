# Full-Stack Validation Guide

This guide runs the full local stack: Ollama, Postgres with PGVector, the Python app image, and the Spring Boot MCP server.

## 1. Confirm local prerequisites

Run these from the repository root:

```bash
docker compose version
docker version
```

If you want to run Python commands outside Docker too:

```bash
.venv/bin/python -m pytest tests
```

## 2. Create a local environment file

Copy the example file if `.env` does not exist:

```bash
cp .env.example .env
```

For commands run on your host machine, set service URLs to localhost:

```bash
GENAI_LAB_OLLAMA_BASE_URL=http://localhost:11434
GENAI_LAB_POSTGRES_DSN=postgresql://genai_lab:genai_lab@localhost:5432/genai_lab
GENAI_LAB_MCP_SERVER_URL=http://localhost:8080
```

For commands run inside the Compose `app` container, keep the service names:

```bash
GENAI_LAB_OLLAMA_BASE_URL=http://ollama:11434
GENAI_LAB_POSTGRES_DSN=postgresql://genai_lab:genai_lab@postgres:5432/genai_lab
GENAI_LAB_MCP_SERVER_URL=http://mcp-server:8080
```

## 3. Build and start infrastructure

Start Postgres, Ollama, and the MCP server:

```bash
docker compose --profile mcp up --build postgres ollama mcp-server
```

In a second terminal, confirm Postgres is healthy:

```bash
docker compose ps
```

## 4. Pull the embedding model into Ollama

The ingestion path defaults to `nomic-embed-text`.

```bash
docker compose exec ollama ollama pull nomic-embed-text
```

Optional, if you want local chat-model experiments later:

```bash
docker compose exec ollama ollama pull llama3.1
```

## 5. Run a dry ingestion first

From your host:

```bash
GENAI_LAB_OLLAMA_BASE_URL=http://localhost:11434 \
GENAI_LAB_POSTGRES_DSN=postgresql://genai_lab:genai_lab@localhost:5432/genai_lab \
.venv/bin/python -m genai_lab.ingestion.cli --dry-run
```

Expected result: documents load and chunks are created, but no embeddings are written.

## 6. Run real ingestion into PGVector

```bash
GENAI_LAB_OLLAMA_BASE_URL=http://localhost:11434 \
GENAI_LAB_POSTGRES_DSN=postgresql://genai_lab:genai_lab@localhost:5432/genai_lab \
.venv/bin/python -m genai_lab.ingestion.cli
```

Expected result: the command reports `vector_store=pgvector`.

## 7. Verify vectors were written

Connect to Postgres:

```bash
docker compose exec postgres psql -U genai_lab -d genai_lab
```

Then inspect tables:

```sql
\dt
select count(*) from data_research_chunks;
```

If the table name differs, use `\dt` to find the generated PGVector table name.

## 8. Run vector-backed RAG

```bash
GENAI_LAB_OLLAMA_BASE_URL=http://localhost:11434 \
GENAI_LAB_POSTGRES_DSN=postgresql://genai_lab:genai_lab@localhost:5432/genai_lab \
.venv/bin/python -m genai_lab.rag.cli "What is embedding drift?" --use-vector-store --trace
```

Expected result: an answer with citations and a local JSONL trace path.

## 9. Exercise the MCP-facing agent path

The current deterministic Python agent uses the local MCP-shaped provider by default, which is good for fast local checks:

```bash
.venv/bin/python -m genai_lab.agents.cli "Build a reading plan for MCP integration" --trace
```

Expected result: `tools_used` includes `mcp_build_reading_plan`.

## 10. Run the evaluation suite

```bash
.venv/bin/python -m genai_lab.evaluation.cli
```

Use this as a smoke signal after retrieval changes. Local metrics are deterministic and cheap; they are not a replacement for model-judged production evaluation.

## 11. Shut down or reset

Stop containers but keep volumes:

```bash
docker compose --profile mcp down
```

Delete Postgres and Ollama volumes for a clean rebuild:

```bash
docker compose --profile mcp down -v
```

## Common failure points

- If ingestion cannot reach Ollama, check whether you are running on host (`localhost:11434`) or inside Compose (`ollama:11434`).
- If PGVector connection fails from the host, use `localhost:5432`; inside Compose, use `postgres:5432`.
- If vector-backed RAG returns no results, confirm ingestion completed and the embedding model matches the query-time model.
- If Compose app build fails, run `docker compose --profile app --profile mcp config` first to validate topology before building images.
