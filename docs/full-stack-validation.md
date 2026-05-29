# Full-Stack Validation Guide

This guide runs the full local stack: Ollama, Postgres with PGVector, the Python CLI modules,
the FastAPI query-tier API, the React frontend, and the Spring Boot MCP server.

## 1. Confirm local prerequisites

Run these from the repository root:

```bash
docker compose version
docker version
node --version        # must be ≥ 18
```

If you want to run Python commands outside Docker too:

```bash
uv run pytest tests
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
uv run python -m genai_lab.evaluation.cli
```

Use this as a smoke signal after retrieval changes. Local metrics are deterministic and cheap; they are not a replacement for model-judged production evaluation.

## 11. Validate the prod memory store (PgLongTermMemoryStore)

With Postgres running, exercise the production memory backend:

```bash
GENAI_LAB_PROFILE=prod \
GENAI_LAB_POSTGRES_DSN=postgresql://genai_lab:genai_lab@localhost:5432/genai_lab \
.venv/bin/python -m genai_lab.memory.cli "embedding drift preferences"
```

Expected result: the CLI prints short-term context and long-term memory results. On first run,
`_ensure_table()` creates the `agent_memory` table in Postgres. Confirm with:

```bash
docker compose exec postgres psql -U genai_lab -d genai_lab -c "\dt"
# agent_memory table should appear
docker compose exec postgres psql -U genai_lab -d genai_lab \
  -c "select kind, left(text,60) from agent_memory;"
```

## 12. Start the FastAPI query-tier API

In a dedicated terminal, from the repository root:

```bash
GENAI_LAB_POSTGRES_DSN=postgresql://genai_lab:genai_lab@localhost:5432/genai_lab \
uv run genai-lab-api
```

Expected result: uvicorn starts on `http://0.0.0.0:8000` with `--reload`.

Confirm the health endpoint:

```bash
curl -s http://localhost:8000/api/health | python3 -m json.tool
# {"status": "ok", "profile": "dev"}
```

Smoke-test a RAG call:

```bash
curl -s -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"question": "What is embedding drift?", "strategy": "hybrid"}' \
  | python3 -m json.tool
```

Expected result: JSON with `answer`, `citations`, `strategy`, and `warnings` fields.

Smoke-test an MCP tool call:

```bash
curl -s -X POST http://localhost:8000/api/mcp/normalize_claim \
  -H "Content-Type: application/json" \
  -d '{"value": "embedding drift hurts retrieval quality"}' \
  | python3 -m json.tool
```

The interactive API docs are available at `http://localhost:8000/docs`.

## 13. Start the frontend and validate full-stack

In a second terminal, from `agentic-ai-system-rag-mcp-fe`:

```bash
cd ../agentic-ai-system-rag-mcp-fe
npm run dev
```

Expected result: Vite starts on `http://localhost:5173`. Open that URL in a browser.

**Golden-path checklist in the UI:**

| Page | Action | Expected |
|---|---|---|
| RAG Query | Submit "What is embedding drift?" | Answer with citations renders |
| Agent | Submit "Build a reading plan for MCP integration" | Status badge = answered, tool trace visible |
| Workflow | Submit "What is embedding drift?" | Final answer + audit log renders |
| Workflow | Submit "Guarantee this is always correct." | Yellow human-review panel appears; click Approve → final answer |
| Memory | Submit query "embedding drift", user "demo-user" | Long-term and entity memory sections populate |
| MCP Tools | Normalize Claim tab, submit a claim | JSON payload renders |
| Evaluate | Click "Run Evaluation" | Five score cards + results table render |

The Vite proxy rewrites all `/api/*` requests to `http://localhost:8000`, so no CORS issues
should appear in the browser console.

## 14. Shut down or reset

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
- If `agent_memory` table is missing after step 11, confirm `GENAI_LAB_PROFILE=prod` was set — the dev profile uses `InMemoryStore` and never touches Postgres.
- If the API returns 500, check the uvicorn terminal for the Python traceback. The most common causes are a missing `.env` file or Postgres/Ollama not yet reachable.
- If the frontend shows a network error or blank panel, confirm the API is running on port 8000 and check the browser DevTools network tab — all `/api/*` requests should proxy to `localhost:8000`.
- If the Workflow page's human-review panel does not appear after submitting a review-worthy prompt, confirm the `audit_log` and `interrupted` fields in the API response (`curl -X POST http://localhost:8000/api/workflow/start ...`).
