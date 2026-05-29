# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup**
```bash
uv sync                         # install all dependencies (including dev)
uv sync --group dev             # dev extras only (ruff, mypy, pytest)
```

**Testing**
```bash
uv run pytest                   # run all tests
uv run pytest tests/test_rag_engine.py   # run a single test file
uv run pytest -k "test_local_rag"        # run matching tests by name
```

**Lint & type-check**
```bash
uv run ruff check src tests     # lint
uv run ruff format src tests    # format (line-length 100, py312 target)
uv run mypy src                 # strict type checking
```

**Running modules (dry-run, no infrastructure required)**
```bash
.venv/bin/python -m genai_lab.ingestion.cli --dry-run
.venv/bin/python -m genai_lab.rag.cli "What is embedding drift?" --strategy hybrid --transform step_back
.venv/bin/python -m genai_lab.agents.cli "What is embedding drift?"
.venv/bin/python -m genai_lab.workflows.cli "What is embedding drift?"
.venv/bin/python -m genai_lab.memory.cli "embedding drift preferences"
.venv/bin/python -m genai_lab.mcp.cli normalize_claim "embedding drift hurts retrieval quality"
.venv/bin/python -m genai_lab.evaluation.cli
```

**With tracing**
```bash
.venv/bin/python -m genai_lab.rag.cli "..." --trace    # writes JSONL to .cache/genai_lab/traces/
```

**Infrastructure (requires Docker)**
```bash
docker compose up postgres ollama -d   # start PGVector + Ollama
docker compose --profile mcp up -d    # also start Spring Boot MCP server
ollama pull nomic-embed-text           # pull the embedding model
cd mcp-server && mvn test             # Spring Boot MCP tests
```

## Architecture

The system is a 7-module progressive lab. Each module builds on the previous.

**Framework responsibilities are strictly separated:**
- **LlamaIndex** — document lifecycle: ingestion, parsing, chunking, embedding, vector indexing (`src/genai_lab/ingestion/`, `rag/retrieval.py`)
- **LangChain** — model boundary, tool-calling, ReAct agent mechanics (`src/genai_lab/agents/`)
- **LangGraph** — stateful orchestration, conditional routing, human-in-the-loop checkpoints (`src/genai_lab/workflows/`)
- **Spring Boot MCP server** (`mcp-server/`) — external Java service implementing the Model Context Protocol; Python consumes it via `src/genai_lab/mcp/`

**Module map:**
| # | Module | Key files |
|---|--------|-----------|
| 1 | Ingestion | `ingestion/pipeline.py`, `ingestion/chunking.py`, `ingestion/metadata.py` |
| 2 | RAG engine | `rag/engine.py`, `rag/retrieval.py`, `rag/query_transform.py`, `rag/security.py` |
| 3 | Tool-using agent | `agents/executor.py` (deterministic), `agents/langchain_agent.py` (model-backed), `agents/tools.py` |
| 4 | Multi-agent workflow | `workflows/graph.py` (LangGraph), `workflows/state.py` |
| 5 | Memory | `memory/manager.py`, `memory/short_term.py`, `memory/store.py`, `memory/entity.py` |
| 6 | MCP integration | `mcp/client.py`, `mcp/langchain_tools.py` |
| 7 | Observability/eval | `observability/tracing.py`, `observability/prompt_registry.py`, `evaluation/runner.py` |

**Settings** (`src/genai_lab/config/settings.py`) — `pydantic-settings` with `GENAI_LAB_` env prefix. Env files: `.env`, `.env.dev`. The `get_settings()` function is `lru_cache`-wrapped; call it at the call site rather than injecting it at module import time. `RuntimeProfile`, `LlmProvider`, and `VectorStoreProvider` are `StrEnum` switches.

**Ports** (`src/genai_lab/ports/`) — `Protocol`-based interfaces for swapping vector store implementations (PGVector ↔ Pinecone) without touching retrieval logic.

**Local vs. vector-store paths** — Every module has a local/dry-run path that requires no external services and is used by tests. The `--use-vector-store` flag (or `use_vector_store=True` in `RagConfig`) activates the PGVector path.

**LangGraph workflow topology** — `plan_task` → (fan-out) `retrieve_evidence` + `assess_risk` → `synthesize_answer` → `verify_answer` → conditional: `human_review` | retry to `plan_task` | `finalize`. Human-in-the-loop uses `langgraph.types.interrupt`. The graph is compiled with `InMemorySaver` as checkpointer.

**Tracing** — `JsonlTraceWriter` appends redacted JSONL to `.cache/genai_lab/traces/`. `build_langsmith_environment()` returns env vars for LangSmith without mutating `os.environ`. Sensitive keys (`api_key`, `token`, etc.) are automatically redacted from trace payloads.

**Test pattern** — Tests use the local/dry-run code paths and never require infrastructure. `Settings()` can be instantiated without env vars; defaults point to Ollama/PGVector but the dry-run paths don't connect. Pass a custom `input_dir` to `RagQueryEngine` in tests that need controlled document content.
