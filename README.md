# genai-lab

Generative AI Engineering Learning Lab for building a production-realistic,
agentic AI system in Python with RAG, agents, memory, orchestration, and MCP.

Current phase: **Module 7 - Observability & Evaluation**.

Start with the cumulative study document:

- `docs/study.html`

## Module 1 ingestion

Dry-run the LlamaIndex ingestion pipeline without embeddings or PGVector writes:

```bash
.venv/bin/python -m genai_lab.ingestion.cli --dry-run
```

Run real indexing after Postgres/PGVector and Ollama are available:

```bash
docker compose up postgres ollama
ollama pull nomic-embed-text
.venv/bin/python -m genai_lab.ingestion.cli
```

## Module 2 RAG

Ask the local dry-run RAG engine without model or vector-store services:

```bash
.venv/bin/python -m genai_lab.rag.cli "What is embedding drift?" --strategy hybrid --transform step_back
```

Use the PGVector-backed retrieval path after Module 1 full indexing has run:

```bash
.venv/bin/python -m genai_lab.rag.cli "What is embedding drift?" --use-vector-store
```

## Module 3 agent

Run the local tool-using research agent:

```bash
.venv/bin/python -m genai_lab.agents.cli "What is embedding drift?"
```

## Module 4 workflow

Run the LangGraph research workflow:

```bash
.venv/bin/python -m genai_lab.workflows.cli "What is embedding drift?"
```

## Module 5 memory

Run the local memory demo:

```bash
.venv/bin/python -m genai_lab.memory.cli "embedding drift preferences"
```

## Module 6 MCP

Run Python MCP fallback tools:

```bash
.venv/bin/python -m genai_lab.mcp.cli normalize_claim "embedding drift hurts retrieval quality"
```

Test the Spring Boot MCP server:

```bash
cd mcp-server
mvn test
```

## Module 7 observability and evaluation

Run the local RAG evaluation suite:

```bash
.venv/bin/python -m genai_lab.evaluation.cli
```

Write a local JSONL trace while running RAG or the agent:

```bash
.venv/bin/python -m genai_lab.rag.cli "What is embedding drift?" --trace
.venv/bin/python -m genai_lab.agents.cli "Build a reading plan for MCP integration" --trace
```
