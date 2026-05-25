# genai-lab

Generative AI Engineering Learning Lab for building a production-realistic,
agentic AI system in Python with RAG, agents, memory, orchestration, and MCP.

Current phase: **Module 4 - Stateful Multi-Agent Workflow**.

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
