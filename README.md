# genai-lab

Generative AI Engineering Learning Lab for building a production-realistic,
agentic AI system in Python with RAG, agents, memory, orchestration, and MCP.

Current phase: **Module 1 - Document Ingestion Pipeline**.

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
