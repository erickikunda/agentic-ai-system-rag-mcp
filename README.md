# GenAI Engineering Learning Lab

A production-realistic, agentic AI Research Assistant built in Python. This project serves as a learning lab for building robust AI systems, covering Retrieval-Augmented Generation (RAG), agents, memory, orchestration, and the Model Context Protocol (MCP).

**Domain**: Research Assistant for technical papers and engineering notes. This domain naturally exercises ingestion, citation quality, semantic comparison, source metadata, and multi-agent workflows.

For detailed design decisions and architecture notes, please read the cumulative study documents:
- `docs/study.html`
- `docs/full-stack-validation.md`

## Architecture Overview
The system is built on best-in-class frameworks to make important operational seams visible:
- **LlamaIndex**: Owns the document lifecycle (ingestion pipeline, parsing, transformations, embeddings, and vector indexing).
- **LangChain**: Owns the model boundary, tool-calling primitives, and ReAct agent patterns.
- **LangGraph**: Owns long-running control flow, stateful routing, and human-in-the-loop (HITL) checkpoints.
- **PGVector / Pinecone**: Vector storage options for local development and production.
- **Spring Boot MCP**: Models an external domain-specific tool registry via the Model Context Protocol.

## Prerequisites & CI

The project uses `uv` for Python dependency management.
The GitHub Actions workflow in `.github/workflows/ci.yml` automatically validates Python tests, Spring Boot MCP Maven tests, and Docker Compose topology.

---

## Modules & Execution

The lab is broken down into 7 progressive modules.

### Module 1: Document Ingestion Pipeline
Handles chunking, metadata enrichment, embedding, and vector insertion.

*Dry-run the LlamaIndex ingestion pipeline (no embeddings or PGVector writes):*
```bash
.venv/bin/python -m genai_lab.ingestion.cli --dry-run
```

*Run real indexing (requires Postgres/PGVector and Ollama to be running):*
```bash
docker compose up postgres ollama -d
ollama pull nomic-embed-text
.venv/bin/python -m genai_lab.ingestion.cli
```

### Module 2: RAG Query Engine
Implements retrieval strategies, prompt-injection screening, context budgeting, and answer synthesis with citations.

*Ask the local dry-run RAG engine without model or vector-store services:*
```bash
.venv/bin/python -m genai_lab.rag.cli "What is embedding drift?" --strategy hybrid --transform step_back
```

*Use the PGVector-backed retrieval path (requires full indexing from Module 1):*
```bash
.venv/bin/python -m genai_lab.rag.cli "What is embedding drift?" --use-vector-store
```

### Module 3: Tool-Using Agent
Introduces a LangChain agent with tool binding, structured output, and error handling.

*Run the local tool-using research agent:*
```bash
.venv/bin/python -m genai_lab.agents.cli "What is embedding drift?"
```

### Module 4: Stateful Multi-Agent Workflow
Coordinates planner, retriever, answerer, verifier, and escalation nodes using LangGraph.

*Run the LangGraph research workflow:*
```bash
.venv/bin/python -m genai_lab.workflows.cli "What is embedding drift?"
```

### Module 5: Memory & Persistence
Adds long-term memory distinguishability and provenance.

*Run the local memory demo:*
```bash
.venv/bin/python -m genai_lab.memory.cli "embedding drift preferences"
```

### Module 6: MCP Integration
Integrates an external Spring Boot MCP server for cross-language tool consumption.

*Run Python MCP fallback tools:*
```bash
.venv/bin/python -m genai_lab.mcp.cli normalize_claim "embedding drift hurts retrieval quality"
```

*Test the Spring Boot MCP server:*
```bash
cd mcp-server
mvn test
```

### Module 7: Observability & Evaluation
Implements evaluation traces, loop-aware logging, and cost attribution.

*Run the local RAG evaluation suite:*
```bash
.venv/bin/python -m genai_lab.evaluation.cli
```

*Write a local JSONL trace while running RAG or the agent:*
```bash
.venv/bin/python -m genai_lab.rag.cli "What is embedding drift?" --trace
.venv/bin/python -m genai_lab.agents.cli "Build a reading plan for MCP integration" --trace
```
