# Deployment Implementation Plan — genai-lab (RAG + MCP research assistant)

Goal: deploy the RAG research assistant to AWS or GCP so it serves concurrent users with
low latency, ingests documents at scale, and keeps per-user memory correct across replicas.

**Architecture shape:** three independently scaled tiers — a stateless query API, a batch
ingestion job, and stateless MCP replicas — plus a static frontend. Do **not** ship them
as one `app` service.

```
 Browser ─▶ CDN / static hosting ─▶ React frontend (agentic-ai-system-rag-mcp-fe)
                │
                │ /api/* (reverse proxy)
                ▼
                ┌──────────────────────────────────────────────┐
 Users ─▶ API GW/LB ─▶ Query service (stateless, autoscaled on RPS/latency)
                │        = FastAPI (src/genai_lab/api/) wrapping RAG / LangGraph / agent
                │            │              │                     │
                │     Managed LLM     Vector store          Memory store
                │     + embeddings    (pgvector/Pinecone)   (Postgres+pgvector) ✓ externalized
                │            │
                │     MCP server (stateless Spring, N replicas)
                └──────────────────────────────────────────────┘

 Doc upload ─▶ Object store ─▶ (event / schedule) ─▶ Ingestion JOB ─▶ vector store
```

Key principle: the **query tier scales on RPS/latency**; the **ingestion tier scales on
document volume and runs as a job, not a long-lived service**. Same image, different entrypoints.

---

## Phase 0 — Foundations (infra + CI/CD)

- [ ] Choose cloud and define `dev`/`staging`/`prod` environments (maps to `RuntimeProfile`).
- [ ] IaC (Terraform/CDK): VPC, managed Postgres+pgvector, object store, secret manager, registry, load balancer.
- [ ] Build pipeline publishing the Python image (query + ingestion entrypoints) and the Java MCP image.
- [ ] Move secrets from `.env` into the secret manager; load via existing `SecretStr` fields (`postgres_dsn`, `*_api_key`).

## Phase 1 — Externalize long-term memory ✓ COMPLETE

`PgLongTermMemoryStore` (`memory/store.py`) is implemented behind the `BaseMemoryStore` ABC.
`MemoryManager.create()` selects it when `GENAI_LAB_PROFILE=prod`. Running >1 query replica
is now safe with the prod profile and a shared Postgres instance.

- [x] Implement a Postgres+pgvector-backed store behind the existing `BaseMemoryStore` boundary (`PgLongTermMemoryStore` in `memory/store.py`).
- [ ] Replace the lexical `_lexical_score` search with embedding-based semantic search — **deferred to Phase 4** (table schema and HNSW index already in place via `memory_table_sql()`).
- [x] Preserve `MemoryNamespace` scoping (`memory/user/<id>`) so per-user isolation holds across replicas.
- [x] `MemoryManager.create()` wires `PgLongTermMemoryStore` for `prod` profile, `InMemoryStore` for `dev`/`test`.

## Phase 2 — HTTP API for the query tier ✓ MOSTLY COMPLETE

`src/genai_lab/api/` is a FastAPI application on port 8000 with seven routes under `/api`.
The `genai-lab-api` entry point in `pyproject.toml` runs uvicorn with `--reload`.

- [x] FastAPI/uvicorn app exposing RAG engine, agent, LangGraph workflow, memory, MCP tools, and evaluation (`src/genai_lab/api/app.py` + `routes/`).
- [x] Request handling stateless — workflow state isolated per `thread_id` via `InMemorySaver`; memory and vector state live in external stores.
- [x] `GET /api/health` returns `{"status": "ok", "profile": ...}`.
- [ ] Separate `/healthz` (liveness) and `/readyz` (readiness) probes for load-balancer health checks.
- [ ] Structured request logging (access log with method, path, status, latency).
- [x] `user_id` passed per request; `MemoryNamespace` scoping works correctly across endpoints.

## Phase 3 — Split ingestion into its own deployable/job

- [ ] Make ingestion a separate entrypoint/job (not part of the query service).
- [ ] Read source documents from object storage instead of `ingestion_input_dir` (local dir).
- [ ] Trigger on upload (event) or on a schedule; write chunks/embeddings to the managed vector store.
- [ ] Reuse the LlamaIndex ingestion cache where possible; make re-runs idempotent.

## Phase 4 — Managed LLM + vector store

- [ ] Implement `ChatModelFactory` / `EmbeddingModelFactory` adapters for the cloud LLM (Bedrock-Claude on AWS, Vertex on GCP); select via `llm_provider`. Retire Ollama for prod.
- [ ] Point retrieval at managed Postgres+pgvector (or Pinecone) via `vector_store_provider`.
- [ ] Wire embedding-based semantic search in `PgLongTermMemoryStore.search()` (replace `_lexical_score` with a pgvector cosine similarity query using the Phase 4 embedding adapter).
- [ ] Tune `rag_top_k`, `rag_rerank_top_n`, `rag_context_token_budget` against the managed model; load-test latency.

## Phase 5 — Scale the Spring MCP server (easy win)

`ResearchToolService` tools (`normalize_claim`, `lookup_venue_metadata`, `build_reading_plan`)
are pure functions with no DB/state — trivially horizontally scalable.

- [ ] Containerize (Dockerfile exists) and run N stateless replicas behind a load balancer.
- [ ] Add Actuator health probes; point the query service's `mcp_server_url` at the LB.
- [ ] No persistence layer needed — keep it stateless.

## Phase 6 — Observability, scaling, ops

- [ ] Ship `JsonlTraceWriter` traces to CloudWatch/Cloud Logging (or enable LangSmith via `langsmith_tracing`).
- [ ] Autoscale the query API on RPS/latency; MCP on CPU; run ingestion as on-demand/scheduled jobs.
- [ ] Alarms: query 5xx + p95 latency, model latency/cost, DB connections, ingestion failures.
- [ ] Capacity-plan the LLM endpoint (provisioned throughput) for concurrent users.

## Phase 7 — Frontend deployment

The React frontend (`agentic-ai-system-rag-mcp-fe`) is built and verified locally. For production:

- [ ] `npm run build` produces static assets in `dist/`; deploy to a CDN or static host (S3+CloudFront, GCS+Cloud CDN, Vercel, etc.).
- [ ] Configure the reverse proxy / CDN to route `/api/*` to the query-tier load balancer, replacing the Vite dev-server proxy.
- [ ] Add `GENAI_LAB_CORS_ORIGINS` (or equivalent) to allow the production frontend origin in the FastAPI CORS middleware.

## Cloud service mapping

| Concern | AWS | GCP |
|---|---|---|
| Frontend (static) | S3 + CloudFront | GCS + Cloud CDN / Firebase Hosting |
| Query API (online) | ECS Fargate / EKS + ALB | Cloud Run / GKE |
| Ingestion (batch) | ECS scheduled task / Batch / Lambda | Cloud Run Jobs / Batch |
| Doc landing + trigger | S3 → EventBridge/SQS | GCS → Eventarc/Pub/Sub |
| MCP server (stateless) | Fargate / EKS | Cloud Run / GKE |
| Vector store + memory | RDS/Aurora pgvector (or Pinecone) | Cloud SQL pgvector (or Vertex / Pinecone) |
| LLM + embeddings | Bedrock (Claude) | Vertex AI |
| Secrets | Secrets Manager | Secret Manager |

## Explicit non-goals / pitfalls

- ~~Do **not** run >1 query replica until memory is externalized~~ — Phase 1 is complete; `GENAI_LAB_PROFILE=prod` enables `PgLongTermMemoryStore` and makes replicas safe.
- Do **not** bundle ingestion into the always-on query service; it is a job, scaled separately.
- Keep the deterministic dry-run paths and evaluation suite green as model-backed components replace them.
- The Vite dev-server proxy (`/api → localhost:8000`) is development-only. Production requires a real reverse proxy or CDN routing rule.
- The in-process `InMemorySaver` checkpointer for LangGraph workflows does not survive a query-service restart. For durable workflow state across restarts, swap to a Postgres-backed checkpointer.

## Suggested order of execution

~~Phase 1 (memory — gating)~~ ✓ · ~~Phase 2 (HTTP API)~~ ✓ →
Phase 0 (IaC/CI) → Phase 4 (managed LLM/vectors + memory embedding search) →
Phase 3 (ingestion job) → Phase 5 (MCP replicas) → Phase 7 (frontend deploy) → Phase 6 (ops).

---
