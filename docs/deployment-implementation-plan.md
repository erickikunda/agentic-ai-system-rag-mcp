# Deployment Implementation Plan — genai-lab (RAG + MCP research assistant)

Goal: deploy the RAG research assistant to AWS or GCP so it serves concurrent users with
low latency, ingests documents at scale, and keeps per-user memory correct across replicas.

**Architecture shape:** three independently scaled tiers — a stateless query API, a batch
ingestion job, and stateless MCP replicas. Do **not** ship them as one `app` service.

```
                ┌──────────────────────────────────────────────┐
 Users ─▶ API GW/LB ─▶ Query service (stateless, autoscaled on RPS/latency)
                │        = FastAPI wrapping RAG engine / LangGraph workflow / agent
                │            │              │                     │
                │     Managed LLM     Vector store          Memory store
                │     + embeddings    (pgvector/Pinecone)   (Postgres+pgvector) ◀ externalized!
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

## Phase 1 — Externalize long-term memory (DO THIS FIRST)

The critical scaling blocker. `LongTermMemoryStore` defaults to LangGraph `InMemoryStore()`
(`memory/store.py`) — lost on restart, not shared across replicas. With >1 replica, a user's
memory depends on which pod they hit.

- [ ] Implement a Postgres+pgvector-backed `LongTermMemoryStore` behind the existing boundary (the code comment already names PGVector as the intended backend).
- [ ] Replace the lexical `_lexical_score` search with embedding-based semantic search (reuse the embedding adapter from Phase 4).
- [ ] Preserve the `MemoryNamespace` scoping (`memory/user/<id>`, `memory/agent/<id>`) so per-user isolation holds across instances.
- [ ] Verify `MemoryManager.create()` wires the persistent store, not `InMemoryStore()`.
- [ ] Only after this lands should you run more than one query-service replica.

## Phase 2 — Add an HTTP API for the query tier

- [ ] Add a FastAPI/uvicorn app exposing the RAG engine, agent, and workflow as endpoints (this is the query-tier container entrypoint — currently everything is CLI only).
- [ ] Keep request handling stateless; all durable state lives in the memory store + vector store.
- [ ] Add `/healthz` / `/readyz` for load-balancer probes; structured request logging.
- [ ] Pass `user_id` per request so memory scoping works.

## Phase 3 — Split ingestion into its own deployable/job

- [ ] Make ingestion a separate entrypoint/job (not part of the query service).
- [ ] Read source documents from object storage instead of `ingestion_input_dir` (local dir).
- [ ] Trigger on upload (event) or on a schedule; write chunks/embeddings to the managed vector store.
- [ ] Reuse the LlamaIndex ingestion cache where possible; make re-runs idempotent.

## Phase 4 — Managed LLM + vector store

- [ ] Implement `ChatModelFactory` / `EmbeddingModelFactory` adapters for the cloud LLM (Bedrock-Claude on AWS, Vertex on GCP); select via `llm_provider`. Retire Ollama for prod.
- [ ] Point retrieval at managed Postgres+pgvector (or Pinecone) via `vector_store_provider`.
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

## Cloud service mapping

| Concern | AWS | GCP |
|---|---|---|
| Query API (online) | ECS Fargate / EKS + ALB | Cloud Run / GKE |
| Ingestion (batch) | ECS scheduled task / Batch / Lambda | Cloud Run Jobs / Batch |
| Doc landing + trigger | S3 → EventBridge/SQS | GCS → Eventarc/Pub/Sub |
| MCP server (stateless) | Fargate / EKS | Cloud Run / GKE |
| Vector store + memory | RDS/Aurora pgvector (or Pinecone) | Cloud SQL pgvector (or Vertex / Pinecone) |
| LLM + embeddings | Bedrock | Vertex AI |
| Secrets | Secrets Manager | Secret Manager |

## Explicit non-goals / pitfalls

- Do **not** run >1 query replica until memory is externalized (Phase 1) — per-user memory will be inconsistent.
- Do **not** bundle ingestion into the always-on query service; it is a job, scaled separately.
- Keep the deterministic dry-run paths and evaluation suite green as model-backed components replace them.

## Suggested order of execution

Phase 0 → **Phase 1 (memory externalization — gating)** → Phase 2 (API) → Phase 4 (managed LLM/vectors) → Phase 3 (ingestion job) → Phase 5 (MCP replicas) → Phase 6 (ops).

---

## Relationship to the sibling project (complaint-lab)

Same scaffolding (ports/adapters, pydantic `Settings`, deterministic dev path, Spring MCP),
but a different workload: **complaint-lab is event-driven (queue + workers)**; this one is
**request/response + batch ingestion**. complaint-lab's hard part was an idempotent stateful
case service; here the MCP server is trivially stateless, but **in-process per-user memory is
the thing that must be fixed first**.
