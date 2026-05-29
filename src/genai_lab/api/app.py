"""FastAPI application for the genai-lab research assistant."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from genai_lab.api.routes import agent, evaluate, health, ingest, memory, mcp, rag, workflow

app = FastAPI(title="genai-lab API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")
app.include_router(rag.router, prefix="/api")
app.include_router(agent.router, prefix="/api")
app.include_router(workflow.router, prefix="/api")
app.include_router(memory.router, prefix="/api")
app.include_router(mcp.router, prefix="/api")
app.include_router(evaluate.router, prefix="/api")


def main() -> None:
    uvicorn.run("genai_lab.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
