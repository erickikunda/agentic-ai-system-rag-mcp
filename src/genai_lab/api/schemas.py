"""FastAPI request schemas for the genai-lab API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RagRequest(BaseModel):
    question: str
    strategy: str = "similarity"
    transform: str = "none"
    use_vector_store: bool = False


class AgentRequest(BaseModel):
    task: str


class WorkflowStartRequest(BaseModel):
    task: str
    thread_id: str | None = None


class WorkflowResumeRequest(BaseModel):
    thread_id: str
    decision: str = Field(description="approved | revise | rejected")


class MemoryQuery(BaseModel):
    query: str
    user_id: str = "demo-user"


class McpRequest(BaseModel):
    value: str
    level: str = "senior CS student"


class EvaluateRequest(BaseModel):
    dataset_path: str | None = None
