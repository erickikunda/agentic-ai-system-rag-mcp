"""LangGraph workflow endpoints."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter
from langgraph.types import Command

from genai_lab.api.schemas import WorkflowResumeRequest, WorkflowStartRequest
from genai_lab.config.settings import get_settings
from genai_lab.workflows.graph import build_research_workflow, build_workflow_context

router = APIRouter()

# Single compiled graph shared across requests; checkpointer isolates per thread_id.
_graph = None


def _get_graph() -> object:
    global _graph
    if _graph is None:
        _graph = build_research_workflow(get_settings())
    return _graph


def _serialize_state(result: dict[str, object], thread_id: str) -> dict[str, object]:
    interrupted = "__interrupt__" in result
    interrupt_payload = None
    if interrupted:
        raw = result["__interrupt__"]
        # LangGraph wraps interrupt values in a tuple of Interrupt objects.
        if isinstance(raw, (list, tuple)) and raw:
            item = raw[0]
            interrupt_payload = item.value if hasattr(item, "value") else str(item)
        else:
            interrupt_payload = str(raw)
    return {
        "thread_id": thread_id,
        "interrupted": interrupted,
        "interrupt_payload": interrupt_payload,
        "final_answer": result.get("final_answer"),
        "draft_answer": result.get("draft_answer"),
        "audit_log": list(result.get("audit_log", [])),
        "confidence": result.get("confidence"),
        "risk_flags": list(result.get("risk_flags", [])),
        "needs_human_review": result.get("needs_human_review", False),
    }


@router.post("/workflow/start")
def start_workflow(req: WorkflowStartRequest) -> dict[str, object]:
    settings = get_settings()
    graph = _get_graph()
    thread_id = req.thread_id or f"workflow-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"task": req.task},
        config=config,
        context=build_workflow_context(settings),
    )
    return _serialize_state(result, thread_id)


@router.post("/workflow/resume")
def resume_workflow(req: WorkflowResumeRequest) -> dict[str, object]:
    settings = get_settings()
    graph = _get_graph()
    config = {"configurable": {"thread_id": req.thread_id}}
    result = graph.invoke(
        Command(resume=req.decision),
        config=config,
        context=build_workflow_context(settings),
    )
    return _serialize_state(result, req.thread_id)
