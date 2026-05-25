"""LangGraph stateful multi-agent workflow for research answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from genai_lab.agents.executor import ToolUsingResearchAgent
from genai_lab.config.settings import Settings

from .state import ResearchWorkflowState


@dataclass(frozen=True)
class WorkflowRuntime:
    """Runtime dependencies and graph limits."""

    settings: Settings
    max_cycles: int


def plan_task(
    state: ResearchWorkflowState,
    runtime: Runtime[WorkflowRuntime],
) -> ResearchWorkflowState:
    """Supervisor/planner node that chooses the next research strategy."""

    cycle_count = state.get("cycle_count", 0) + 1
    task = state["task"]
    plan = (
        "Use corpus-grounded tools, collect evidence and risk signals, synthesize with citations, "
        "then verify confidence before finalizing."
    )
    return {
        "cycle_count": cycle_count,
        "max_cycles": runtime.context.max_cycles,
        "plan": plan,
        "audit_log": [f"planner: cycle={cycle_count} task={task!r}"],
    }


def retrieve_evidence(
    state: ResearchWorkflowState,
    runtime: Runtime[WorkflowRuntime],
) -> ResearchWorkflowState:
    """Evidence worker node backed by the Module 3 tool-using agent."""

    agent = ToolUsingResearchAgent(settings=runtime.context.settings)
    run = agent.run(state["task"])
    citations = run.response.citations
    return {
        "evidence": [run.response.answer],
        "citations": citations,
        "risk_flags": run.response.warnings,
        "audit_log": [f"evidence_worker: tools={','.join(run.response.tools_used)}"],
    }


def assess_risk(
    state: ResearchWorkflowState,
    runtime: Runtime[WorkflowRuntime],
) -> ResearchWorkflowState:
    """Verifier worker that scans task shape for review-worthy risk."""

    task = state["task"].lower()
    flags: list[str] = []
    if any(term in task for term in ("prove", "guarantee", "always", "never")):
        flags.append("absolute_claim")
    if any(term in task for term in ("publish", "external", "high-stakes", "production")):
        flags.append("human_review_recommended")
    return {"risk_flags": flags, "audit_log": [f"risk_worker: flags={','.join(flags) or 'none'}"]}


def synthesize_answer(
    state: ResearchWorkflowState,
    runtime: Runtime[WorkflowRuntime],
) -> ResearchWorkflowState:
    """Synthesis node that merges parallel worker outputs into a draft."""

    evidence = state.get("evidence", [])
    citations = sorted(set(state.get("citations", [])))
    if not evidence:
        draft = "No evidence was retrieved."
    else:
        draft = " ".join(evidence)
    if citations:
        draft = f"{draft}\n\nSources: {', '.join(citations)}"
    return {"draft_answer": draft, "audit_log": ["synthesizer: draft_created"]}


def verify_answer(
    state: ResearchWorkflowState,
    runtime: Runtime[WorkflowRuntime],
) -> ResearchWorkflowState:
    """Verifier/supervisor node that decides whether to finalize, retry, or interrupt."""

    citations = state.get("citations", [])
    risk_flags = state.get("risk_flags", [])
    confidence = 0.85 if citations else 0.35
    if risk_flags:
        confidence -= 0.2
    needs_review = bool(risk_flags) or confidence < 0.6
    return {
        "confidence": max(0.0, confidence),
        "needs_human_review": needs_review,
        "audit_log": [f"verifier: confidence={max(0.0, confidence):.2f} review={needs_review}"],
    }


def human_review(
    state: ResearchWorkflowState,
    runtime: Runtime[WorkflowRuntime],
) -> ResearchWorkflowState:
    """Human-in-the-loop interrupt node for low-confidence or risky drafts."""

    decision = interrupt(
        {
            "task": state["task"],
            "draft_answer": state.get("draft_answer", ""),
            "risk_flags": state.get("risk_flags", []),
            "confidence": state.get("confidence", 0.0),
            "request": "Approve, revise, or reject the draft.",
        }
    )
    normalized = str(decision).lower()
    if normalized not in {"approved", "revise", "rejected"}:
        normalized = "revise"
    return {
        "human_decision": normalized,  # type: ignore[typeddict-item]
        "audit_log": [f"human_review: decision={normalized}"],
    }


def finalize_answer(
    state: ResearchWorkflowState,
    runtime: Runtime[WorkflowRuntime],
) -> ResearchWorkflowState:
    """Finalize the workflow output."""

    decision = state.get("human_decision")
    draft = state.get("draft_answer", "No answer was produced.")
    if decision == "rejected":
        final = "Human review rejected the draft. No final answer was produced."
    elif decision == "revise":
        final = f"Needs revision before use:\n\n{draft}"
    else:
        final = draft
    return {"final_answer": final, "audit_log": ["finalizer: complete"]}


def route_after_plan(state: ResearchWorkflowState) -> list[str] | str:
    """Fan out to workers or stop if cycle budget is exhausted."""

    if state.get("cycle_count", 0) > state.get("max_cycles", 2):
        return "finalize"
    return ["retrieve_evidence", "assess_risk"]


def route_after_verify(state: ResearchWorkflowState) -> Literal["human_review", "plan_task", "finalize"]:
    """Conditional edge for review, retry, or finalization."""

    if state.get("needs_human_review", False):
        return "human_review"
    if not state.get("citations") and state.get("cycle_count", 0) < 2:
        return "plan_task"
    return "finalize"


def build_research_workflow(settings: Settings):
    """Compile the LangGraph workflow with an in-memory checkpointer."""

    builder = StateGraph(ResearchWorkflowState, context_schema=WorkflowRuntime)
    builder.add_node("plan_task", plan_task)
    builder.add_node("retrieve_evidence", retrieve_evidence)
    builder.add_node("assess_risk", assess_risk)
    builder.add_node("synthesize_answer", synthesize_answer)
    builder.add_node("verify_answer", verify_answer)
    builder.add_node("human_review", human_review)
    builder.add_node("finalize", finalize_answer)

    builder.add_edge(START, "plan_task")
    builder.add_conditional_edges(
        "plan_task",
        route_after_plan,
        {
            "retrieve_evidence": "retrieve_evidence",
            "assess_risk": "assess_risk",
            "finalize": "finalize",
        },
    )
    builder.add_edge("retrieve_evidence", "synthesize_answer")
    builder.add_edge("assess_risk", "synthesize_answer")
    builder.add_edge("synthesize_answer", "verify_answer")
    builder.add_conditional_edges(
        "verify_answer",
        route_after_verify,
        {
            "human_review": "human_review",
            "plan_task": "plan_task",
            "finalize": "finalize",
        },
    )
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=InMemorySaver())


def build_workflow_context(settings: Settings) -> WorkflowRuntime:
    """Create the typed runtime context passed to LangGraph invocations."""

    return WorkflowRuntime(settings=settings, max_cycles=settings.workflow_max_cycles)
