"""State schema for the research workflow graph."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class ResearchWorkflowState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    task: str
    plan: str
    cycle_count: int
    max_cycles: int
    evidence: Annotated[list[str], operator.add]
    citations: Annotated[list[str], operator.add]
    risk_flags: Annotated[list[str], operator.add]
    draft_answer: str
    final_answer: str
    confidence: float
    needs_human_review: bool
    human_decision: Literal["approved", "revise", "rejected"]
    audit_log: Annotated[list[str], operator.add]
