"""Structured agent inputs, outputs, and trace records."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class AgentStatus(StrEnum):
    """Final status for a tool-using agent run."""

    ANSWERED = "answered"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class ResearchAgentResponse(BaseModel):
    """Structured final response returned by the agent."""

    status: Literal["answered", "needs_review", "failed"] = Field(
        description="Whether the agent produced an answer, needs human review, or failed."
    )
    answer: str = Field(description="Grounded answer written for the user.")
    citations: list[str] = Field(default_factory=list, description="Source names used as evidence.")
    tools_used: list[str] = Field(default_factory=list, description="Tools called during the run.")
    warnings: list[str] = Field(default_factory=list, description="Operational or safety warnings.")


class RagSearchInput(BaseModel):
    """Input schema for the RAG search tool."""

    question: str = Field(description="Research question to answer from the indexed corpus.")
    strategy: Literal["similarity", "mmr", "hybrid"] = Field(
        default="hybrid",
        description="Retrieval strategy. Use hybrid for mixed lexical/semantic questions.",
    )


class CompareSourcesInput(BaseModel):
    """Input schema for comparing known source files."""

    left_source: str = Field(description="First source file name or path fragment.")
    right_source: str = Field(description="Second source file name or path fragment.")
    focus: str = Field(default="main claims", description="Comparison focus.")


class CitationLookupInput(BaseModel):
    """Input schema for citation/source lookup."""

    source_name: str = Field(description="Source file name or path fragment to inspect.")


@dataclass(frozen=True)
class ToolTrace:
    """Trace record for a tool call."""

    tool_name: str
    attempt: int
    ok: bool
    input_summary: str
    output_summary: str


@dataclass(frozen=True)
class AgentRun:
    """Internal agent run with trace and structured response."""

    response: ResearchAgentResponse
    traces: tuple[ToolTrace, ...] = field(default_factory=tuple)

