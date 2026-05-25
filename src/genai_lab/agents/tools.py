"""LangChain tools for the research assistant agent."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from genai_lab.config.settings import Settings
from genai_lab.mcp.client import build_mcp_provider
from genai_lab.mcp.langchain_tools import build_mcp_langchain_tools
from genai_lab.rag.engine import RagQueryEngine, build_rag_config
from genai_lab.rag.schema import RetrievalStrategy

from .schema import CitationLookupInput, CompareSourcesInput, RagSearchInput


def build_research_tools(settings: Settings) -> list[BaseTool]:
    """Create LangChain tools with explicit schemas and production-minded descriptions."""

    @tool(
        "rag_search",
        args_schema=RagSearchInput,
        description=(
            "Search the research corpus and return a grounded answer with citations. "
            "Use this before answering factual questions about the corpus. "
            "Prefer strategy='hybrid' unless the user asks for diversity, then use 'mmr'."
        ),
    )
    def rag_search(question: str, strategy: str = "hybrid") -> str:
        config = replace(
            build_rag_config(settings),
            strategy=RetrievalStrategy(strategy),
            min_score=0.0,
        )
        answer = RagQueryEngine(settings=settings, config=config).query(question)
        return json.dumps(
            {
                "answer": answer.answer,
                "citations": [citation.source_name for citation in answer.citations],
                "warnings": list(answer.warnings),
            },
            sort_keys=True,
        )

    @tool(
        "compare_sources",
        args_schema=CompareSourcesInput,
        description=(
            "Compare two source files already present in the local research corpus. "
            "Use when the task asks for contrast, agreement, disagreement, or trade-offs."
        ),
    )
    def compare_sources(left_source: str, right_source: str, focus: str = "main claims") -> str:
        left = _find_source(settings, left_source)
        right = _find_source(settings, right_source)
        left_text = _read_excerpt(left)
        right_text = _read_excerpt(right)
        return json.dumps(
            {
                "focus": focus,
                "left_source": left.name,
                "right_source": right.name,
                "comparison": (
                    f"{left.name} emphasizes {_first_content_sentence(left_text)} "
                    f"{right.name} emphasizes {_first_content_sentence(right_text)}"
                ),
            },
            sort_keys=True,
        )

    @tool(
        "citation_lookup",
        args_schema=CitationLookupInput,
        description=(
            "Inspect metadata and a short excerpt for a specific source file. "
            "Use when the agent needs to verify citation provenance."
        ),
    )
    def citation_lookup(source_name: str) -> str:
        source = _find_source(settings, source_name)
        text = _read_excerpt(source)
        return json.dumps(
            {
                "source_name": source.name,
                "source_path": str(source),
                "excerpt": _first_content_sentence(text),
            },
            sort_keys=True,
        )

    mcp_tools = build_mcp_langchain_tools(build_mcp_provider(settings))
    return [rag_search, compare_sources, citation_lookup, *mcp_tools]


def _find_source(settings: Settings, query: str) -> Path:
    input_dir = Path(settings.ingestion_input_dir)
    candidates = sorted(input_dir.rglob("*"))
    for candidate in candidates:
        if candidate.is_file() and query.lower() in str(candidate).lower():
            return candidate
    raise FileNotFoundError(f"No source matched {query!r} under {input_dir}")


def _read_excerpt(path: Path, *, max_chars: int = 1200) -> str:
    return path.read_text(encoding="utf-8")[:max_chars]


def _first_content_sentence(text: str) -> str:
    normalized = " ".join(line.strip("# ").strip() for line in text.splitlines() if line.strip())
    for delimiter in (". ", "? ", "! "):
        if delimiter in normalized:
            return normalized.split(delimiter, 1)[0].strip() + delimiter.strip()
    return normalized[:220].strip()
