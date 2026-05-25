"""Lightweight RAGAS-style metrics for local CI.

These metrics are lexical and intentionally conservative. They are not a
replacement for model-judged RAGAS, but they catch regressions in grounding,
retrieval recall, and answer relevance without needing external credentials.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from genai_lab.rag.schema import RagAnswer


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "how",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "why",
    "with",
}


@dataclass(frozen=True)
class RagEvaluationScore:
    """Scores for one RAG answer."""

    faithfulness: float
    answer_relevance: float
    context_recall: float
    citation_coverage: float
    overall: float
    notes: tuple[str, ...]


def score_rag_answer(
    *,
    answer: RagAnswer,
    reference_answer: str,
    required_terms: tuple[str, ...],
    retrieved_contexts: tuple[str, ...] = (),
) -> RagEvaluationScore:
    """Score a RAG answer using deterministic local metrics."""

    context_text = " ".join(retrieved_contexts) or " ".join(
        f"{citation.source_name} {citation.source_path} {citation.snippet}"
        for citation in answer.citations
    )
    faithfulness = _coverage(_content_tokens(answer.answer), _content_tokens(context_text))
    answer_relevance = _coverage(_content_tokens(answer.question), _content_tokens(answer.answer))
    reference_recall = _coverage(_content_tokens(reference_answer), _content_tokens(context_text))
    required_recall = _coverage(_content_tokens(" ".join(required_terms)), _content_tokens(context_text))
    context_recall = round((reference_recall + required_recall) / 2, 4)
    citation_coverage = 1.0 if answer.citations and "[" in answer.answer else 0.0
    overall = round(
        (0.35 * faithfulness)
        + (0.25 * answer_relevance)
        + (0.3 * context_recall)
        + (0.1 * citation_coverage),
        4,
    )
    notes = _notes(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_recall=context_recall,
        citation_coverage=citation_coverage,
    )
    return RagEvaluationScore(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_recall=context_recall,
        citation_coverage=citation_coverage,
        overall=overall,
        notes=notes,
    )


def _content_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOPWORDS and not token.isdigit()
    }


def _coverage(needed: set[str], observed: set[str]) -> float:
    if not needed:
        return 1.0
    return round(len(needed & observed) / len(needed), 4)


def _notes(
    *,
    faithfulness: float,
    answer_relevance: float,
    context_recall: float,
    citation_coverage: float,
) -> tuple[str, ...]:
    notes: list[str] = []
    if faithfulness < 0.6:
        notes.append("low_faithfulness_overlap")
    if answer_relevance < 0.35:
        notes.append("low_answer_relevance")
    if context_recall < 0.5:
        notes.append("low_context_recall")
    if citation_coverage < 1.0:
        notes.append("missing_inline_citation")
    return tuple(notes)
