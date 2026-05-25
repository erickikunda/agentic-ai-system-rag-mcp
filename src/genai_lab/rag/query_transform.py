"""Query transformation helpers.

Module 2 keeps these deterministic so tests and local runs do not require an
LLM. Later modules can swap these templates for model-generated rewrites.
"""

from langchain_core.prompts import ChatPromptTemplate

from genai_lab.rag.schema import QueryTransform


HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a short hypothetical research-note passage that would answer the question. "
            "Do not invent citations or source names.",
        ),
        ("human", "{question}"),
    ]
)

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the question as a broader conceptual retrieval query. "
            "Preserve the domain terms that should appear in source documents.",
        ),
        ("human", "{question}"),
    ]
)


def build_query_variants(question: str, transform: QueryTransform) -> tuple[str, ...]:
    """Return retrieval queries for the requested transform."""

    normalized = " ".join(question.split())
    if transform == QueryTransform.NONE:
        return (normalized,)

    if transform == QueryTransform.HYDE:
        # This is a deterministic HyDE-lite variant for local retrieval. The
        # prompt object above documents the model-backed version used later.
        return (
            normalized,
            f"hypothetical answer passage about {normalized}",
        )

    if transform == QueryTransform.STEP_BACK:
        return (
            normalized,
            f"broader concepts, assumptions, trade-offs, and failure modes for {normalized}",
        )

    raise ValueError(f"Unsupported query transform: {transform}")

