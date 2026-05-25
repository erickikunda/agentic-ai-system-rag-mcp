"""Entity memory extraction."""

from __future__ import annotations

import re


CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9]+(?:\s+|$)){1,4}")
TECH_TERM_RE = re.compile(
    r"\b(?:RAG|MCP|PGVector|Pinecone|LangChain|LangGraph|LlamaIndex|Ollama|OpenAI|Gemini|Claude)\b"
)
ENTITY_STOPWORDS = {
    "Please",
    "Now",
    "User",
    "Assistant",
    "Compare",
    "Explain",
    "Noted",
}


def extract_entities(text: str) -> tuple[str, ...]:
    """Extract simple entity candidates without an LLM dependency."""

    entities: set[str] = set()
    entities.update(match.group(0).strip() for match in TECH_TERM_RE.finditer(text))
    for match in CAPITALIZED_PHRASE_RE.finditer(text):
        candidate = " ".join(match.group(0).split())
        if len(candidate) > 2 and candidate not in ENTITY_STOPWORDS:
            entities.add(candidate)
    return tuple(sorted(entities))
