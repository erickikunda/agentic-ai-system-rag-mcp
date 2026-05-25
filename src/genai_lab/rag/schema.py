"""Shared RAG data structures."""

from dataclasses import dataclass, field
from enum import StrEnum


class RetrievalStrategy(StrEnum):
    """Supported retrieval modes."""

    SIMILARITY = "similarity"
    MMR = "mmr"
    HYBRID = "hybrid"


class QueryTransform(StrEnum):
    """Query transformation modes."""

    NONE = "none"
    HYDE = "hyde"
    STEP_BACK = "step_back"


@dataclass(frozen=True)
class RagConfig:
    """Runtime knobs for retrieval and synthesis."""

    strategy: RetrievalStrategy
    transform: QueryTransform
    top_k: int
    rerank_top_n: int
    context_token_budget: int
    min_score: float
    use_vector_store: bool = False


@dataclass(frozen=True)
class Citation:
    """Citation metadata exposed with an answer."""

    source_id: int
    source_name: str
    source_path: str
    score: float
    snippet: str
    flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RetrievedContext:
    """A retrieved context chunk after scoring, reranking, and screening."""

    text: str
    metadata: dict[str, str]
    score: float
    flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RagAnswer:
    """Grounded answer plus retrieval diagnostics."""

    question: str
    transformed_queries: tuple[str, ...]
    answer: str
    citations: tuple[Citation, ...]
    strategy: RetrievalStrategy
    transform: QueryTransform
    warnings: tuple[str, ...] = field(default_factory=tuple)

