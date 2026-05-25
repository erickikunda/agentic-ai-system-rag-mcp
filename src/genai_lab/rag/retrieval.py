"""Retrieval implementations for Module 2."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from genai_lab.config.settings import Settings
from genai_lab.ingestion.chunking import ChunkingProfile
from genai_lab.ingestion.pipeline import create_embedding_model, create_pgvector_store, load_documents
from genai_lab.rag.schema import RagConfig, RetrievedContext, RetrievalStrategy
from genai_lab.rag.security import screen_context


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize text for deterministic local retrieval."""

    return tuple(token.lower() for token in TOKEN_RE.findall(text))


def estimate_tokens(text: str) -> int:
    """Cheap token estimate used for context budgeting."""

    return max(1, math.ceil(len(text.split()) * 1.25))


def node_to_context(node: BaseNode, *, score: float) -> RetrievedContext:
    """Convert a LlamaIndex node into the local context schema."""

    text = node.get_content()
    metadata = {str(key): str(value) for key, value in node.metadata.items()}
    return RetrievedContext(text=text, metadata=metadata, score=score, flags=screen_context(text))


class LocalCorpusRetriever:
    """Deterministic retriever over local source files for tests and dry runs."""

    def __init__(self, *, input_dir: Path, chunking: ChunkingProfile) -> None:
        documents = load_documents(input_dir)
        splitter = SentenceSplitter(
            chunk_size=chunking.chunk_size_tokens,
            chunk_overlap=chunking.chunk_overlap_tokens,
        )
        self._nodes = list(splitter.get_nodes_from_documents(documents))
        self._doc_freq = self._document_frequency(self._nodes)

    @staticmethod
    def _document_frequency(nodes: list[BaseNode]) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for node in nodes:
            frequencies.update(set(tokenize(node.get_content())))
        return frequencies

    def _score(self, query: str, node: BaseNode) -> float:
        query_terms = tokenize(query)
        if not query_terms:
            return 0.0

        node_terms = tokenize(node.get_content())
        node_counts = Counter(node_terms)
        score = 0.0
        total_nodes = max(1, len(self._nodes))
        for term in query_terms:
            if term not in node_counts:
                continue
            idf = math.log((1 + total_nodes) / (1 + self._doc_freq[term])) + 1
            score += node_counts[term] * idf

        return score / max(1, len(node_terms))

    def retrieve(self, queries: tuple[str, ...], config: RagConfig) -> list[RetrievedContext]:
        """Retrieve, fuse, rerank, and budget contexts."""

        fused: dict[str, tuple[BaseNode, float]] = {}
        for query in queries:
            ranked = sorted(
                ((node, self._score(query, node)) for node in self._nodes),
                key=lambda item: item[1],
                reverse=True,
            )
            for rank, (node, score) in enumerate(ranked[: config.top_k], start=1):
                if score < config.min_score:
                    continue
                fusion_bonus = 1 / (rank + 60)
                if config.strategy == RetrievalStrategy.HYBRID and query.lower() in node.get_content().lower():
                    fusion_bonus += 0.05
                current = fused.get(node.node_id)
                fused_score = score + fusion_bonus
                if current is None or fused_score > current[1]:
                    fused[node.node_id] = (node, fused_score)

        contexts = [node_to_context(node, score=score) for node, score in fused.values()]
        contexts.sort(key=lambda context: context.score, reverse=True)

        if config.strategy == RetrievalStrategy.MMR:
            contexts = maximal_marginal_relevance(contexts, limit=config.rerank_top_n)
        else:
            contexts = contexts[: config.rerank_top_n]

        return apply_context_budget(contexts, config.context_token_budget)


def maximal_marginal_relevance(
    contexts: list[RetrievedContext],
    *,
    limit: int,
    diversity_lambda: float = 0.7,
) -> list[RetrievedContext]:
    """Select relevant contexts while penalizing near-duplicates."""

    selected: list[RetrievedContext] = []
    remaining = contexts[:]
    while remaining and len(selected) < limit:
        best_index = 0
        best_score = float("-inf")
        for index, candidate in enumerate(remaining):
            candidate_terms = set(tokenize(candidate.text))
            similarity = 0.0
            for existing in selected:
                existing_terms = set(tokenize(existing.text))
                union = candidate_terms | existing_terms
                if union:
                    similarity = max(similarity, len(candidate_terms & existing_terms) / len(union))
            mmr_score = diversity_lambda * candidate.score - (1 - diversity_lambda) * similarity
            if mmr_score > best_score:
                best_index = index
                best_score = mmr_score
        selected.append(remaining.pop(best_index))
    return selected


def apply_context_budget(contexts: list[RetrievedContext], token_budget: int) -> list[RetrievedContext]:
    """Keep highest-ranked contexts within a rough token budget."""

    budgeted: list[RetrievedContext] = []
    used = 0
    for context in contexts:
        cost = estimate_tokens(context.text)
        if budgeted and used + cost > token_budget:
            break
        budgeted.append(context)
        used += cost
    return budgeted


def create_vector_retriever(settings: Settings, config: RagConfig) -> object:
    """Create a LlamaIndex retriever over the configured PGVector store."""

    from llama_index.core import VectorStoreIndex

    vector_store = create_pgvector_store(settings)
    embed_model = create_embedding_model(settings)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    query_mode = {
        RetrievalStrategy.SIMILARITY: VectorStoreQueryMode.DEFAULT,
        RetrievalStrategy.MMR: VectorStoreQueryMode.MMR,
        RetrievalStrategy.HYBRID: VectorStoreQueryMode.HYBRID,
    }[config.strategy]
    return index.as_retriever(similarity_top_k=config.top_k, vector_store_query_mode=query_mode)


def parsed_database_name(settings: Settings) -> str:
    """Expose the configured database name for diagnostics and tests."""

    return urlparse(settings.postgres_dsn.get_secret_value()).path.removeprefix("/")

