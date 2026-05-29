"""RAG query engine composition."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from llama_index.core.schema import NodeWithScore

from genai_lab.config.settings import Settings
from genai_lab.ingestion.chunking import ChunkingProfile
from genai_lab.rag.query_transform import build_query_variants
from genai_lab.rag.retrieval import (
    LocalCorpusRetriever,
    apply_context_budget,
    create_vector_retriever,
    node_to_context,
    tokenize,
)
from genai_lab.rag.schema import Citation, RagAnswer, RagConfig, RetrievedContext
from genai_lab.rag.security import context_boundary_prompt


def build_rag_config(settings: Settings, *, use_vector_store: bool = False) -> RagConfig:
    """Build a RAG config from application settings."""

    from genai_lab.rag.schema import QueryTransform, RetrievalStrategy

    return RagConfig(
        strategy=RetrievalStrategy.SIMILARITY,
        transform=QueryTransform.NONE,
        top_k=settings.rag_top_k,
        rerank_top_n=settings.rag_rerank_top_n,
        context_token_budget=settings.rag_context_token_budget,
        min_score=settings.rag_min_score,
        use_vector_store=use_vector_store,
    )


def synthesize_deterministic_answer(
    *,
    question: str,
    contexts: list[RetrievedContext],
) -> str:
    """Produce a grounded extractive answer for local verification."""

    if not contexts:
        return "I do not have enough retrieved evidence to answer that question."

    parts: list[str] = []
    for index, context in enumerate(contexts, start=1):
        sentence = best_sentence_for_question(context.text, question)
        parts.append(f"{sentence} [{index}]")
    return " ".join(parts)


def first_sentence(text: str) -> str:
    """Return a compact first sentence for deterministic synthesis."""

    normalized = " ".join(text.split())
    for delimiter in (". ", "? ", "! "):
        if delimiter in normalized:
            return normalized.split(delimiter, 1)[0].strip() + delimiter.strip()
    return normalized[:280].strip()


def best_sentence_for_question(text: str, question: str) -> str:
    """Choose the sentence in a chunk that overlaps the question best."""

    normalized = " ".join(text.split())
    sentences = [
        candidate.strip()
        for candidate in normalized.replace("? ", ". ").replace("! ", ". ").split(". ")
        if candidate.strip()
    ]
    if not sentences:
        return first_sentence(text)

    question_terms = set(tokenize(question))
    if not question_terms:
        return sentences[0]

    best = max(
        sentences,
        key=lambda sentence: len(question_terms & set(tokenize(sentence))) / max(1, len(question_terms)),
    )
    return best if best.endswith((".", "?", "!")) else f"{best}."


def build_citations(contexts: list[RetrievedContext]) -> tuple[Citation, ...]:
    """Create stable citation objects from retrieved contexts."""

    citations: list[Citation] = []
    for index, context in enumerate(contexts, start=1):
        citations.append(
            Citation(
                source_id=index,
                source_name=context.metadata.get("source_name", "unknown"),
                source_path=context.metadata.get("source_path", "unknown"),
                score=round(context.score, 4),
                snippet=first_sentence(context.text),
                flags=context.flags,
            )
        )
    return tuple(citations)


class RagQueryEngine:
    """High-level RAG engine used by CLI, future agents, and tests."""

    def __init__(
        self,
        *,
        settings: Settings,
        config: RagConfig,
        input_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.input_dir = input_dir or Path(settings.ingestion_input_dir)

    def query(self, question: str) -> RagAnswer:
        """Answer a question with citations and retrieval diagnostics."""

        queries = build_query_variants(question, self.config.transform)
        if self.config.use_vector_store:
            return self._query_vector_store(question=question, queries=queries)
        return self._query_local_corpus(question=question, queries=queries)

    def _query_local_corpus(self, *, question: str, queries: tuple[str, ...]) -> RagAnswer:
        chunking = ChunkingProfile(
            chunk_size_tokens=self.settings.chunk_size_tokens,
            chunk_overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        retriever = LocalCorpusRetriever(input_dir=self.input_dir, chunking=chunking)
        contexts = retriever.retrieve(queries, self.config)
        warnings = tuple(
            sorted({flag for context in contexts for flag in context.flags})
        )
        answer = synthesize_deterministic_answer(question=question, contexts=contexts)
        return RagAnswer(
            question=question,
            transformed_queries=queries,
            answer=answer,
            citations=build_citations(contexts),
            strategy=self.config.strategy,
            transform=self.config.transform,
            warnings=warnings,
        )

    def _query_vector_store(self, *, question: str, queries: tuple[str, ...]) -> RagAnswer:
        """Run the LlamaIndex vector query path.

        Calls the retriever directly — no LLM synthesis step — then pipes nodes
        through the same deterministic synthesis used by the local path. This
        avoids any dependency on OpenAI or other hosted LLMs.
        """

        retriever = create_vector_retriever(self.settings, self.config)

        # Fuse results across all query variants.
        fused: dict[str, tuple[object, float]] = {}
        for query in queries:
            nodes: list[NodeWithScore] = retriever.retrieve(
                f"{context_boundary_prompt()}\n\n{query}"
            )
            for rank, node in enumerate(nodes, start=1):
                score = float(node.score or 0.0) + 1 / (rank + 60)
                if node.node_id not in fused or score > fused[node.node_id][1]:
                    fused[node.node_id] = (node, score)

        sorted_nodes = sorted(fused.values(), key=lambda item: item[1], reverse=True)
        contexts = [
            node_to_context(nws.node, score=score)  # type: ignore[union-attr]
            for nws, score in sorted_nodes[: self.config.rerank_top_n]
        ]
        contexts = apply_context_budget(contexts, self.config.context_token_budget)
        warnings = tuple(sorted({flag for ctx in contexts for flag in ctx.flags}))
        answer = synthesize_deterministic_answer(question=question, contexts=contexts)
        return RagAnswer(
            question=question,
            transformed_queries=queries,
            answer=answer,
            citations=build_citations(contexts),
            strategy=self.config.strategy,
            transform=self.config.transform,
            warnings=warnings,
        )

    def with_config(self, **changes: object) -> "RagQueryEngine":
        """Return a copy with updated RAG config values."""

        return RagQueryEngine(
            settings=self.settings,
            config=replace(self.config, **changes),
            input_dir=self.input_dir,
        )
