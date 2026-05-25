"""LlamaIndex-backed ingestion pipeline construction and execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from genai_lab.config.settings import LlmProvider, Settings, VectorStoreProvider
from genai_lab.ingestion.chunking import ChunkingProfile
from genai_lab.ingestion.metadata import build_source_metadata


@dataclass(frozen=True)
class IngestionResult:
    """Summary returned by an ingestion run."""

    documents_loaded: int
    nodes_created: int
    vector_store: str
    dry_run: bool


def load_documents(input_dir: Path) -> list[object]:
    """Load documents with LlamaIndex's file reader and attach normalized metadata."""

    from llama_index.core import SimpleDirectoryReader

    reader = SimpleDirectoryReader(
        input_dir=str(input_dir),
        recursive=True,
        required_exts=[".md", ".txt", ".pdf"],
        file_metadata=lambda path: build_source_metadata(path),
    )
    return list(reader.load_data())


def create_embedding_model(settings: Settings) -> object:
    """Create the LlamaIndex embedding transformation for the active provider."""

    if settings.llm_provider == LlmProvider.OLLAMA:
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(
            model_name=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

    if settings.llm_provider == LlmProvider.OPENAI:
        from llama_index.embeddings.openai import OpenAIEmbedding

        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAIEmbedding(model=settings.embedding_model, api_key=api_key)

    raise NotImplementedError(
        f"Embedding provider {settings.llm_provider!s} is not wired for Module 1 yet."
    )


def create_pgvector_store(settings: Settings) -> object:
    """Create a LlamaIndex PGVector store from the configured Postgres DSN."""

    from llama_index.vector_stores.postgres import PGVectorStore

    parsed = urlparse(settings.postgres_dsn.get_secret_value())
    return PGVectorStore.from_params(
        database=parsed.path.removeprefix("/"),
        host=parsed.hostname or "localhost",
        password=parsed.password or "",
        port=parsed.port or 5432,
        user=parsed.username or "postgres",
        table_name=settings.postgres_vector_table,
        embed_dim=settings.embedding_dimension,
    )


def create_ingestion_cache(settings: Settings) -> object:
    """Create or load the persistent LlamaIndex ingestion cache."""

    from llama_index.core.ingestion import IngestionCache

    cache_path = Path(settings.ingestion_cache_dir) / "cache.json"
    if cache_path.exists():
        return IngestionCache.from_persist_path(str(cache_path), collection="research_documents")
    return IngestionCache(collection="research_documents")


def build_pipeline(
    *,
    settings: Settings,
    chunking: ChunkingProfile,
    dry_run: bool,
) -> object:
    """Build a LlamaIndex ingestion pipeline.

    Dry runs parse documents into nodes without embedding or writing vectors.
    Full runs append the embedding transformation and attach PGVector.
    """

    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter

    chunking.validate()
    transformations: list[object] = [
        SentenceSplitter(
            chunk_size=chunking.chunk_size_tokens,
            chunk_overlap=chunking.chunk_overlap_tokens,
        )
    ]

    vector_store = None
    if not dry_run:
        transformations.append(create_embedding_model(settings))
        if settings.vector_store_provider != VectorStoreProvider.PGVECTOR:
            raise NotImplementedError("Module 1 implements PGVector indexing; Pinecone arrives later.")
        vector_store = create_pgvector_store(settings)

    return IngestionPipeline(
        transformations=transformations,
        vector_store=vector_store,
        cache=create_ingestion_cache(settings),
    )


def run_ingestion(
    *,
    settings: Settings,
    input_dir: Path,
    chunking: ChunkingProfile,
    dry_run: bool = False,
) -> IngestionResult:
    """Load, parse, optionally embed, and optionally index source documents."""

    if not input_dir.exists():
        raise FileNotFoundError(f"Ingestion input directory does not exist: {input_dir}")

    documents = load_documents(input_dir)
    pipeline = build_pipeline(settings=settings, chunking=chunking, dry_run=dry_run)
    nodes = pipeline.run(documents=documents)
    cache_dir = Path(settings.ingestion_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline.cache.persist(str(cache_dir / "cache.json"))

    vector_store = "none" if dry_run else settings.vector_store_provider.value
    return IngestionResult(
        documents_loaded=len(documents),
        nodes_created=len(nodes),
        vector_store=vector_store,
        dry_run=dry_run,
    )
