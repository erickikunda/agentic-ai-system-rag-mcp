"""Vector store port for dev/prod swapping.

The implementation will hide PGVector/Pinecone details behind collection-level
operations so retrieval code can focus on search semantics rather than storage.
"""

from typing import Protocol


class VectorStoreFactory(Protocol):
    """Creates vector-store clients for ingestion, RAG, and memory indexes."""

    def create_document_store(self) -> object:
        """Return the vector store used for source document chunks."""

    def create_memory_store(self) -> object:
        """Return the vector store used for long-term memory."""

