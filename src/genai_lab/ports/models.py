"""Provider-neutral model ports.

Concrete adapters will wrap LangChain chat models and embedding models. Agent,
workflow, and RAG code should depend on these factories or on LangChain's base
interfaces, not on provider-specific classes such as ChatOpenAI.
"""

from typing import Protocol


class ChatModelFactory(Protocol):
    """Creates a chat model for chains, tools, and graph nodes."""

    def create_chat_model(self, *, streaming: bool = False) -> object:
        """Return a provider-backed LangChain chat model."""


class EmbeddingModelFactory(Protocol):
    """Creates embedding models for ingestion, retrieval, and memory."""

    def create_embedding_model(self) -> object:
        """Return an embedding model compatible with LlamaIndex/LangChain adapters."""

