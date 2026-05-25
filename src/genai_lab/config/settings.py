"""Environment-aware settings skeleton.

Module 0 intentionally defines configuration shape only. Later modules will
bind these settings to concrete LangChain, LlamaIndex, vector store, and MCP
adapters.
"""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeProfile(StrEnum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class LlmProvider(StrEnum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class VectorStoreProvider(StrEnum):
    PGVECTOR = "pgvector"
    PINECONE = "pinecone"


class Settings(BaseSettings):
    """Validated application settings loaded from env and profile files."""

    model_config = SettingsConfigDict(
        env_prefix="GENAI_LAB_",
        env_nested_delimiter="__",
        env_file=(".env", ".env.dev"),
        extra="ignore",
    )

    profile: RuntimeProfile = RuntimeProfile.DEV

    llm_provider: LlmProvider = LlmProvider.OLLAMA
    chat_model: str = "llama3.1"
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://ollama:11434"

    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    vector_store_provider: VectorStoreProvider = VectorStoreProvider.PGVECTOR
    postgres_dsn: SecretStr = Field(
        default=SecretStr("postgresql://genai_lab:genai_lab@postgres:5432/genai_lab")
    )
    postgres_vector_table: str = "research_chunks"
    embedding_dimension: int = 768
    pinecone_api_key: SecretStr | None = None
    pinecone_index: str = "genai-lab-research"

    ingestion_input_dir: str = "data/research_notes"
    ingestion_cache_dir: str = ".cache/llamaindex/ingestion"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 80
    rag_top_k: int = 5
    rag_rerank_top_n: int = 3
    rag_context_token_budget: int = 1600
    rag_min_score: float = 0.05
    agent_max_iterations: int = 4
    agent_tool_retry_attempts: int = 2
    agent_token_budget: int = 2200

    mcp_server_url: str = "http://mcp-server:8080"
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings after validation."""

    return Settings()
