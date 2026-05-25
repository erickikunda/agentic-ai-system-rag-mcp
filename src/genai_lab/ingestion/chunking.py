"""Chunking policy for research documents."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingProfile:
    """Token-oriented chunking settings for LlamaIndex node parsing."""

    chunk_size_tokens: int
    chunk_overlap_tokens: int

    def validate(self) -> None:
        """Fail fast on chunking settings that would create poor or invalid nodes."""

        if self.chunk_size_tokens < 128:
            raise ValueError("chunk_size_tokens must be at least 128 for research prose.")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens cannot be negative.")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens.")

    @property
    def overlap_ratio(self) -> float:
        """Return the overlap fraction for reporting and sanity checks."""

        return self.chunk_overlap_tokens / self.chunk_size_tokens


def default_chunking_profile() -> ChunkingProfile:
    """Default compromise: preserve argument flow without flooding context windows."""

    return ChunkingProfile(chunk_size_tokens=512, chunk_overlap_tokens=80)

