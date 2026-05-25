"""Metadata enrichment for source documents before indexing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class SourceMetadata:
    """Normalized metadata attached to each loaded document."""

    source_path: str
    source_name: str
    source_type: str
    corpus: str
    ingested_at: str

    def as_dict(self) -> dict[str, str]:
        """Return metadata in the flat shape expected by LlamaIndex documents."""

        return {
            "source_path": self.source_path,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "corpus": self.corpus,
            "ingested_at": self.ingested_at,
        }


def classify_source_type(path: Path) -> str:
    """Classify a source file into a stable, filterable type."""

    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".pdf":
        return "paper_pdf"
    if suffix in {".txt", ".text"}:
        return "text"
    return suffix.removeprefix(".") or "unknown"


def build_source_metadata(path: str | Path, *, corpus: str = "research") -> dict[str, str]:
    """Build deterministic source metadata, except for the ingestion timestamp."""

    source_path = Path(path)
    metadata = SourceMetadata(
        source_path=str(source_path),
        source_name=source_path.name,
        source_type=classify_source_type(source_path),
        corpus=corpus,
        ingested_at=datetime.now(UTC).isoformat(),
    )
    return metadata.as_dict()

