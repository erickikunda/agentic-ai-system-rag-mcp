"""Memory data structures and taxonomy."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


class MemoryKind(StrEnum):
    """Kinds of long-term memory."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    ENTITY = "entity"
    PROCEDURAL = "procedural"


@dataclass(frozen=True)
class ConversationTurn:
    """A single user/assistant exchange for short-term memory."""

    user: str
    assistant: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def text(self) -> str:
        return f"User: {self.user}\nAssistant: {self.assistant}"


@dataclass(frozen=True)
class MemoryRecord:
    """A persisted memory item."""

    text: str
    kind: MemoryKind
    namespace: tuple[str, ...]
    key: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def as_store_value(self) -> dict[str, object]:
        """Return a LangGraph store-compatible value."""

        return {
            "text": self.text,
            "kind": self.kind.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class MemorySearchResult:
    """Search result returned from long-term memory."""

    key: str
    text: str
    kind: MemoryKind
    score: float
    metadata: dict[str, str]


@dataclass(frozen=True)
class MemorySnapshot:
    """Prompt-ready memory bundle."""

    short_term_context: str
    long_term_context: tuple[MemorySearchResult, ...]
    entity_context: tuple[MemorySearchResult, ...]
    summary: str | None = None

