"""Local trace records and LangSmith environment wiring.

The local trace writer is intentionally boring JSONL. LangSmith is the primary
production tracing backend for this lab, but a local sink keeps tests and
offline development deterministic.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from genai_lab.config.settings import Settings


SENSITIVE_KEYS = {"api_key", "authorization", "password", "secret", "token"}


@dataclass(frozen=True)
class TraceEvent:
    """A single structured event in a RAG, agent, or workflow trace."""

    run_id: str
    name: str
    kind: str
    status: str
    started_at_ms: int
    duration_ms: int
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class JsonlTraceWriter:
    """Append-only local trace sink for development and CI."""

    def __init__(self, trace_dir: Path | str) -> None:
        self.trace_dir = Path(trace_dir)

    def write(self, event: TraceEvent) -> Path:
        """Write one redacted trace event and return the JSONL file path."""

        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{event.run_id}.jsonl"
        payload = asdict(event)
        payload["inputs"] = redact_mapping(payload["inputs"])
        payload["outputs"] = redact_mapping(payload["outputs"])
        payload["metadata"] = redact_mapping(payload["metadata"])
        with path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(payload, sort_keys=True) + "\n")
        return path


class SpanTimer:
    """Small helper for measuring wall-clock duration around local spans."""

    def __init__(self, *, name: str, kind: str, run_id: str | None = None) -> None:
        self.run_id = run_id or str(uuid4())
        self.name = name
        self.kind = kind
        self.started_at_ms = int(time.time() * 1000)

    def finish(
        self,
        *,
        status: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Create a completed trace event."""

        finished_at_ms = int(time.time() * 1000)
        return TraceEvent(
            run_id=self.run_id,
            name=self.name,
            kind=self.kind,
            status=status,
            started_at_ms=self.started_at_ms,
            duration_ms=max(0, finished_at_ms - self.started_at_ms),
            inputs=inputs or {},
            outputs=outputs or {},
            metadata=metadata or {},
        )


def trace_rag_answer(
    *,
    settings: Settings,
    question: str,
    answer: str,
    citation_count: int,
    warnings: tuple[str, ...],
    writer: JsonlTraceWriter | None = None,
) -> Path:
    """Persist a compact RAG trace event for offline inspection."""

    timer = SpanTimer(name="rag.query", kind="rag")
    event = timer.finish(
        status="ok",
        inputs={"question": question},
        outputs={"answer_preview": answer[:320], "citation_count": citation_count},
        metadata={
            "warnings": list(warnings),
            "prompt_version": settings.prompt_registry_version,
            "provider": settings.observability_provider.value,
        },
    )
    return (writer or JsonlTraceWriter(settings.trace_log_dir)).write(event)


def redact_mapping(value: Any) -> Any:
    """Recursively redact sensitive-looking keys from trace payloads."""

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(sensitive in key_text for sensitive in SENSITIVE_KEYS):
                redacted[str(key)] = "[redacted]"
            else:
                redacted[str(key)] = redact_mapping(item)
        return redacted
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    if isinstance(value, tuple):
        return [redact_mapping(item) for item in value]
    return value


def build_langsmith_environment(settings: Settings) -> dict[str, str]:
    """Return env vars needed for LangSmith tracing without mutating process env."""

    environment = {
        "LANGSMITH_TRACING": "true" if settings.langsmith_tracing else "false",
        "LANGSMITH_PROJECT": settings.langsmith_project,
    }
    if settings.langsmith_api_key is not None:
        environment["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    return environment
