"""Evaluation runner for the local RAG engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from genai_lab.config.settings import Settings
from genai_lab.rag.engine import RagQueryEngine, build_rag_config

from .metrics import RagEvaluationScore, score_rag_answer


@dataclass(frozen=True)
class EvaluationCase:
    """One RAG evaluation case."""

    case_id: str
    question: str
    reference_answer: str
    required_terms: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation output for one case."""

    case: EvaluationCase
    score: RagEvaluationScore
    answer: str
    citations: tuple[str, ...]


def load_evaluation_cases(path: Path) -> tuple[EvaluationCase, ...]:
    """Load JSONL evaluation cases."""

    cases: list[EvaluationCase] = []
    with path.open(encoding="utf-8") as case_file:
        for line_number, line in enumerate(case_file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            cases.append(
                EvaluationCase(
                    case_id=str(payload.get("case_id", f"case-{line_number}")),
                    question=str(payload["question"]),
                    reference_answer=str(payload["reference_answer"]),
                    required_terms=tuple(str(term) for term in payload.get("required_terms", [])),
                )
            )
    return tuple(cases)


def run_rag_evaluation(
    *,
    settings: Settings,
    cases: tuple[EvaluationCase, ...] | None = None,
    dataset_path: Path | None = None,
) -> tuple[EvaluationResult, ...]:
    """Run local RAG evaluation cases and return deterministic scores."""

    if cases is None:
        path = dataset_path or Path(settings.evaluation_dataset_path)
        cases = load_evaluation_cases(path)

    engine = RagQueryEngine(settings=settings, config=build_rag_config(settings))
    results: list[EvaluationResult] = []
    for case in cases:
        answer = engine.query(case.question)
        retrieved_contexts = _read_cited_contexts(answer)
        score = score_rag_answer(
            answer=answer,
            reference_answer=case.reference_answer,
            required_terms=case.required_terms,
            retrieved_contexts=retrieved_contexts,
        )
        results.append(
            EvaluationResult(
                case=case,
                score=score,
                answer=answer.answer,
                citations=tuple(citation.source_name for citation in answer.citations),
            )
        )
    return tuple(results)


def summarize_results(results: tuple[EvaluationResult, ...]) -> dict[str, float]:
    """Return mean scores for a result set."""

    if not results:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_recall": 0.0,
            "citation_coverage": 0.0,
            "overall": 0.0,
        }
    fields = ("faithfulness", "answer_relevance", "context_recall", "citation_coverage", "overall")
    summary: dict[str, float] = {}
    for field in fields:
        summary[field] = round(
            sum(float(getattr(result.score, field)) for result in results) / len(results),
            4,
        )
    return summary


def result_to_dict(result: EvaluationResult) -> dict[str, object]:
    """Serialize an evaluation result for CLI output."""

    return {
        "case": asdict(result.case),
        "score": asdict(result.score),
        "answer": result.answer,
        "citations": list(result.citations),
    }


def _read_cited_contexts(answer: object) -> tuple[str, ...]:
    """Read cited local files when available so local metrics see real evidence."""

    contexts: list[str] = []
    citations = getattr(answer, "citations", ())
    for citation in citations:
        path = Path(citation.source_path)
        if path.exists() and path.is_file():
            contexts.append(path.read_text(encoding="utf-8"))
        else:
            contexts.append(citation.snippet)
    return tuple(contexts)
