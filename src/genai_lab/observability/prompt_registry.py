"""Small prompt registry with explicit versions.

LangSmith should own production prompt lifecycle once credentials are enabled.
This local registry gives the lab stable prompt identifiers and documents what
is sent to model-backed paths before those prompts are pushed to LangSmith.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptVersion:
    """Versioned prompt template metadata."""

    name: str
    version: str
    template: str
    owner: str
    changelog: str


class PromptRegistry:
    """In-process prompt registry used by local tests and examples."""

    def __init__(self, prompts: dict[str, PromptVersion] | None = None) -> None:
        self._prompts = prompts or _default_prompts()

    def get(self, prompt_id: str) -> PromptVersion:
        """Return a prompt by id or raise a useful error."""

        try:
            return self._prompts[prompt_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._prompts))
            raise KeyError(f"Unknown prompt id {prompt_id!r}. Known prompts: {known}") from exc

    def render(self, prompt_id: str, **values: str) -> str:
        """Render a prompt with strict placeholders."""

        prompt = self.get(prompt_id)
        return prompt.template.format(**values)

    def list_versions(self) -> tuple[PromptVersion, ...]:
        """Return registered prompts in stable order."""

        return tuple(self._prompts[key] for key in sorted(self._prompts))


def _default_prompts() -> dict[str, PromptVersion]:
    return {
        "rag-answer-v1": PromptVersion(
            name="rag-answer",
            version="v1",
            owner="genai-lab",
            changelog="Grounded answer prompt with citation requirement and document-injection boundary.",
            template=(
                "Answer the research question using only the trusted context.\n"
                "Question: {question}\n"
                "Trusted context:\n{context}\n"
                "Return a concise answer with citation ids for every factual claim."
            ),
        ),
        "agent-supervisor-v1": PromptVersion(
            name="agent-supervisor",
            version="v1",
            owner="genai-lab",
            changelog="Routes research tasks across retrieval, comparison, MCP tools, and human review.",
            template=(
                "You supervise a research assistant. Choose the smallest reliable tool sequence for: "
                "{task}. Prefer retrieval for evidence questions and MCP tools for external domain actions."
            ),
        ),
    }
