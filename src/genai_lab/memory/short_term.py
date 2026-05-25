"""Short-term conversation memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from genai_lab.memory.schema import ConversationTurn


def estimate_tokens(text: str) -> int:
    """Approximate tokens for budget decisions."""

    return max(1, int(len(text.split()) * 1.25))


@dataclass
class ConversationBuffer:
    """Bounded short-term memory for a single thread."""

    max_turns: int
    summary_trigger_tokens: int
    turns: list[ConversationTurn] = field(default_factory=list)
    rolling_summary: str | None = None

    def add_turn(self, user: str, assistant: str) -> None:
        """Append a turn and summarize if the buffer is too large."""

        self.turns.append(ConversationTurn(user=user, assistant=assistant))
        self._trim_by_turns()
        if estimate_tokens(self.render()) > self.summary_trigger_tokens:
            self.summarize_oldest()

    def render(self) -> str:
        """Return prompt-ready short-term context."""

        pieces: list[str] = []
        if self.rolling_summary:
            pieces.append(f"Conversation summary:\n{self.rolling_summary}")
        pieces.extend(turn.text for turn in self.turns)
        return "\n\n".join(pieces)

    def summarize_oldest(self) -> None:
        """Compress older turns into a deterministic rolling summary."""

        if len(self.turns) <= 1:
            return
        oldest = self.turns[:-1]
        bullet_points = [
            f"- User asked about {turn.user[:90].strip()}; assistant answered {turn.assistant[:90].strip()}."
            for turn in oldest
        ]
        new_summary = "\n".join(bullet_points)
        self.rolling_summary = (
            f"{self.rolling_summary}\n{new_summary}" if self.rolling_summary else new_summary
        )
        self.turns = self.turns[-1:]

    def _trim_by_turns(self) -> None:
        if len(self.turns) <= self.max_turns:
            return
        overflow = self.turns[: -self.max_turns]
        self.turns = self.turns[-self.max_turns :]
        overflow_summary = "\n".join(
            f"- Earlier: user asked {turn.user[:80].strip()}." for turn in overflow
        )
        self.rolling_summary = (
            f"{self.rolling_summary}\n{overflow_summary}"
            if self.rolling_summary
            else overflow_summary
        )

