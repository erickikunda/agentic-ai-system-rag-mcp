"""Deterministic tool-using agent executor.

The executor mirrors ReAct mechanics for local verification: decide an action,
call a tool, observe the result, and produce a structured final response. The
model-backed LangChain factory in ``langchain_agent.py`` uses the same tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from genai_lab.config.settings import Settings

from .schema import AgentRun, AgentStatus, ResearchAgentResponse, ToolTrace
from .tools import build_research_tools


@dataclass(frozen=True)
class AgentLimits:
    """Operational limits for agent loops."""

    max_iterations: int
    retry_attempts: int
    token_budget: int


class ToolUsingResearchAgent:
    """Small ReAct-style agent with custom tools and structured output."""

    def __init__(self, *, settings: Settings, tools: list[BaseTool] | None = None) -> None:
        self.settings = settings
        self.tools = {tool.name: tool for tool in (tools or build_research_tools(settings))}
        self.limits = AgentLimits(
            max_iterations=settings.agent_max_iterations,
            retry_attempts=settings.agent_tool_retry_attempts,
            token_budget=settings.agent_token_budget,
        )

    def run(self, user_task: str) -> AgentRun:
        """Run a bounded tool loop and return structured output."""

        traces: list[ToolTrace] = []
        remaining_budget = self.limits.token_budget - _estimate_tokens(user_task)
        if remaining_budget <= 0:
            return _failed("User task exceeds the configured token budget.", traces)

        plan = self._plan(user_task)
        observations: list[dict[str, object]] = []

        for iteration, action in enumerate(plan, start=1):
            if iteration > self.limits.max_iterations:
                return _needs_review("Agent hit max_iterations before completing the task.", traces)

            tool_name, tool_input = action
            result, trace = self._call_tool(tool_name, tool_input)
            traces.extend(trace)
            observations.append({"tool": tool_name, "result": result})
            remaining_budget -= _estimate_tokens(json.dumps(result, sort_keys=True))
            if remaining_budget <= 0:
                return _needs_review("Agent exhausted the token budget while processing tools.", traces)

        return AgentRun(response=self._synthesize(user_task, observations), traces=tuple(traces))

    def _plan(self, user_task: str) -> list[tuple[str, dict[str, str]]]:
        task_lower = user_task.lower()
        if any(word in task_lower for word in ("compare", "contrast", "difference", "versus")):
            return [
                (
                    "compare_sources",
                    {
                        "left_source": "rag_failure_modes",
                        "right_source": "agentic_workflows",
                        "focus": user_task,
                    },
                )
            ]
        return [("rag_search", {"question": user_task, "strategy": "hybrid"})]

    def _call_tool(
        self,
        tool_name: str,
        tool_input: dict[str, str],
    ) -> tuple[dict[str, object], list[ToolTrace]]:
        traces: list[ToolTrace] = []
        tool = self.tools[tool_name]
        last_error = ""
        for attempt in range(1, self.limits.retry_attempts + 1):
            try:
                raw = tool.invoke(tool_input)
                parsed = json.loads(raw) if isinstance(raw, str) else {"result": raw}
                traces.append(
                    ToolTrace(
                        tool_name=tool_name,
                        attempt=attempt,
                        ok=True,
                        input_summary=_summarize(tool_input),
                        output_summary=_summarize(parsed),
                    )
                )
                return parsed, traces
            except Exception as exc:  # Tool boundary: convert errors to traceable observations.
                last_error = str(exc)
                traces.append(
                    ToolTrace(
                        tool_name=tool_name,
                        attempt=attempt,
                        ok=False,
                        input_summary=_summarize(tool_input),
                        output_summary=last_error[:240],
                    )
                )
        return {"error": last_error, "tool": tool_name}, traces

    def _synthesize(
        self,
        user_task: str,
        observations: list[dict[str, object]],
    ) -> ResearchAgentResponse:
        tools_used = [str(observation["tool"]) for observation in observations]
        warnings: list[str] = []
        citations: list[str] = []
        answer_parts: list[str] = []

        for observation in observations:
            result = observation["result"]
            if not isinstance(result, dict):
                continue
            if "error" in result:
                return ResearchAgentResponse(
                    status=AgentStatus.NEEDS_REVIEW.value,
                    answer=f"Tool {result.get('tool')} failed: {result['error']}",
                    tools_used=tools_used,
                    warnings=["tool_failure"],
                )
            if "answer" in result:
                answer_parts.append(str(result["answer"]))
            if "comparison" in result:
                answer_parts.append(str(result["comparison"]))
            citations.extend(str(item) for item in result.get("citations", []))
            warnings.extend(str(item) for item in result.get("warnings", []))

        answer = " ".join(answer_parts) or f"No tool produced a direct answer for: {user_task}"
        return ResearchAgentResponse(
            status=AgentStatus.ANSWERED.value,
            answer=answer,
            citations=sorted(set(citations)),
            tools_used=tools_used,
            warnings=sorted(set(warnings)),
        )


def _failed(message: str, traces: list[ToolTrace]) -> AgentRun:
    return AgentRun(
        response=ResearchAgentResponse(status=AgentStatus.FAILED.value, answer=message),
        traces=tuple(traces),
    )


def _needs_review(message: str, traces: list[ToolTrace]) -> AgentRun:
    return AgentRun(
        response=ResearchAgentResponse(status=AgentStatus.NEEDS_REVIEW.value, answer=message),
        traces=tuple(traces),
    )


def _summarize(value: object, *, max_chars: int = 240) -> str:
    text = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
    return text[:max_chars]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))

