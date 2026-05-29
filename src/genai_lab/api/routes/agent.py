"""Agent task endpoint."""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter

from genai_lab.agents.executor import ToolUsingResearchAgent
from genai_lab.api.schemas import AgentRequest
from genai_lab.config.settings import get_settings

router = APIRouter()


@router.post("/agent")
def run_agent(req: AgentRequest) -> dict[str, object]:
    settings = get_settings()
    agent = ToolUsingResearchAgent(settings=settings)
    run = agent.run(req.task)
    return {
        "response": run.response.model_dump(),
        "traces": [dataclasses.asdict(t) for t in run.traces],
    }
