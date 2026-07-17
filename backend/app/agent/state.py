"""Agent state shared across LangGraph nodes."""
from __future__ import annotations

from typing import Any, TypedDict


class Step(TypedDict):
    thought: str
    tool: str
    args: dict[str, Any]
    observation: str


class AgentState(TypedDict, total=False):
    task: str
    plan: str
    scratchpad: list[Step]
    step: int
    max_steps: int
    critiques: int
    max_critiques: int
    pending: dict[str, Any] | None
    final_answer: str | None
    forced_finish: bool
    status: str
