"""Pydantic schemas for TaskForge API I/O."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=2000)
    max_steps: int | None = Field(None, ge=1, le=20)


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    edited_args: dict[str, Any] | None = None


class TaskItem(BaseModel):
    id: int
    title: str
    notes: str | None = None
    due: str | None = None
    status: str
    created_at: str


class RunEvent(BaseModel):
    event: str
    data: dict[str, Any]
    ts: str


class RunSummary(BaseModel):
    id: str
    task: str
    status: str
    created_at: str
    elapsed_s: float | None = None
    final_preview: str | None = None


class RunDetail(RunSummary):
    events: list[RunEvent]
