from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HarnessTaskState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    state: Literal["pending", "in_progress", "done"]


class WorkspaceState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str | None = None
    question: str | None = None
    memory_summary: str | None = None
    active_plan: str | None = None
    graph_summary: str | None = None
    evidence_count: int | None = Field(default=None, ge=0)
    artifacts: dict[str, object] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    open_tasks: list[str] = Field(default_factory=list)
    unresolved_contradictions: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    notes: dict[str, object] = Field(default_factory=dict)


class HarnessOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["in_progress", "completed", "blocked", "needs_review", "failed"]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    gates_passed: list[str] = Field(default_factory=list)
    gates_failed: list[str] = Field(default_factory=list)
    artifacts_produced: list[str] = Field(default_factory=list)
    next_recommended_action: str | None = None
    human_review_needed: bool = False
    workspace_state: WorkspaceState | None = None
    task_progress: list[HarnessTaskState] = Field(default_factory=list)
    details: dict[str, object] = Field(default_factory=dict)


__all__ = ["HarnessOutcome", "HarnessTaskState", "WorkspaceState"]
