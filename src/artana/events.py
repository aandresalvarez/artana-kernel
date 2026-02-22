from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EventType = Literal[
    "model_requested",
    "model_completed",
    "tool_requested",
    "tool_completed",
    "pause_requested",
    "workflow_step_requested",
    "workflow_step_completed",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ModelRequestedPayload(BaseModel):
    kind: Literal["model_requested"] = "model_requested"
    model: str
    prompt: str
    messages: list[ChatMessage]
    allowed_tools: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments_json: str


class ModelCompletedPayload(BaseModel):
    kind: Literal["model_completed"] = "model_completed"
    model: str
    output_json: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class ToolRequestedPayload(BaseModel):
    kind: Literal["tool_requested"] = "tool_requested"
    tool_name: str
    arguments_json: str
    idempotency_key: str
    tool_version: str = "1.0.0"
    schema_version: str = "1"


class ToolCompletedPayload(BaseModel):
    kind: Literal["tool_completed"] = "tool_completed"
    tool_name: str
    result_json: str
    outcome: Literal[
        "success",
        "transient_error",
        "permanent_error",
        "unknown_outcome",
    ] = "success"
    received_idempotency_key: str | None = None
    effect_id: str | None = None
    request_id: str | None = None
    error_message: str | None = None


class PauseRequestedPayload(BaseModel):
    kind: Literal["pause_requested"] = "pause_requested"
    reason: str


class WorkflowStepRequestedPayload(BaseModel):
    kind: Literal["workflow_step_requested"] = "workflow_step_requested"
    step_index: int = Field(ge=0)
    step_name: str


class WorkflowStepCompletedPayload(BaseModel):
    kind: Literal["workflow_step_completed"] = "workflow_step_completed"
    step_index: int = Field(ge=0)
    step_name: str
    result_json: str


EventPayload = (
    ModelRequestedPayload
    | ModelCompletedPayload
    | ToolRequestedPayload
    | ToolCompletedPayload
    | PauseRequestedPayload
    | WorkflowStepRequestedPayload
    | WorkflowStepCompletedPayload
)


class KernelEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    run_id: str
    tenant_id: str
    seq: int = Field(ge=1)
    event_type: EventType
    prev_event_hash: str | None = None
    event_hash: str
    timestamp: datetime = Field(default_factory=utc_now)
    payload: EventPayload

    @model_validator(mode="after")
    def payload_matches_event_type(self) -> "KernelEvent":
        if self.event_type != self.payload.kind:
            raise ValueError(
                f"event_type={self.event_type} does not match payload kind={self.payload.kind}"
            )
        expected_hash = compute_event_hash(
            event_id=self.event_id,
            run_id=self.run_id,
            tenant_id=self.tenant_id,
            seq=self.seq,
            event_type=self.event_type,
            prev_event_hash=self.prev_event_hash,
            timestamp=self.timestamp,
            payload=self.payload,
        )
        if expected_hash != self.event_hash:
            raise ValueError(
                "event_hash mismatch for "
                f"seq={self.seq}. expected={expected_hash}, got={self.event_hash}"
            )
        return self


def payload_to_canonical_json(payload: EventPayload) -> str:
    return json.dumps(payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def compute_event_hash(
    *,
    event_id: str,
    run_id: str,
    tenant_id: str,
    seq: int,
    event_type: EventType,
    prev_event_hash: str | None,
    timestamp: datetime,
    payload: EventPayload,
) -> str:
    joined = "|".join(
        (
            event_id,
            run_id,
            tenant_id,
            str(seq),
            event_type,
            prev_event_hash or "",
            timestamp.isoformat(),
            payload_to_canonical_json(payload),
        )
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
