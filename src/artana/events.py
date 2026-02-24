from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from artana.events_hashing import compute_event_hash as compute_event_hash


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    RESUME_REQUESTED = "resume_requested"
    HARNESS_INITIALIZED = "harness_initialized"
    HARNESS_WAKE = "harness_wake"
    HARNESS_SLEEP = "harness_sleep"
    HARNESS_FAILED = "harness_failed"
    HARNESS_STAGE = "harness_stage"
    MODEL_REQUESTED = "model_requested"
    MODEL_COMPLETED = "model_completed"
    REPLAYED_WITH_DRIFT = "replayed_with_drift"
    TOOL_REQUESTED = "tool_requested"
    TOOL_COMPLETED = "tool_completed"
    PAUSE_REQUESTED = "pause_requested"
    RUN_SUMMARY = "run_summary"
    WORKFLOW_STEP_REQUESTED = "workflow_step_requested"
    WORKFLOW_STEP_COMPLETED = "workflow_step_completed"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list["ToolCallMessage"] | None = None


class ToolFunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCallMessage(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolFunctionCall


class ToolSignatureRecord(BaseModel):
    name: str
    tool_version: str
    schema_version: str
    schema_hash: str


class ContextVersionRecord(BaseModel):
    system_prompt_hash: str | None = None
    context_builder_version: str | None = None
    compaction_version: str | None = None


class ModelRequestedPayload(BaseModel):
    kind: Literal["model_requested"] = "model_requested"
    model: str
    prompt: str
    messages: list[ChatMessage]
    allowed_tools: list[str] = Field(default_factory=list)
    allowed_tool_signatures: list[ToolSignatureRecord] = Field(default_factory=list)
    allowed_tools_hash: str | None = None
    step_key: str | None = None
    context_version: ContextVersionRecord | None = None


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments_json: str
    tool_call_id: str | None = None


class ModelCompletedPayload(BaseModel):
    kind: Literal["model_completed"] = "model_completed"
    model: str
    output_json: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class ReplayedWithDriftPayload(BaseModel):
    kind: Literal["replayed_with_drift"] = "replayed_with_drift"
    step_key: str | None = None
    model: str
    drift_fields: list[str] = Field(default_factory=list)
    source_model_requested_event_id: str
    source_model_completed_seq: int | None = None
    replay_policy: Literal["allow_prompt_drift", "fork_on_drift"]
    fork_run_id: str | None = None


class ToolRequestedPayload(BaseModel):
    kind: Literal["tool_requested"] = "tool_requested"
    tool_name: str
    arguments_json: str
    idempotency_key: str
    tool_version: str = "1.0.0"
    schema_version: str = "1"
    step_key: str | None = None
    semantic_idempotency_key: str | None = None
    intent_id: str | None = None
    amount_usd: float | None = Field(default=None, ge=0.0)


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
    context_json: str | None = None
    step_key: str | None = None


class RunSummaryPayload(BaseModel):
    kind: Literal["run_summary"] = "run_summary"
    summary_type: str
    summary_json: str
    step_key: str | None = None


class HarnessInitializedPayload(BaseModel):
    kind: Literal["harness_initialized"] = "harness_initialized"
    harness_name: str | None = None
    model: str | None = None


class HarnessWakePayload(BaseModel):
    kind: Literal["harness_wake"] = "harness_wake"
    run_created: bool
    reason: str | None = None


class HarnessSleepPayload(BaseModel):
    kind: Literal["harness_sleep"] = "harness_sleep"
    status: Literal["completed", "failed"] = "completed"
    execution_error_type: str | None = None
    sleep_error_type: str | None = None


class HarnessFailedPayload(BaseModel):
    kind: Literal["harness_failed"] = "harness_failed"
    error_type: str
    message: str
    last_step_key: str | None = None


class HarnessStagePayload(BaseModel):
    kind: Literal["harness_stage"] = "harness_stage"
    stage: str
    round: int | None = None
    claims_count: int | None = None


class RunStartedPayload(BaseModel):
    kind: Literal["run_started"] = "run_started"


class ResumeRequestedPayload(BaseModel):
    kind: Literal["resume_requested"] = "resume_requested"
    human_input_json: str | None = None


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
    RunStartedPayload
    | ResumeRequestedPayload
    | HarnessInitializedPayload
    | HarnessWakePayload
    | HarnessSleepPayload
    | HarnessFailedPayload
    | HarnessStagePayload
    | ModelRequestedPayload
    | ModelCompletedPayload
    | ReplayedWithDriftPayload
    | ToolRequestedPayload
    | ToolCompletedPayload
    | PauseRequestedPayload
    | RunSummaryPayload
    | WorkflowStepRequestedPayload
    | WorkflowStepCompletedPayload
)


def payload_to_canonical_json(payload: EventPayload) -> str:
    from artana.events_hashing import payload_to_canonical_json as serialize_payload

    return serialize_payload(payload)


def compute_allowed_tools_hash(tool_names: list[str]) -> str:
    from artana.events_hashing import compute_allowed_tools_hash as hash_allowed_tools

    return hash_allowed_tools(tool_names)


class KernelEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    run_id: str
    tenant_id: str
    seq: int = Field(ge=1)
    event_type: EventType
    prev_event_hash: str | None = None
    event_hash: str
    parent_step_key: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)
    payload: EventPayload

    @model_validator(mode="after")
    def payload_matches_event_type(self) -> "KernelEvent":
        if self.event_type.value != self.payload.kind:
            raise ValueError(
                "event_type="
                f"{self.event_type.value} does not match payload kind={self.payload.kind}"
            )
        expected_hash = compute_event_hash(
            event_id=self.event_id,
            run_id=self.run_id,
            tenant_id=self.tenant_id,
            seq=self.seq,
            event_type=self.event_type,
            prev_event_hash=self.prev_event_hash,
            timestamp=self.timestamp,
            parent_step_key=self.parent_step_key,
            payload=self.payload,
        )
        if expected_hash != self.event_hash:
            raise ValueError(
                "event_hash mismatch for "
                f"seq={self.seq}. expected={expected_hash}, got={self.event_hash}"
            )
        return self
