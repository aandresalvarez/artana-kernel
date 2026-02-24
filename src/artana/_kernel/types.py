from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

from artana.events import ChatMessage
from artana.ports.model import ModelUsage, ToolCall
from artana.ports.tool import ToolReturnValue

OutputT = TypeVar("OutputT", bound=BaseModel)
ToolCallable = Callable[..., Awaitable[ToolReturnValue]]


class CapabilityDeniedError(PermissionError):
    pass


class ReplayConsistencyError(RuntimeError):
    pass


class ToolExecutionFailedError(RuntimeError):
    pass


class PolicyViolationError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        tool_name: str,
        fingerprint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.tool_name = tool_name
        self.fingerprint = fingerprint


class ApprovalRequiredError(RuntimeError):
    def __init__(
        self,
        *,
        tool_name: str,
        approval_key: str,
        mode: Literal["human", "critic"],
        message: str,
        critic_model: str | None = None,
        fingerprint: str | None = None,
        arguments_json: str | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.approval_key = approval_key
        self.mode = mode
        self.critic_model = critic_model
        self.fingerprint = fingerprint
        self.arguments_json = arguments_json


@dataclass(frozen=True, slots=True)
class PauseTicket:
    run_id: str
    ticket_id: str
    seq: int
    reason: str


@dataclass(frozen=True, slots=True)
class RunHandle:
    run_id: str
    tenant_id: str


type RunRef = RunHandle
type ReplayPolicy = Literal["strict", "allow_prompt_drift", "fork_on_drift"]
type TraceLevel = Literal["minimal", "stage", "verbose"]
type RunLifecycleStatus = Literal["active", "paused", "failed", "completed"]


@dataclass(frozen=True, slots=True)
class ModelInput:
    kind: Literal["prompt", "messages"]
    prompt: str | None = None
    messages: tuple[ChatMessage, ...] | None = None

    @classmethod
    def from_prompt(cls, prompt: str) -> "ModelInput":
        return cls(kind="prompt", prompt=prompt)

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[ChatMessage],
        *,
        prompt: str | None = None,
    ) -> "ModelInput":
        return cls(kind="messages", messages=tuple(messages), prompt=prompt)


@dataclass(frozen=True, slots=True)
class KernelPolicy:
    mode: Literal["permissive", "enforced", "enforced_v2"] = "permissive"

    @classmethod
    def enforced(cls) -> "KernelPolicy":
        return cls(mode="enforced")

    @classmethod
    def enforced_v2(cls) -> "KernelPolicy":
        return cls(mode="enforced_v2")


@dataclass(frozen=True, slots=True)
class ContextVersion:
    system_prompt_hash: str | None = None
    context_builder_version: str | None = None
    compaction_version: str | None = None


@dataclass(frozen=True, slots=True)
class RunStatus:
    run_id: str
    tenant_id: str
    status: RunLifecycleStatus
    last_event_seq: int
    last_event_type: str
    updated_at: datetime
    blocked_on: str | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ResumePoint:
    run_id: str
    last_event_seq: int
    last_step_key: str | None
    blocked_on: str | None


@dataclass(frozen=True, slots=True)
class RunLease:
    run_id: str
    worker_id: str
    lease_expires_at: datetime


@dataclass(frozen=True, slots=True)
class ToolFingerprint:
    tool_name: str
    tool_version: str
    schema_version: str
    schema_hash: str
    risk_level: Literal["low", "medium", "high", "critical"]
    sandbox_profile: str | None


@dataclass(frozen=True, slots=True)
class StepModelResult(Generic[OutputT]):
    run_id: str
    seq: int
    output: OutputT
    usage: ModelUsage
    tool_calls: tuple[ToolCall, ...]
    replayed: bool
    replayed_with_drift: bool = False
    forked_from_run_id: str | None = None
    drift_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StepToolResult:
    run_id: str
    seq: int
    tool_name: str
    result_json: str
    replayed: bool
