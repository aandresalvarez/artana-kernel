from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
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
    mode: Literal["permissive", "enforced"] = "permissive"

    @classmethod
    def enforced(cls) -> "KernelPolicy":
        return cls(mode="enforced")


@dataclass(frozen=True, slots=True)
class ContextVersion:
    system_prompt_hash: str | None = None
    context_builder_version: str | None = None
    compaction_version: str | None = None


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


@dataclass(frozen=True, slots=True)
class StepToolResult:
    run_id: str
    seq: int
    tool_name: str
    result_json: str
    replayed: bool
