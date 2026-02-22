from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

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
class ChatResponse(Generic[OutputT]):
    run_id: str
    output: OutputT
    usage: ModelUsage
    replayed: bool


@dataclass(frozen=True, slots=True)
class RunResumeState:
    run_id: str
    status: Literal["paused", "pending_tool", "ready", "complete"]
    last_seq: int
    pause_reason: str | None
    pending_tool: ToolCall | None

