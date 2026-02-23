from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from artana.events import ChatMessage

OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ModelUsage:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    arguments_schema_json: str
    tool_version: str = "1.0.0"
    schema_version: str = "1"
    schema_hash: str = ""


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool_name: str
    arguments_json: str
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ModelRequest(Generic[OutputT]):
    run_id: str
    model: str
    prompt: str
    messages: Sequence[ChatMessage]
    output_schema: type[OutputT]
    allowed_tools: Sequence[ToolDefinition]


@dataclass(frozen=True, slots=True)
class ModelResult(Generic[OutputT]):
    output: OutputT
    usage: ModelUsage
    tool_calls: tuple[ToolCall, ...] = ()
    raw_output: str = ""


class ModelPort(Protocol):
    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        ...


class ModelTimeoutError(RuntimeError):
    pass


class ModelTransientError(RuntimeError):
    pass


class ModelPermanentError(RuntimeError):
    pass


@runtime_checkable
class SupportsModelDump(Protocol):
    def model_dump(self) -> dict[str, object]:
        ...


class LiteLLMCompletionFn(Protocol):
    async def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        response_format: type[BaseModel],
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        ...
