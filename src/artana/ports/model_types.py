from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel

from artana.events import ChatMessage

OutputT = TypeVar("OutputT", bound=BaseModel)
ModelAPIMode = Literal["auto", "responses", "chat"]
ModelAPIModeUsed = Literal["responses", "chat"]
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]
VerbosityLevel = Literal["low", "medium", "high"]


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
    risk_level: str = "medium"
    sandbox_profile: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool_name: str
    arguments_json: str
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ModelCallOptions:
    api_mode: ModelAPIMode = "auto"
    reasoning_effort: ReasoningEffort | None = None
    verbosity: VerbosityLevel | None = None
    previous_response_id: str | None = None


@dataclass(frozen=True, slots=True)
class ModelRequest(Generic[OutputT]):
    run_id: str
    model: str
    prompt: str
    messages: Sequence[ChatMessage]
    output_schema: type[OutputT]
    allowed_tools: Sequence[ToolDefinition]
    model_options: ModelCallOptions = field(default_factory=ModelCallOptions)


@dataclass(frozen=True, slots=True)
class ModelResult(Generic[OutputT]):
    output: OutputT
    usage: ModelUsage
    tool_calls: tuple[ToolCall, ...] = ()
    raw_output: str = ""
    api_mode_used: ModelAPIModeUsed = "chat"
    response_id: str | None = None
    response_output_items: tuple[dict[str, object], ...] = field(default_factory=tuple)


class ModelPort(Protocol):
    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        ...


class ModelTimeoutError(RuntimeError):
    pass


class ModelTransientError(RuntimeError):
    pass


class ModelPermanentError(RuntimeError):
    pass


class ModelOutputValidationError(ModelPermanentError):
    """Provider-bound structured output that failed the requested schema."""

    def __init__(
        self,
        *,
        raw_output: str,
        usage: ModelUsage,
        api_mode_used: ModelAPIModeUsed,
        response_id: str | None,
        response_output_items: tuple[dict[str, object], ...] = (),
    ) -> None:
        super().__init__("Provider structured output failed schema validation.")
        self.raw_output = raw_output
        self.usage = usage
        self.api_mode_used = api_mode_used
        self.response_id = response_id
        self.response_output_items = tuple(
            cast(
                dict[str, object],
                json.loads(json.dumps(item, sort_keys=True, separators=(",", ":"))),
            )
            for item in response_output_items
        )
        self.run_id: str | None = None
        self.seq: int | None = None
        self.replayed: bool | None = None

    @property
    def output(self) -> object:
        """Return a fresh decoding so callers cannot mutate the custody source."""

        try:
            return cast(object, json.loads(self.raw_output))
        except json.JSONDecodeError:
            return self.raw_output

    def bind_kernel_terminal(self, *, run_id: str, seq: int, replayed: bool) -> None:
        """Attach the persisted kernel terminal identity before propagation."""

        if run_id == "":
            raise ValueError("run_id must be nonempty")
        if seq < 1:
            raise ValueError("seq must be positive")
        if self.run_id is not None or self.seq is not None or self.replayed is not None:
            raise RuntimeError("kernel terminal identity is already bound")
        self.run_id = run_id
        self.seq = seq
        self.replayed = replayed


class ModelRefusalError(ModelPermanentError):
    def __init__(
        self,
        refusal: str,
        *,
        usage: ModelUsage | None = None,
        api_mode_used: ModelAPIModeUsed | None = None,
        response_id: str | None = None,
        response_output_items: tuple[dict[str, object], ...] = (),
        raw_output: str = "",
    ) -> None:
        super().__init__(refusal)
        self.refusal = refusal
        self.usage = usage
        self.api_mode_used = api_mode_used
        self.response_id = response_id
        self.response_output_items = response_output_items
        self.raw_output = raw_output


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


class LiteLLMResponsesFn(Protocol):
    async def __call__(
        self,
        *,
        input: str | list[dict[str, object]],
        model: str,
        previous_response_id: str | None = None,
        reasoning: dict[str, object] | None = None,
        text: dict[str, object] | None = None,
        text_format: type[BaseModel] | dict[str, object] | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> object:
        ...


class SupportsModelDumpJSON(Protocol):
    def model_dump_json(self) -> str:
        ...
