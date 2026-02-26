from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from artana.events import ChatMessage
from artana.models import TenantContext
from artana.ports.model import ModelCallOptions, ModelUsage, ToolDefinition


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedToolRequest:
    arguments_json: str
    semantic_idempotency_key: str | None = None
    intent_id: str | None = None
    amount_usd: float | None = None

    def with_arguments_json(self, arguments_json: str) -> "PreparedToolRequest":
        return replace(self, arguments_json=arguments_json)

    def merge(self, other: "PreparedToolRequest") -> "PreparedToolRequest":
        return PreparedToolRequest(
            arguments_json=other.arguments_json,
            semantic_idempotency_key=(
                other.semantic_idempotency_key
                if other.semantic_idempotency_key is not None
                else self.semantic_idempotency_key
            ),
            intent_id=other.intent_id if other.intent_id is not None else self.intent_id,
            amount_usd=other.amount_usd if other.amount_usd is not None else self.amount_usd,
        )


@dataclass(frozen=True, slots=True)
class ModelInvocation:
    run_id: str
    tenant: TenantContext
    model: str
    prompt: str
    messages: tuple[ChatMessage, ...]
    model_options: ModelCallOptions
    allowed_tools: tuple[ToolDefinition, ...]
    tool_capability_by_name: dict[str, str | None]

    def with_updates(
        self,
        *,
        prompt: str | None = None,
        messages: tuple[ChatMessage, ...] | None = None,
        model_options: ModelCallOptions | None = None,
        allowed_tools: tuple[ToolDefinition, ...] | None = None,
    ) -> "ModelInvocation":
        return replace(
            self,
            prompt=self.prompt if prompt is None else prompt,
            messages=self.messages if messages is None else messages,
            model_options=(
                self.model_options if model_options is None else model_options
            ),
            allowed_tools=self.allowed_tools if allowed_tools is None else allowed_tools,
        )


class KernelMiddleware(Protocol):
    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        ...

    async def before_model(self, *, run_id: str, tenant: TenantContext) -> None:
        ...

    async def after_model(
        self, *, run_id: str, tenant: TenantContext, usage: ModelUsage
    ) -> None:
        ...

    async def prepare_tool_request(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments_json: str,
    ) -> str | PreparedToolRequest:
        ...

    async def prepare_tool_result(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        result_json: str,
    ) -> str:
        ...
