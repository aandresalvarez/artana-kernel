from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from artana.events import ChatMessage
from artana.models import TenantContext
from artana.ports.model import ModelUsage, ToolDefinition


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelInvocation:
    run_id: str
    tenant: TenantContext
    model: str
    prompt: str
    messages: tuple[ChatMessage, ...]
    allowed_tools: tuple[ToolDefinition, ...]
    tool_capability_by_name: dict[str, str | None]

    def with_updates(
        self,
        *,
        prompt: str | None = None,
        messages: tuple[ChatMessage, ...] | None = None,
        allowed_tools: tuple[ToolDefinition, ...] | None = None,
    ) -> "ModelInvocation":
        return replace(
            self,
            prompt=self.prompt if prompt is None else prompt,
            messages=self.messages if messages is None else messages,
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
