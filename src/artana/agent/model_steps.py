from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from artana._kernel.types import ContextVersion, ReplayPolicy, StepModelResult
from artana.events import ChatMessage
from artana.kernel import ArtanaKernel, ModelInput
from artana.models import TenantContext

OutputT = TypeVar("OutputT", bound=BaseModel)


async def execute_model_step(
    *,
    kernel: ArtanaKernel,
    run_id: str,
    tenant: TenantContext,
    model: str,
    messages: tuple[ChatMessage, ...],
    output_schema: type[OutputT],
    step_key: str,
    visible_tool_names: set[str] | None,
    replay_policy: ReplayPolicy = "strict",
    context_version: ContextVersion | None = None,
) -> StepModelResult[OutputT]:
    return await kernel.step_model_with_visible_tools(
        run_id=run_id,
        tenant=tenant,
        model=model,
        input=ModelInput.from_messages(messages),
        output_schema=output_schema,
        visible_tool_names=visible_tool_names,
        step_key=step_key,
        replay_policy=replay_policy,
        context_version=context_version,
    )


__all__ = ["execute_model_step"]
