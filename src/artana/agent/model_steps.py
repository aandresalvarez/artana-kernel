from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from artana.events import ChatMessage
from artana.kernel import (
    ArtanaKernel,
    ContextVersion,
    ModelInput,
    ReplayPolicy,
    StepModelResult,
)
from artana.models import TenantContext
from artana.ports.model import ModelCallOptions

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
    model_options: ModelCallOptions | None = None,
    replay_policy: ReplayPolicy = "strict",
    context_version: ContextVersion | None = None,
    retry_failed_step: bool = False,
    parent_step_key: str | None = None,
) -> StepModelResult[OutputT]:
    return await kernel.step_model_with_visible_tools(
        run_id=run_id,
        tenant=tenant,
        model=model,
        input=ModelInput.from_messages(messages),
        output_schema=output_schema,
        visible_tool_names=visible_tool_names,
        model_options=model_options,
        step_key=step_key,
        replay_policy=replay_policy,
        context_version=context_version,
        retry_failed_step=retry_failed_step,
        parent_step_key=parent_step_key,
    )


__all__ = ["execute_model_step"]
