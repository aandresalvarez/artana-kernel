from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from artana._kernel.model_cycle import get_or_execute_model_step
from artana._kernel.policies import apply_prepare_model_middleware
from artana._kernel.replay import validate_tenant_for_run
from artana._kernel.types import StepModelResult
from artana.events import ChatMessage
from artana.kernel import ArtanaKernel
from artana.middleware.base import ModelInvocation
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
) -> StepModelResult[OutputT]:
    events = await kernel._store.get_events_for_run(run_id)
    if not events:
        raise ValueError(f"No events found for run_id={run_id!r}; start run first.")
    validate_tenant_for_run(events=events, tenant=tenant)

    all_tools = kernel._tool_port.to_all_tool_definitions()
    capability_map = kernel._tool_port.capability_map()
    if visible_tool_names is None:
        allowed_tool_definitions = tuple(all_tools)
        filtered_capabilities = capability_map
    else:
        allowed_tool_definitions = tuple(
            tool for tool in all_tools if tool.name in visible_tool_names
        )
        filtered_capabilities = {
            name: capability
            for name, capability in capability_map.items()
            if name in visible_tool_names
        }

    invocation = ModelInvocation(
        run_id=run_id,
        tenant=tenant,
        model=model,
        prompt=messages[-1].content if messages else "",
        messages=messages,
        allowed_tools=allowed_tool_definitions,
        tool_capability_by_name=filtered_capabilities,
    )
    prepared = await apply_prepare_model_middleware(kernel._middleware, invocation)

    result = await get_or_execute_model_step(
        store=kernel._store,
        model_port=kernel._model_port,
        middleware=kernel._middleware,
        run_id=run_id,
        prompt=prepared.prompt,
        messages=prepared.messages,
        model=prepared.model,
        tenant=tenant,
        output_schema=output_schema,
        tool_definitions=prepared.allowed_tools,
        allowed_tool_names=[tool.name for tool in prepared.allowed_tools],
        events=events,
        step_key=step_key,
    )
    return StepModelResult(
        run_id=run_id,
        seq=result.completed_seq,
        output=result.output,
        usage=result.usage,
        tool_calls=result.tool_calls,
        replayed=result.replayed,
    )


__all__ = ["execute_model_step"]
