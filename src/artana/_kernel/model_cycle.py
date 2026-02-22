from __future__ import annotations

from collections.abc import Sequence

from artana._kernel.replay import (
    ModelStepResult,
    deserialize_model_completed,
    find_matching_model_cycle,
)
from artana._kernel.types import OutputT
from artana.events import (
    ChatMessage,
    KernelEvent,
    ModelCompletedPayload,
    ModelRequestedPayload,
    ToolCallRecord,
)
from artana.middleware.base import KernelMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelPort, ModelRequest, ToolDefinition
from artana.store.base import EventStore


async def get_or_execute_model_step(
    *,
    store: EventStore,
    model_port: ModelPort,
    middleware: Sequence[KernelMiddleware],
    run_id: str,
    prompt: str,
    messages: tuple[ChatMessage, ...],
    model: str,
    tenant: TenantContext,
    output_schema: type[OutputT],
    tool_definitions: Sequence[ToolDefinition],
    allowed_tool_names: list[str],
    events: Sequence[KernelEvent],
) -> ModelStepResult[OutputT]:
    request_event, completed_event = find_matching_model_cycle(
        events=events,
        prompt=prompt,
        model=model,
        allowed_tool_names=allowed_tool_names,
    )
    if completed_event is not None:
        return deserialize_model_completed(
            event=completed_event,
            output_schema=output_schema,
            replayed=True,
        )

    if request_event is None:
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type="model_requested",
            payload=ModelRequestedPayload(
                model=model,
                prompt=prompt,
                messages=list(messages),
                allowed_tools=allowed_tool_names,
            ),
        )

    for middleware_item in middleware:
        await middleware_item.before_model(run_id=run_id, tenant=tenant)

    result = await model_port.complete(
        ModelRequest(
            run_id=run_id,
            model=model,
            prompt=prompt,
            output_schema=output_schema,
            allowed_tools=tool_definitions,
        )
    )
    completed_event = await store.append_event(
        run_id=run_id,
        tenant_id=tenant.tenant_id,
        event_type="model_completed",
        payload=ModelCompletedPayload(
            model=model,
            output_json=result.output.model_dump_json(),
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            cost_usd=result.usage.cost_usd,
            tool_calls=[
                ToolCallRecord(
                    tool_name=tool_call.tool_name,
                    arguments_json=tool_call.arguments_json,
                )
                for tool_call in result.tool_calls
            ],
        ),
    )
    for middleware_item in middleware:
        await middleware_item.after_model(
            run_id=run_id,
            tenant=tenant,
            usage=result.usage,
        )

    return ModelStepResult(
        completed_seq=completed_event.seq,
        output=result.output,
        usage=result.usage,
        tool_calls=result.tool_calls,
        replayed=False,
    )
