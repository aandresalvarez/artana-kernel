from __future__ import annotations

import json
from collections.abc import Sequence

from artana._kernel.replay import (
    ModelStepResult,
    deserialize_model_completed,
    find_matching_model_cycle,
)
from artana._kernel.types import ContextVersion, OutputT, ReplayPolicy
from artana.canonicalization import (
    canonical_json_dumps,
    canonicalize_json_object,
)
from artana.events import (
    ChatMessage,
    ContextVersionRecord,
    EventPayload,
    EventType,
    KernelEvent,
    ModelCompletedPayload,
    ModelRequestedPayload,
    ReplayedWithDriftPayload,
    ToolCallRecord,
    ToolSignatureRecord,
    compute_allowed_tools_hash,
)
from artana.json_utils import sha256_hex
from artana.middleware.base import KernelMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelPort, ModelRequest, ToolDefinition
from artana.store.base import EventStore


async def _append_event_with_parent(
    store: EventStore,
    *,
    run_id: str,
    tenant_id: str,
    event_type: EventType,
    payload: EventPayload,
    parent_step_key: str | None = None,
) -> KernelEvent:
    append_kwargs = {
        "run_id": run_id,
        "tenant_id": tenant_id,
        "event_type": event_type,
        "payload": payload,
    }
    if parent_step_key is not None:
        append_kwargs["parent_step_key"] = parent_step_key
    return await store.append_event(**append_kwargs)


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
    events: Sequence[KernelEvent],
    step_key: str | None = None,
    parent_step_key: str | None = None,
    replay_policy: ReplayPolicy = "strict",
    context_version: ContextVersion | None = None,
) -> ModelStepResult[OutputT]:
    tool_signatures = tool_signatures_from_definitions(tool_definitions)
    normalized_tool_names = sorted(tool.name for tool in tool_definitions)
    signature_tokens = [_signature_token(signature) for signature in tool_signatures]
    lookup = find_matching_model_cycle(
        events=events,
        prompt=prompt,
        messages=messages,
        model=model,
        allowed_tool_signatures=tool_signatures,
        step_key=step_key,
        replay_policy=replay_policy,
    )
    request_event = lookup.request_event
    completed_event = lookup.completed_event

    if completed_event is not None:
        if lookup.replayed_with_drift and request_event is not None:
            await _append_event_with_parent(
                store=store,
                run_id=run_id,
                tenant_id=tenant.tenant_id,
                event_type=EventType.REPLAYED_WITH_DRIFT,
                parent_step_key=parent_step_key,
                payload=ReplayedWithDriftPayload(
                    step_key=step_key,
                    model=model,
                    drift_fields=list(lookup.drift_fields),
                    source_model_requested_event_id=request_event.event_id,
                    source_model_completed_seq=completed_event.seq,
                    replay_policy="allow_prompt_drift",
                ),
            )
            return deserialize_model_completed(
                event=completed_event,
                output_schema=output_schema,
                replayed=True,
                replayed_with_drift=lookup.replayed_with_drift,
                drift_fields=lookup.drift_fields,
            )
        return deserialize_model_completed(
            event=completed_event,
            output_schema=output_schema,
            replayed=True,
            replayed_with_drift=False,
        )

    if request_event is None:
        await _append_event_with_parent(
            store=store,
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.MODEL_REQUESTED,
            parent_step_key=parent_step_key,
            payload=ModelRequestedPayload(
                model=model,
                prompt=prompt,
                messages=list(messages),
                allowed_tools=normalized_tool_names,
                allowed_tool_signatures=tool_signatures,
                allowed_tools_hash=compute_allowed_tools_hash(signature_tokens),
                step_key=step_key,
                context_version=_to_context_version_record(
                    context_version,
                    prompt=prompt,
                    messages=messages,
                ),
            ),
        )

    for middleware_item in middleware:
        await middleware_item.before_model(run_id=run_id, tenant=tenant)

    result = await model_port.complete(
        ModelRequest(
            run_id=run_id,
            model=model,
            prompt=prompt,
            messages=messages,
            output_schema=output_schema,
            allowed_tools=tool_definitions,
        )
    )
    completed_event = await _append_event_with_parent(
        store=store,
        run_id=run_id,
        tenant_id=tenant.tenant_id,
        event_type=EventType.MODEL_COMPLETED,
        parent_step_key=parent_step_key,
        payload=ModelCompletedPayload(
            model=model,
            output_json=result.output.model_dump_json(),
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            cost_usd=result.usage.cost_usd,
            tool_calls=[
                ToolCallRecord(
                    tool_name=tool_call.tool_name,
                    arguments_json=canonicalize_json_object(tool_call.arguments_json),
                    tool_call_id=tool_call.tool_call_id,
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
        replayed_with_drift=False,
    )


def tool_signatures_from_definitions(
    tool_definitions: Sequence[ToolDefinition],
) -> list[ToolSignatureRecord]:
    signatures = [
        ToolSignatureRecord(
            name=tool.name,
            tool_version=tool.tool_version,
            schema_version=tool.schema_version,
            schema_hash=_tool_schema_hash(tool),
        )
        for tool in tool_definitions
    ]
    return sorted(
        signatures,
        key=lambda signature: (
            signature.name,
            signature.tool_version,
            signature.schema_version,
            signature.schema_hash,
        ),
    )


def _tool_schema_hash(tool: ToolDefinition) -> str:
    if tool.schema_hash != "":
        return tool.schema_hash
    parsed = json.loads(tool.arguments_schema_json)
    return sha256_hex(canonical_json_dumps(parsed))


def _signature_token(signature: ToolSignatureRecord) -> str:
    return (
        f"{signature.name}|{signature.tool_version}|"
        f"{signature.schema_version}|{signature.schema_hash}"
    )


def _to_context_version_record(
    context_version: ContextVersion | None,
    *,
    prompt: str,
    messages: tuple[ChatMessage, ...],
) -> ContextVersionRecord | None:
    if context_version is None:
        return ContextVersionRecord(
            system_prompt_hash=_fallback_system_prompt_hash(prompt=prompt, messages=messages),
            context_builder_version="unknown",
            compaction_version="unknown",
        )
    return ContextVersionRecord(
        system_prompt_hash=context_version.system_prompt_hash,
        context_builder_version=context_version.context_builder_version,
        compaction_version=context_version.compaction_version,
    )

def _fallback_system_prompt_hash(*, prompt: str, messages: tuple[ChatMessage, ...]) -> str:
    system_messages = [message.content for message in messages if message.role == "system"]
    if system_messages:
        return sha256_hex("\n".join(system_messages))
    return sha256_hex(prompt)
