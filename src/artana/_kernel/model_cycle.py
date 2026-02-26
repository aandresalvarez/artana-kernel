from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

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
from artana.ports.model import ModelCallOptions, ModelPort, ModelRequest, ToolDefinition
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
    if parent_step_key is None:
        return await store.append_event(
            run_id=run_id,
            tenant_id=tenant_id,
            event_type=event_type,
            payload=payload,
        )
    return await store.append_event(
        run_id=run_id,
        tenant_id=tenant_id,
        event_type=event_type,
        payload=payload,
        parent_step_key=parent_step_key,
    )


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
    model_options: ModelCallOptions | None = None,
    responses_input_items: list[dict[str, object]] | None = None,
    step_key: str | None = None,
    parent_step_key: str | None = None,
    replay_policy: ReplayPolicy = "strict",
    context_version: ContextVersion | None = None,
) -> ModelStepResult[OutputT]:
    resolved_model_options = model_options or ModelCallOptions()
    canonical_responses_input_items = (
        _canonicalize_items(responses_input_items)
        if responses_input_items is not None
        else None
    )
    tool_signatures = tool_signatures_from_definitions(tool_definitions)
    normalized_tool_names = sorted(tool.name for tool in tool_definitions)
    signature_tokens = [_signature_token(signature) for signature in tool_signatures]
    lookup = find_matching_model_cycle(
        events=events,
        prompt=prompt,
        messages=messages,
        model=model,
        model_options=resolved_model_options,
        responses_input_items=canonical_responses_input_items,
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
                api_mode=resolved_model_options.api_mode,
                reasoning_effort=resolved_model_options.reasoning_effort,
                verbosity=resolved_model_options.verbosity,
                previous_response_id=resolved_model_options.previous_response_id,
                responses_input_items=canonical_responses_input_items,
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
            model_options=resolved_model_options,
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
            api_mode_used=result.api_mode_used,
            response_id=result.response_id,
            responses_output_items=_canonicalize_items(result.response_output_items),
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
        api_mode_used=result.api_mode_used,
        response_id=result.response_id,
        response_output_items=result.response_output_items,
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


def _canonicalize_items(
    items: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        cast(
            dict[str, object],
            json.loads(json.dumps(item, sort_keys=True, separators=(",", ":"))),
        )
        for item in items
    ]
