from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import Literal, cast
from uuid import uuid4

from artana._kernel.replay import (
    ModelStepResult,
    deserialize_model_terminal,
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
    ModelRequestedPayload,
    ModelTerminalPayload,
    ReplayedWithDriftPayload,
    ToolCallRecord,
    ToolSignatureRecord,
    compute_allowed_tools_hash,
)
from artana.json_utils import sha256_hex
from artana.middleware.base import KernelMiddleware
from artana.models import TenantContext
from artana.ports.model import (
    ModelCallOptions,
    ModelPermanentError,
    ModelPort,
    ModelRequest,
    ModelResult,
    ModelTimeoutError,
    ModelTransientError,
    ToolDefinition,
)
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
    retry_failed_step: bool = False,
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
    terminal_event = lookup.terminal_event

    if terminal_event is not None:
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
                    source_model_terminal_seq=terminal_event.seq,
                    replay_policy="allow_prompt_drift",
                ),
            )
            return deserialize_model_terminal(
                event=terminal_event,
                output_schema=output_schema,
                replayed=True,
                replayed_with_drift=lookup.replayed_with_drift,
                drift_fields=lookup.drift_fields,
            )
        if retry_failed_step and _is_failed_terminal_event(terminal_event):
            request_event = None
        else:
            return deserialize_model_terminal(
                event=terminal_event,
                output_schema=output_schema,
                replayed=True,
                replayed_with_drift=False,
            )

    if request_event is None:
        request_event = await _append_event_with_parent(
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
                model_cycle_id=uuid4().hex,
                context_version=_to_context_version_record(
                    context_version,
                    prompt=prompt,
                    messages=messages,
                ),
            ),
        )

    request_payload = request_event.payload
    model_cycle_id = (
        request_payload.model_cycle_id
        if isinstance(request_payload, ModelRequestedPayload)
        and request_payload.model_cycle_id is not None
        else request_event.event_id
    )
    terminal_payload: ModelTerminalPayload | None = None
    terminal_event_written: KernelEvent | None = None
    pending_exception: BaseException | None = None
    result: ModelResult[OutputT] | None = None
    started_at = time.monotonic()

    try:
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
        for middleware_item in middleware:
            await middleware_item.after_model(
                run_id=run_id,
                tenant=tenant,
                usage=result.usage,
            )

        terminal_payload = ModelTerminalPayload(
            outcome="completed",
            model=model,
            model_cycle_id=model_cycle_id,
            source_model_requested_event_id=request_event.event_id,
            step_key=step_key,
            elapsed_ms=_elapsed_ms(started_at),
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
        )
    except BaseException as exc:
        pending_exception = exc
        outcome, error_category = _classify_failure(exc)
        prompt_tokens = result.usage.prompt_tokens if result is not None else None
        completion_tokens = result.usage.completion_tokens if result is not None else None
        cost_usd = result.usage.cost_usd if result is not None else None
        api_mode_used = result.api_mode_used if result is not None else None
        response_id = result.response_id if result is not None else None
        responses_output_items = (
            _canonicalize_items(result.response_output_items) if result is not None else []
        )
        terminal_payload = ModelTerminalPayload(
            outcome=outcome,
            model=model,
            model_cycle_id=model_cycle_id,
            source_model_requested_event_id=request_event.event_id,
            step_key=step_key,
            failure_reason=error_category,
            error_category=error_category,
            error_class=type(exc).__name__,
            http_status=_extract_http_status(exc),
            provider_request_id=_extract_provider_request_id(exc),
            elapsed_ms=_elapsed_ms(started_at),
            diagnostics_json=canonical_json_dumps(
                {
                    "message": str(exc),
                    "exception_module": type(exc).__module__,
                }
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            api_mode_used=api_mode_used,
            response_id=response_id,
            responses_output_items=responses_output_items,
        )
    finally:
        if terminal_payload is not None:
            terminal_event_written = await _append_event_with_parent(
                store=store,
                run_id=run_id,
                tenant_id=tenant.tenant_id,
                event_type=EventType.MODEL_TERMINAL,
                parent_step_key=parent_step_key,
                payload=terminal_payload,
            )

    if pending_exception is not None:
        raise pending_exception
    if terminal_event_written is None or result is None:
        raise RuntimeError("Model terminal event was not persisted for executed model step.")

    return ModelStepResult(
        completed_seq=terminal_event_written.seq,
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


def _is_failed_terminal_event(event: KernelEvent) -> bool:
    if event.event_type != EventType.MODEL_TERMINAL:
        return False
    payload = event.payload
    return isinstance(payload, ModelTerminalPayload) and payload.outcome != "completed"


def _elapsed_ms(started_at: float) -> int:
    elapsed_seconds = max(0.0, time.monotonic() - started_at)
    return int(elapsed_seconds * 1000)


def _classify_failure(
    exc: BaseException,
) -> tuple[Literal["failed", "timeout", "cancelled"], str]:
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled", "cancelled"
    if isinstance(exc, (asyncio.TimeoutError, ModelTimeoutError)):
        return "timeout", "timeout"
    http_status = _extract_http_status(exc)
    if http_status is not None:
        if 400 <= http_status < 500:
            return "failed", "provider_4xx"
        if http_status >= 500:
            return "failed", "provider_5xx"
    if isinstance(exc, ModelTransientError):
        return "failed", "transient"
    if isinstance(exc, ModelPermanentError):
        return "failed", "permanent"
    if isinstance(exc, (ConnectionError, OSError)):
        return "failed", "network"
    return "failed", "internal"


def _extract_http_status(exc: BaseException) -> int | None:
    for attribute in ("status_code", "status", "http_status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _extract_provider_request_id(exc: BaseException) -> str | None:
    for attribute in ("request_id", "provider_request_id", "x_request_id"):
        value = getattr(exc, attribute, None)
        if isinstance(value, str) and value != "":
            return value
    return None


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
