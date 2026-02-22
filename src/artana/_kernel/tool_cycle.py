from __future__ import annotations

import hashlib
from collections.abc import Sequence

from artana._kernel.policies import assert_tool_allowed_for_tenant
from artana._kernel.replay import collect_tool_payloads, validate_tenant_for_run
from artana._kernel.types import (
    CapabilityDeniedError,
    ReplayConsistencyError,
    ToolExecutionFailedError,
)
from artana.events import ToolCompletedPayload, ToolRequestedPayload
from artana.models import TenantContext
from artana.ports.model import ToolCall
from artana.ports.tool import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolPort,
    ToolUnknownOutcomeError,
)
from artana.store.base import EventStore


async def execute_tool_with_replay(
    *,
    store: EventStore,
    tool_port: ToolPort,
    run_id: str,
    tenant: TenantContext,
    tool_name: str,
    arguments_json: str,
) -> str:
    events = await store.get_events_for_run(run_id)
    validate_tenant_for_run(events=events, tenant=tenant)
    assert_tool_allowed_for_tenant(
        tool_name=tool_name,
        tenant=tenant,
        capability_map=tool_port.capability_map(),
    )

    requested_payloads, completed_payloads = collect_tool_payloads(events)
    for request_index in range(len(requested_payloads) - 1, -1, -1):
        requested = requested_payloads[request_index]
        if requested.tool_name != tool_name:
            continue
        if requested.arguments_json != arguments_json:
            continue

        if request_index < len(completed_payloads):
            completed = completed_payloads[request_index]
            if completed.tool_name != tool_name:
                raise ReplayConsistencyError(
                    "Tool completion payload does not match requested tool."
                )
            if completed.outcome != "success":
                raise ToolExecutionFailedError(
                    f"Tool {tool_name!r} previously failed with outcome={completed.outcome!r}."
                )
            return completed.result_json

        return await complete_pending_tool_request(
            store=store,
            tool_port=tool_port,
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            tool_name=tool_name,
            arguments_json=arguments_json,
            idempotency_key=requested.idempotency_key,
            request_event_id=None,
            tool_version=requested.tool_version,
            schema_version=requested.schema_version,
        )

    next_seq = events[-1].seq + 1 if events else 1
    idempotency_key = derive_idempotency_key(run_id=run_id, seq=next_seq)
    request_event = await store.append_event(
        run_id=run_id,
        tenant_id=tenant.tenant_id,
        event_type="tool_requested",
        payload=ToolRequestedPayload(
            tool_name=tool_name,
            arguments_json=arguments_json,
            idempotency_key=idempotency_key,
        ),
    )
    return await complete_pending_tool_request(
        store=store,
        tool_port=tool_port,
        run_id=run_id,
        tenant_id=tenant.tenant_id,
        tool_name=tool_name,
        arguments_json=arguments_json,
        idempotency_key=idempotency_key,
        request_event_id=request_event.event_id,
        tool_version="1.0.0",
        schema_version="1",
    )


async def execute_or_replay_tools_for_model(
    *,
    store: EventStore,
    tool_port: ToolPort,
    run_id: str,
    tenant: TenantContext,
    model_completed_seq: int,
    expected_tool_calls: Sequence[ToolCall],
    allowed_tool_names: frozenset[str],
) -> None:
    current_events = await store.get_events_for_run(run_id)
    tail_events = [event for event in current_events if event.seq > model_completed_seq]
    requested_payloads, completed_payloads = collect_tool_payloads(tail_events)

    if len(requested_payloads) > len(expected_tool_calls):
        raise ReplayConsistencyError(
            "Event log contains more tool_requested events than model emitted tool calls."
        )
    if len(completed_payloads) > len(requested_payloads):
        raise ReplayConsistencyError(
            "Event log contains more tool_completed events than tool_requested events."
        )

    for index, tool_call in enumerate(expected_tool_calls):
        if tool_call.tool_name not in allowed_tool_names:
            raise CapabilityDeniedError(
                f"Tool {tool_call.tool_name!r} is not allowed for tenant {tenant.tenant_id!r}."
            )

        if index < len(requested_payloads):
            existing_request = requested_payloads[index]
            if (
                existing_request.tool_name != tool_call.tool_name
                or existing_request.arguments_json != tool_call.arguments_json
            ):
                raise ReplayConsistencyError(
                    "Tool request payload does not match the model-emitted tool call."
                )
            idempotency_key = existing_request.idempotency_key
            request_event_id: str | None = None
            tool_version = existing_request.tool_version
            schema_version = existing_request.schema_version
        else:
            next_seq = current_events[-1].seq + 1 if current_events else 1
            idempotency_key = derive_idempotency_key(
                run_id=run_id,
                seq=next_seq,
            )
            request_event = await store.append_event(
                run_id=run_id,
                tenant_id=tenant.tenant_id,
                event_type="tool_requested",
                payload=ToolRequestedPayload(
                    tool_name=tool_call.tool_name,
                    arguments_json=tool_call.arguments_json,
                    idempotency_key=idempotency_key,
                ),
            )
            current_events.append(request_event)
            request_event_id = request_event.event_id
            tool_version = "1.0.0"
            schema_version = "1"

        if index < len(completed_payloads):
            existing_completed = completed_payloads[index]
            if existing_completed.tool_name != tool_call.tool_name:
                raise ReplayConsistencyError(
                    "Tool completion payload does not match the model-emitted tool call."
                )
            if existing_completed.outcome != "success":
                raise ToolExecutionFailedError(
                    "Tool "
                    f"{tool_call.tool_name!r} failed with outcome={existing_completed.outcome!r}."
                )
            continue

        await complete_pending_tool_request(
            store=store,
            tool_port=tool_port,
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            tool_name=tool_call.tool_name,
            arguments_json=tool_call.arguments_json,
            idempotency_key=idempotency_key,
            request_event_id=request_event_id,
            tool_version=tool_version,
            schema_version=schema_version,
        )
        current_events = await store.get_events_for_run(run_id)


async def complete_pending_tool_request(
    *,
    store: EventStore,
    tool_port: ToolPort,
    run_id: str,
    tenant_id: str,
    tool_name: str,
    arguments_json: str,
    idempotency_key: str,
    request_event_id: str | None,
    tool_version: str,
    schema_version: str,
) -> str:
    try:
        tool_result = await tool_port.call(
            tool_name=tool_name,
            arguments_json=arguments_json,
            context=ToolExecutionContext(
                run_id=run_id,
                tenant_id=tenant_id,
                idempotency_key=idempotency_key,
                request_event_id=request_event_id,
                tool_version=tool_version,
                schema_version=schema_version,
            ),
        )
    except ToolUnknownOutcomeError:
        raise

    await append_tool_completed_event(
        store=store,
        run_id=run_id,
        tenant_id=tenant_id,
        tool_name=tool_name,
        result=tool_result,
    )
    if tool_result.outcome != "success":
        raise ToolExecutionFailedError(
            f"Tool {tool_name!r} failed with outcome={tool_result.outcome!r}."
        )
    return tool_result.result_json


async def append_tool_completed_event(
    *,
    store: EventStore,
    run_id: str,
    tenant_id: str,
    tool_name: str,
    result: ToolExecutionResult,
) -> None:
    await store.append_event(
        run_id=run_id,
        tenant_id=tenant_id,
        event_type="tool_completed",
        payload=ToolCompletedPayload(
            tool_name=tool_name,
            result_json=result.result_json,
            outcome=result.outcome,
            received_idempotency_key=result.received_idempotency_key,
            effect_id=result.effect_id,
            request_id=result.request_id,
            error_message=result.error_message,
        ),
    )


def derive_idempotency_key(*, run_id: str, seq: int) -> str:
    token = f"{run_id}:{seq}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
