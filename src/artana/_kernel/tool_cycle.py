from __future__ import annotations

from collections.abc import Sequence

from artana._kernel.policies import assert_tool_allowed_for_tenant
from artana._kernel.replay import validate_tenant_for_run
from artana._kernel.tool_execution import (
    complete_pending_tool_request,
    derive_idempotency_key,
    mark_pending_request_unknown,
    resolve_completed_tool_result,
)
from artana._kernel.tool_state import resolve_tool_resolutions
from artana._kernel.types import (
    CapabilityDeniedError,
    ReplayConsistencyError,
    ToolExecutionFailedError,
)
from artana.events import ToolRequestedPayload
from artana.models import TenantContext
from artana.ports.model import ToolCall
from artana.ports.tool import ToolPort
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

    resolutions = resolve_tool_resolutions(events)
    for resolution in reversed(resolutions):
        requested = resolution.request.payload
        if requested.tool_name != tool_name:
            continue
        if requested.arguments_json != arguments_json:
            continue
        completion = resolution.completion
        if completion is not None:
            return resolve_completed_tool_result(
                expected_tool_name=tool_name,
                tool_name_from_completion=completion.tool_name,
                outcome=completion.outcome,
                result_json=completion.result_json,
            )

        await mark_pending_request_unknown(
            store=store,
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            tool_name=tool_name,
            idempotency_key=requested.idempotency_key,
            request_event_id=resolution.request.event_id,
        )
        raise ToolExecutionFailedError(
            f"Tool {tool_name!r} has an unresolved pending request and requires reconciliation."
        )

    events = await store.get_events_for_run(run_id)
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
    resolutions = resolve_tool_resolutions(tail_events)

    if len(resolutions) > len(expected_tool_calls):
        raise ReplayConsistencyError(
            "Event log contains more tool_requested events than model emitted tool calls."
        )

    for index, tool_call in enumerate(expected_tool_calls):
        if tool_call.tool_name not in allowed_tool_names:
            raise CapabilityDeniedError(
                f"Tool {tool_call.tool_name!r} is not allowed for tenant {tenant.tenant_id!r}."
            )

        if index < len(resolutions):
            resolution = resolutions[index]
            requested = resolution.request.payload
            if (
                requested.tool_name != tool_call.tool_name
                or requested.arguments_json != tool_call.arguments_json
            ):
                raise ReplayConsistencyError(
                    "Tool request payload does not match the model-emitted tool call."
                )
            idempotency_key = requested.idempotency_key
            request_event_id = resolution.request.event_id
            tool_version = requested.tool_version
            schema_version = requested.schema_version
            completion = resolution.completion
        else:
            next_seq = current_events[-1].seq + 1 if current_events else 1
            idempotency_key = derive_idempotency_key(run_id=run_id, seq=next_seq)
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
            completion = None

        if completion is not None:
            resolve_completed_tool_result(
                expected_tool_name=tool_call.tool_name,
                tool_name_from_completion=completion.tool_name,
                outcome=completion.outcome,
                result_json=completion.result_json,
            )
            continue

        if index < len(resolutions):
            await mark_pending_request_unknown(
                store=store,
                run_id=run_id,
                tenant_id=tenant.tenant_id,
                tool_name=tool_call.tool_name,
                idempotency_key=idempotency_key,
                request_event_id=request_event_id,
            )
            raise ToolExecutionFailedError(
                "Tool "
                f"{tool_call.tool_name!r} has an unresolved pending request and requires "
                "reconciliation."
            )

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


async def reconcile_tool_with_replay(
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

    resolutions = resolve_tool_resolutions(events)
    for resolution in reversed(resolutions):
        requested = resolution.request.payload
        if requested.tool_name != tool_name:
            continue
        if requested.arguments_json != arguments_json:
            continue

        completion = resolution.completion
        if completion is None:
            raise ToolExecutionFailedError(
                f"Tool {tool_name!r} has no completion event to reconcile."
            )
        if completion.outcome == "success":
            return completion.result_json
        if completion.outcome != "unknown_outcome":
            raise ToolExecutionFailedError(
                f"Tool {tool_name!r} cannot be reconciled from outcome={completion.outcome!r}."
            )
        return await complete_pending_tool_request(
            store=store,
            tool_port=tool_port,
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            tool_name=tool_name,
            arguments_json=arguments_json,
            idempotency_key=requested.idempotency_key,
            request_event_id=resolution.request.event_id,
            tool_version=requested.tool_version,
            schema_version=requested.schema_version,
        )

    raise ValueError(
        f"No tool request found for tool_name={tool_name!r} and the provided arguments_json."
    )
