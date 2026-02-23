from __future__ import annotations

from dataclasses import dataclass

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
    ToolExecutionFailedError,
)
from artana.events import EventType, ToolRequestedPayload
from artana.models import TenantContext
from artana.ports.tool import ToolPort
from artana.store.base import EventStore


@dataclass(frozen=True, slots=True)
class ToolStepReplayResult:
    result_json: str
    seq: int
    replayed: bool


async def execute_tool_step_with_replay(
    *,
    store: EventStore,
    tool_port: ToolPort,
    run_id: str,
    tenant: TenantContext,
    tool_name: str,
    arguments_json: str,
    step_key: str | None = None,
) -> ToolStepReplayResult:
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
        if not _matches_tool_request(
            requested=requested,
            tool_name=tool_name,
            arguments_json=arguments_json,
            step_key=step_key,
        ):
            continue
        completion = resolution.completion
        if completion is not None:
            replay_result = resolve_completed_tool_result(
                expected_tool_name=tool_name,
                tool_name_from_completion=completion.payload.tool_name,
                outcome=completion.payload.outcome,
                result_json=completion.payload.result_json,
            )
            return ToolStepReplayResult(
                result_json=replay_result,
                seq=completion.seq,
                replayed=True,
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

    idempotency_key = derive_idempotency_key(
        run_id=run_id,
        tool_name=tool_name,
        arguments_json=arguments_json,
        step_key=step_key,
    )
    request_event = await store.append_event(
        run_id=run_id,
        tenant_id=tenant.tenant_id,
        event_type=EventType.TOOL_REQUESTED,
        payload=ToolRequestedPayload(
            tool_name=tool_name,
            arguments_json=arguments_json,
            idempotency_key=idempotency_key,
            step_key=step_key,
        ),
    )
    completed = await complete_pending_tool_request(
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
        tenant_capabilities=tenant.capabilities,
        tenant_budget_usd_limit=tenant.budget_usd_limit,
    )
    return ToolStepReplayResult(
        result_json=completed.result_json,
        seq=completed.seq,
        replayed=False,
    )


async def reconcile_tool_with_replay(
    *,
    store: EventStore,
    tool_port: ToolPort,
    run_id: str,
    tenant: TenantContext,
    tool_name: str,
    arguments_json: str,
    step_key: str | None = None,
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
        if not _matches_tool_request(
            requested=requested,
            tool_name=tool_name,
            arguments_json=arguments_json,
            step_key=step_key,
        ):
            continue

        completion = resolution.completion
        if completion is None:
            raise ToolExecutionFailedError(
                f"Tool {tool_name!r} has no completion event to reconcile."
            )
        if completion.payload.outcome == "success":
            return completion.payload.result_json
        if completion.payload.outcome != "unknown_outcome":
            raise ToolExecutionFailedError(
                "Tool "
                f"{tool_name!r} cannot be reconciled from outcome={completion.payload.outcome!r}."
            )
        completed = await complete_pending_tool_request(
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
            tenant_capabilities=tenant.capabilities,
            tenant_budget_usd_limit=tenant.budget_usd_limit,
        )
        return completed.result_json

    raise ValueError(
        f"No tool request found for tool_name={tool_name!r} and the provided arguments_json."
    )


def _matches_tool_request(
    *,
    requested: ToolRequestedPayload,
    tool_name: str,
    arguments_json: str,
    step_key: str | None,
) -> bool:
    if requested.tool_name != tool_name:
        return False
    if requested.arguments_json != arguments_json:
        return False
    if step_key is None:
        return requested.step_key is None
    return requested.step_key == step_key
