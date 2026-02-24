from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from artana._kernel.policies import apply_prepare_tool_result_middleware
from artana._kernel.types import ReplayConsistencyError, ToolExecutionFailedError
from artana.events import EventPayload, EventType, KernelEvent, ToolCompletedPayload
from artana.json_utils import sha256_hex
from artana.middleware.base import KernelMiddleware
from artana.models import TenantContext
from artana.ports.tool import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolPort,
    ToolUnknownOutcomeError,
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


@dataclass(frozen=True, slots=True)
class ToolCompletionResult:
    result_json: str
    seq: int


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
    middleware: Sequence[KernelMiddleware],
    tenant: TenantContext,
    tenant_capabilities: frozenset[str] = frozenset(),
    tenant_budget_usd_limit: float | None = None,
    parent_step_key: str | None = None,
) -> ToolCompletionResult:
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
                tenant_capabilities=tenant_capabilities,
                tenant_budget_usd_limit=tenant_budget_usd_limit,
            ),
        )
    except ToolUnknownOutcomeError as exc:
        await append_tool_completed_event(
            store=store,
            run_id=run_id,
            tenant_id=tenant_id,
            parent_step_key=parent_step_key,
            tool_name=tool_name,
            result=ToolExecutionResult(
                outcome="unknown_outcome",
                result_json="",
                received_idempotency_key=idempotency_key,
                request_id=request_event_id,
                error_message=str(exc),
            ),
            request_event_id=request_event_id,
        )
        raise ToolExecutionFailedError(
            f"Tool {tool_name!r} ended with unknown outcome and requires reconciliation."
        ) from exc

    prepared_result_json = await apply_prepare_tool_result_middleware(
        middleware,
        run_id=run_id,
        tenant=tenant,
        tool_name=tool_name,
        result_json=tool_result.result_json,
    )
    tool_result = replace(tool_result, result_json=prepared_result_json)

    completed_seq = await append_tool_completed_event(
        store=store,
        run_id=run_id,
        tenant_id=tenant_id,
        parent_step_key=parent_step_key,
        tool_name=tool_name,
        result=tool_result,
        request_event_id=request_event_id,
    )
    if tool_result.outcome != "success":
        raise ToolExecutionFailedError(
            f"Tool {tool_name!r} failed with outcome={tool_result.outcome!r}."
        )
    return ToolCompletionResult(result_json=tool_result.result_json, seq=completed_seq)


async def append_tool_completed_event(
    *,
    store: EventStore,
    run_id: str,
    tenant_id: str,
    tool_name: str,
    result: ToolExecutionResult,
    request_event_id: str | None = None,
    parent_step_key: str | None = None,
) -> int:
    request_id = result.request_id if result.request_id is not None else request_event_id
    event = await _append_event_with_parent(
        store=store,
        run_id=run_id,
        tenant_id=tenant_id,
        event_type=EventType.TOOL_COMPLETED,
        parent_step_key=parent_step_key,
        payload=ToolCompletedPayload(
            tool_name=tool_name,
            result_json=result.result_json,
            outcome=result.outcome,
            received_idempotency_key=result.received_idempotency_key,
            effect_id=result.effect_id,
            request_id=request_id,
            error_message=result.error_message,
        ),
    )
    return event.seq


async def mark_pending_request_unknown(
    *,
    store: EventStore,
    run_id: str,
    tenant_id: str,
    tool_name: str,
    idempotency_key: str,
    request_event_id: str | None,
    parent_step_key: str | None = None,
) -> None:
    await append_tool_completed_event(
        store=store,
        run_id=run_id,
        tenant_id=tenant_id,
        parent_step_key=parent_step_key,
        tool_name=tool_name,
        result=ToolExecutionResult(
            outcome="unknown_outcome",
            result_json="",
            received_idempotency_key=idempotency_key,
            request_id=request_event_id,
            error_message=(
                "Pending tool request found without completion event. "
                "Reconciliation is required before retry."
            ),
        ),
        request_event_id=request_event_id,
    )


def resolve_completed_tool_result(
    *,
    expected_tool_name: str,
    tool_name_from_completion: str,
    outcome: str,
    result_json: str,
) -> str:
    if tool_name_from_completion != expected_tool_name:
        raise ReplayConsistencyError(
            "Tool completion payload does not match requested/model-emitted tool call."
        )
    if outcome != "success":
        raise ToolExecutionFailedError(
            f"Tool {expected_tool_name!r} previously failed with outcome={outcome!r}."
        )
    return result_json


def derive_idempotency_key(
    *,
    run_id: str,
    tool_name: str,
    seq: int,
) -> str:
    token = f"{run_id}:{tool_name}:{seq}"
    return sha256_hex(token)
