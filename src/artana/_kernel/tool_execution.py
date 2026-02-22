from __future__ import annotations

import hashlib
from dataclasses import dataclass

from artana._kernel.types import ReplayConsistencyError, ToolExecutionFailedError
from artana.events import EventType, ToolCompletedPayload
from artana.ports.tool import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolPort,
    ToolUnknownOutcomeError,
)
from artana.store.base import EventStore


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
            ),
        )
    except ToolUnknownOutcomeError as exc:
        await append_tool_completed_event(
            store=store,
            run_id=run_id,
            tenant_id=tenant_id,
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

    completed_seq = await append_tool_completed_event(
        store=store,
        run_id=run_id,
        tenant_id=tenant_id,
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
) -> int:
    request_id = result.request_id if result.request_id is not None else request_event_id
    event = await store.append_event(
        run_id=run_id,
        tenant_id=tenant_id,
        event_type=EventType.TOOL_COMPLETED,
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
) -> None:
    await append_tool_completed_event(
        store=store,
        run_id=run_id,
        tenant_id=tenant_id,
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


def derive_idempotency_key(*, run_id: str, seq: int, step_key: str | None = None) -> str:
    token = f"{run_id}:{step_key}" if step_key is not None else f"{run_id}:{seq}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
