from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Literal

from artana._kernel.types import OutputT, ReplayConsistencyError, RunResumeState
from artana.events import (
    KernelEvent,
    ModelCompletedPayload,
    ModelRequestedPayload,
    PauseRequestedPayload,
    ToolCompletedPayload,
    ToolRequestedPayload,
    WorkflowStepCompletedPayload,
    WorkflowStepRequestedPayload,
)
from artana.models import TenantContext
from artana.ports.model import ModelUsage, ToolCall


@dataclass(frozen=True, slots=True)
class ModelStepResult(Generic[OutputT]):
    completed_seq: int
    output: OutputT
    usage: ModelUsage
    tool_calls: tuple[ToolCall, ...]
    replayed: bool


def deserialize_model_completed(
    *,
    event: KernelEvent,
    output_schema: type[OutputT],
    replayed: bool,
) -> ModelStepResult[OutputT]:
    payload = event.payload
    if not isinstance(payload, ModelCompletedPayload):
        raise ReplayConsistencyError(
            f"Expected model_completed payload at seq={event.seq}, got {type(payload)!r}."
        )
    output = output_schema.model_validate_json(payload.output_json)
    return ModelStepResult(
        completed_seq=event.seq,
        output=output,
        usage=ModelUsage(
            prompt_tokens=payload.prompt_tokens,
            completion_tokens=payload.completion_tokens,
            cost_usd=payload.cost_usd,
        ),
        tool_calls=tuple(
            ToolCall(
                tool_name=tool_call.tool_name,
                arguments_json=tool_call.arguments_json,
            )
            for tool_call in payload.tool_calls
        ),
        replayed=replayed,
    )


def find_matching_model_cycle(
    *,
    events: Sequence[KernelEvent],
    prompt: str,
    model: str,
    allowed_tool_names: list[str],
) -> tuple[KernelEvent | None, KernelEvent | None]:
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        if event.event_type != "model_requested":
            continue
        payload = event.payload
        if not isinstance(payload, ModelRequestedPayload):
            continue
        if payload.model != model or payload.prompt != prompt:
            continue
        if payload.allowed_tools != allowed_tool_names:
            raise ReplayConsistencyError(
                "Cannot resume run with changed allowed tools for the same model request."
            )
        completed = find_model_completed_after(events=events, start_index=index + 1)
        return event, completed
    return None, None


def find_model_completed_after(
    *, events: Sequence[KernelEvent], start_index: int
) -> KernelEvent | None:
    for event in events[start_index:]:
        if event.event_type == "model_requested":
            break
        if event.event_type == "model_completed":
            return event
    return None


def collect_tool_payloads(
    events: Sequence[KernelEvent],
) -> tuple[list[ToolRequestedPayload], list[ToolCompletedPayload]]:
    requested_payloads: list[ToolRequestedPayload] = []
    completed_payloads: list[ToolCompletedPayload] = []

    for event in events:
        if event.event_type == "tool_requested":
            if not isinstance(event.payload, ToolRequestedPayload):
                raise ReplayConsistencyError(
                    f"Expected ToolRequestedPayload at seq={event.seq}."
                )
            requested_payloads.append(event.payload)
        if event.event_type == "tool_completed":
            if not isinstance(event.payload, ToolCompletedPayload):
                raise ReplayConsistencyError(
                    f"Expected ToolCompletedPayload at seq={event.seq}."
                )
            completed_payloads.append(event.payload)
    return requested_payloads, completed_payloads


def validate_tenant_for_run(*, events: Sequence[KernelEvent], tenant: TenantContext) -> None:
    if not events:
        return
    expected_tenant_id = events[0].tenant_id
    if expected_tenant_id != tenant.tenant_id:
        raise ReplayConsistencyError(
            "Run tenant mismatch. "
            f"run tenant={expected_tenant_id!r}, request tenant={tenant.tenant_id!r}."
        )
    for event in events:
        if event.tenant_id != expected_tenant_id:
            raise ReplayConsistencyError(
                f"Corrupted run: mixed tenants found in run events for run_id={event.run_id!r}."
            )


def derive_run_resume_state(events: Sequence[KernelEvent]) -> RunResumeState:
    last_event = events[-1]
    pause_reason = _extract_latest_pause_reason(events)
    pending_tool = _derive_pending_tool(events)
    status: Literal["paused", "pending_tool", "ready", "complete"] = "complete"

    if last_event.event_type == "pause_requested":
        status = "paused"
    elif pending_tool is not None:
        status = "pending_tool"
    elif _has_pending_workflow_step(events):
        status = "ready"
    elif last_event.event_type == "model_requested":
        status = "ready"
    elif _latest_model_completed_has_unrequested_tools(events):
        status = "ready"

    return RunResumeState(
        run_id=last_event.run_id,
        status=status,
        last_seq=last_event.seq,
        pause_reason=pause_reason,
        pending_tool=pending_tool,
    )


def _extract_latest_pause_reason(events: Sequence[KernelEvent]) -> str | None:
    for event in reversed(events):
        if event.event_type != "pause_requested":
            continue
        payload = event.payload
        if isinstance(payload, PauseRequestedPayload):
            return payload.reason
    return None


def _derive_pending_tool(events: Sequence[KernelEvent]) -> ToolCall | None:
    latest_completed = _latest_model_completed_event(events)
    if latest_completed is None:
        requested_payloads, completed_payloads = collect_tool_payloads(events)
    else:
        requested_payloads, completed_payloads = collect_tool_payloads(
            [event for event in events if event.seq > latest_completed.seq]
        )

    if len(completed_payloads) < len(requested_payloads):
        pending_requested = requested_payloads[len(completed_payloads)]
        return ToolCall(
            tool_name=pending_requested.tool_name,
            arguments_json=pending_requested.arguments_json,
        )
    return None


def _latest_model_completed_has_unrequested_tools(events: Sequence[KernelEvent]) -> bool:
    latest_completed = _latest_model_completed_event(events)
    if latest_completed is None:
        return False
    payload = latest_completed.payload
    if not isinstance(payload, ModelCompletedPayload):
        return False
    expected_tool_calls = payload.tool_calls
    tail_events = [event for event in events if event.seq > latest_completed.seq]
    requested_payloads, _ = collect_tool_payloads(tail_events)
    return len(requested_payloads) < len(expected_tool_calls)


def _latest_model_completed_event(events: Sequence[KernelEvent]) -> KernelEvent | None:
    for event in reversed(events):
        if event.event_type == "model_completed":
            return event
    return None


def _has_pending_workflow_step(events: Sequence[KernelEvent]) -> bool:
    requested_indices: set[int] = set()
    completed_indices: set[int] = set()

    for event in events:
        if event.event_type == "workflow_step_requested":
            payload = event.payload
            if not isinstance(payload, WorkflowStepRequestedPayload):
                raise ReplayConsistencyError(
                    f"Expected WorkflowStepRequestedPayload at seq={event.seq}."
                )
            requested_indices.add(payload.step_index)
        if event.event_type == "workflow_step_completed":
            payload = event.payload
            if not isinstance(payload, WorkflowStepCompletedPayload):
                raise ReplayConsistencyError(
                    f"Expected WorkflowStepCompletedPayload at seq={event.seq}."
                )
            completed_indices.add(payload.step_index)

    return any(index not in completed_indices for index in requested_indices)
