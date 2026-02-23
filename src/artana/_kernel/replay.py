from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

from artana._kernel.types import OutputT, ReplayConsistencyError
from artana.events import (
    ChatMessage,
    EventType,
    KernelEvent,
    ModelCompletedPayload,
    ModelRequestedPayload,
    compute_allowed_tools_hash,
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
    messages: tuple[ChatMessage, ...],
    model: str,
    allowed_tool_names: list[str],
    step_key: str | None = None,
) -> tuple[KernelEvent | None, KernelEvent | None]:
    expected_messages = list(messages)
    expected_allowed_tools = sorted(allowed_tool_names)
    expected_allowed_tools_hash = compute_allowed_tools_hash(expected_allowed_tools)
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        if event.event_type != EventType.MODEL_REQUESTED:
            continue
        payload = event.payload
        if not isinstance(payload, ModelRequestedPayload):
            continue
        if payload.model != model or payload.prompt != prompt:
            continue
        if payload.messages != expected_messages:
            continue
        if payload.step_key != step_key:
            continue
        payload_allowed_tools = sorted(payload.allowed_tools)
        payload_allowed_tools_hash = payload.allowed_tools_hash
        if payload_allowed_tools_hash is not None:
            if payload_allowed_tools_hash != expected_allowed_tools_hash:
                raise ReplayConsistencyError(
                    "Cannot resume run with changed allowed tools for the same model request."
                )
        if payload_allowed_tools != expected_allowed_tools:
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
        if event.event_type == EventType.MODEL_REQUESTED:
            break
        if event.event_type == EventType.MODEL_COMPLETED:
            return event
    return None


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
