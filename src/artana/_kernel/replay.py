from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

from artana._kernel.types import OutputT, ReplayConsistencyError, ReplayPolicy
from artana.events import (
    ChatMessage,
    EventType,
    KernelEvent,
    ModelCompletedPayload,
    ModelRequestedPayload,
    ToolSignatureRecord,
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
    replayed_with_drift: bool = False


@dataclass(frozen=True, slots=True)
class ModelCycleLookup:
    request_event: KernelEvent | None
    completed_event: KernelEvent | None
    replayed_with_drift: bool = False
    drift_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PromptDriftCandidate:
    request_event: KernelEvent
    completed_event: KernelEvent | None
    drift_fields: tuple[str, ...]


def deserialize_model_completed(
    *,
    event: KernelEvent,
    output_schema: type[OutputT],
    replayed: bool,
    replayed_with_drift: bool = False,
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
                tool_call_id=tool_call.tool_call_id,
            )
            for tool_call in payload.tool_calls
        ),
        replayed=replayed,
        replayed_with_drift=replayed_with_drift,
    )


def find_prompt_drift_candidate(
    *,
    events: Sequence[KernelEvent],
    prompt: str,
    messages: tuple[ChatMessage, ...],
    model: str,
    allowed_tool_names: list[str],
    allowed_tool_signatures: list[ToolSignatureRecord],
    step_key: str | None = None,
) -> PromptDriftCandidate | None:
    if step_key is None:
        return None
    expected_messages = list(messages)
    expected_allowed_tools = sorted(allowed_tool_names)
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        if event.event_type != EventType.MODEL_REQUESTED:
            continue
        payload = event.payload
        if not isinstance(payload, ModelRequestedPayload):
            continue
        if payload.model != model:
            continue
        if payload.step_key != step_key:
            continue
        _assert_allowed_tools_compatible(
            payload=payload,
            expected_allowed_tools=expected_allowed_tools,
            expected_tool_signatures=allowed_tool_signatures,
        )
        prompt_matches = payload.prompt == prompt
        messages_match = payload.messages == expected_messages
        if prompt_matches and messages_match:
            return None
        drift_fields = _drift_fields(prompt_matches=prompt_matches, messages_match=messages_match)
        completed = find_model_completed_after(events=events, start_index=index + 1)
        return PromptDriftCandidate(
            request_event=event,
            completed_event=completed,
            drift_fields=drift_fields,
        )
    return None


def find_matching_model_cycle(
    *,
    events: Sequence[KernelEvent],
    prompt: str,
    messages: tuple[ChatMessage, ...],
    model: str,
    allowed_tool_names: list[str],
    allowed_tool_signatures: list[ToolSignatureRecord],
    step_key: str | None = None,
    replay_policy: ReplayPolicy = "strict",
) -> ModelCycleLookup:
    expected_messages = list(messages)
    expected_allowed_tools = sorted(allowed_tool_names)

    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        if event.event_type != EventType.MODEL_REQUESTED:
            continue
        payload = event.payload
        if not isinstance(payload, ModelRequestedPayload):
            continue
        if payload.model != model:
            continue
        if payload.step_key != step_key:
            continue

        _assert_allowed_tools_compatible(
            payload=payload,
            expected_allowed_tools=expected_allowed_tools,
            expected_tool_signatures=allowed_tool_signatures,
        )
        prompt_matches = payload.prompt == prompt
        messages_match = payload.messages == expected_messages
        if prompt_matches and messages_match:
            completed = find_model_completed_after(events=events, start_index=index + 1)
            return ModelCycleLookup(request_event=event, completed_event=completed)
        if step_key is None:
            continue

        drift_fields = _drift_fields(prompt_matches=prompt_matches, messages_match=messages_match)
        if replay_policy == "strict":
            raise ReplayConsistencyError(
                "Cannot resume run with changed prompt/messages for the same model step."
            )
        if replay_policy == "allow_prompt_drift":
            completed = find_model_completed_after(events=events, start_index=index + 1)
            if completed is None:
                # Found drift on the latest request cycle, but no completion to replay.
                return ModelCycleLookup(request_event=None, completed_event=None)
            return ModelCycleLookup(
                request_event=event,
                completed_event=completed,
                replayed_with_drift=True,
                drift_fields=drift_fields,
            )
        raise ReplayConsistencyError(
            "fork_on_drift must fork the run before replay matching executes."
        )
    return ModelCycleLookup(request_event=None, completed_event=None)


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


def _assert_allowed_tools_compatible(
    *,
    payload: ModelRequestedPayload,
    expected_allowed_tools: list[str],
    expected_tool_signatures: list[ToolSignatureRecord],
) -> None:
    expected_signatures = _sort_signatures(expected_tool_signatures)
    payload_signatures = _sort_signatures(payload.allowed_tool_signatures)
    if payload_signatures:
        expected_signature_tokens = [
            _signature_token(signature) for signature in expected_signatures
        ]
        payload_signature_tokens = [
            _signature_token(signature) for signature in payload_signatures
        ]
        expected_hash = compute_allowed_tools_hash(expected_signature_tokens)
        if payload.allowed_tools_hash is not None and payload.allowed_tools_hash != expected_hash:
            raise ReplayConsistencyError(
                "Cannot resume run with changed allowed tool signatures for the same model step."
            )
        if payload_signature_tokens != expected_signature_tokens:
            raise ReplayConsistencyError(
                "Cannot resume run with changed allowed tool signatures for the same model step."
            )
        return

    payload_allowed_tools = sorted(payload.allowed_tools)
    expected_allowed_tools_hash = compute_allowed_tools_hash(expected_allowed_tools)
    if payload.allowed_tools_hash is not None:
        if payload.allowed_tools_hash != expected_allowed_tools_hash:
            raise ReplayConsistencyError(
                "Cannot resume run with changed allowed tools for the same model step."
            )
    if payload_allowed_tools != expected_allowed_tools:
        raise ReplayConsistencyError(
            "Cannot resume run with changed allowed tools for the same model step."
        )


def _sort_signatures(signatures: Sequence[ToolSignatureRecord]) -> list[ToolSignatureRecord]:
    return sorted(
        signatures,
        key=lambda signature: (
            signature.name,
            signature.tool_version,
            signature.schema_version,
            signature.schema_hash,
        ),
    )


def _signature_token(signature: ToolSignatureRecord) -> str:
    return (
        f"{signature.name}|{signature.tool_version}|"
        f"{signature.schema_version}|{signature.schema_hash}"
    )


def _drift_fields(*, prompt_matches: bool, messages_match: bool) -> tuple[str, ...]:
    fields: list[str] = []
    if not prompt_matches:
        fields.append("prompt")
    if not messages_match:
        fields.append("messages")
    return tuple(fields)
