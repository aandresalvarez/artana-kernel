from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, cast

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
from artana.ports.model import (
    ModelAPIModeUsed,
    ModelCallOptions,
    ModelUsage,
    ToolCall,
)


@dataclass(frozen=True, slots=True)
class ModelStepResult(Generic[OutputT]):
    completed_seq: int
    output: OutputT
    usage: ModelUsage
    tool_calls: tuple[ToolCall, ...]
    replayed: bool
    api_mode_used: ModelAPIModeUsed = "chat"
    response_id: str | None = None
    response_output_items: tuple[dict[str, object], ...] = ()
    replayed_with_drift: bool = False
    drift_fields: tuple[str, ...] = ()


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
    drift_fields: tuple[str, ...] = (),
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
        api_mode_used=payload.api_mode_used,
        response_id=payload.response_id,
        response_output_items=tuple(payload.responses_output_items),
        replayed_with_drift=replayed_with_drift,
        drift_fields=drift_fields,
    )


def find_prompt_drift_candidate(
    *,
    events: Sequence[KernelEvent],
    prompt: str,
    messages: tuple[ChatMessage, ...],
    model: str,
    model_options: ModelCallOptions,
    responses_input_items: list[dict[str, object]] | None,
    allowed_tool_signatures: list[ToolSignatureRecord],
    step_key: str | None = None,
) -> PromptDriftCandidate | None:
    if step_key is None:
        return None
    expected_messages = list(messages)
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
            expected_tool_signatures=allowed_tool_signatures,
        )
        prompt_matches = payload.prompt == prompt
        messages_match = payload.messages == expected_messages
        options_drift_fields = _model_options_drift_fields(
            payload=payload,
            model_options=model_options,
            responses_input_items=responses_input_items,
        )
        if prompt_matches and messages_match and len(options_drift_fields) == 0:
            return None
        drift_fields = _drift_fields(
            prompt_matches=prompt_matches,
            messages_match=messages_match,
            options_drift_fields=options_drift_fields,
        )
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
    model_options: ModelCallOptions,
    responses_input_items: list[dict[str, object]] | None,
    allowed_tool_signatures: list[ToolSignatureRecord],
    step_key: str | None = None,
    replay_policy: ReplayPolicy = "strict",
) -> ModelCycleLookup:
    expected_messages = list(messages)

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
            expected_tool_signatures=allowed_tool_signatures,
        )
        prompt_matches = payload.prompt == prompt
        messages_match = payload.messages == expected_messages
        options_drift_fields = _model_options_drift_fields(
            payload=payload,
            model_options=model_options,
            responses_input_items=responses_input_items,
        )
        if prompt_matches and messages_match and len(options_drift_fields) == 0:
            completed = find_model_completed_after(events=events, start_index=index + 1)
            return ModelCycleLookup(request_event=event, completed_event=completed)
        if step_key is None:
            continue

        drift_fields = _drift_fields(
            prompt_matches=prompt_matches,
            messages_match=messages_match,
            options_drift_fields=options_drift_fields,
        )
        if replay_policy == "strict":
            raise ReplayConsistencyError(
                "Cannot resume run with changed model inputs/options for the same model step."
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
    expected_tool_signatures: list[ToolSignatureRecord],
) -> None:
    expected_signatures = _sort_signatures(expected_tool_signatures)
    payload_signatures = _sort_signatures(payload.allowed_tool_signatures)
    expected_signature_tokens = [
        _signature_token(signature) for signature in expected_signatures
    ]
    payload_signature_tokens = [
        _signature_token(signature) for signature in payload_signatures
    ]
    expected_hash = compute_allowed_tools_hash(expected_signature_tokens)
    if payload.allowed_tools_hash != expected_hash:
        raise ReplayConsistencyError(
            "Cannot resume run with changed allowed tool signatures for the same model step."
        )
    if payload_signature_tokens != expected_signature_tokens:
        raise ReplayConsistencyError(
            "Cannot resume run with changed allowed tool signatures for the same model step."
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


def _drift_fields(
    *,
    prompt_matches: bool,
    messages_match: bool,
    options_drift_fields: tuple[str, ...],
) -> tuple[str, ...]:
    fields: list[str] = []
    if not prompt_matches:
        fields.append("prompt")
    if not messages_match:
        fields.append("messages")
    fields.extend(options_drift_fields)
    return tuple(fields)


def _model_options_drift_fields(
    *,
    payload: ModelRequestedPayload,
    model_options: ModelCallOptions,
    responses_input_items: list[dict[str, object]] | None,
) -> tuple[str, ...]:
    fields: list[str] = []
    if payload.api_mode != model_options.api_mode:
        fields.append("api_mode")
    if payload.reasoning_effort != model_options.reasoning_effort:
        fields.append("reasoning_effort")
    if payload.verbosity != model_options.verbosity:
        fields.append("verbosity")
    if payload.previous_response_id != model_options.previous_response_id:
        fields.append("previous_response_id")

    # Compatibility path for legacy rows that don't have responses_input_items.
    if payload.responses_input_items is None:
        return tuple(fields)

    if not _responses_input_equal(payload.responses_input_items, responses_input_items):
        fields.append("responses_input_items")
    return tuple(fields)


def _responses_input_equal(
    stored_items: list[dict[str, object]],
    expected_items: list[dict[str, object]] | None,
) -> bool:
    return _canonicalize_items(stored_items) == _canonicalize_items(expected_items or [])


def _canonicalize_items(items: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        cast(
            dict[str, object],
            json.loads(json.dumps(item, sort_keys=True, separators=(",", ":"))),
        )
        for item in items
    ]
