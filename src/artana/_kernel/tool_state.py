from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from artana._kernel.types import ReplayConsistencyError
from artana.events import EventType, KernelEvent, ToolCompletedPayload, ToolRequestedPayload


@dataclass(frozen=True, slots=True)
class ToolRequestRecord:
    event_id: str
    seq: int
    payload: ToolRequestedPayload


@dataclass(frozen=True, slots=True)
class ToolCompletionRecord:
    event_id: str
    seq: int
    payload: ToolCompletedPayload


@dataclass(frozen=True, slots=True)
class ToolResolution:
    request: ToolRequestRecord
    completion: ToolCompletionRecord | None


def resolve_tool_resolutions(events: Sequence[KernelEvent]) -> list[ToolResolution]:
    requested: list[ToolRequestRecord] = []
    completions_by_request_id: dict[str, ToolCompletionRecord] = {}
    legacy_completions: list[ToolCompletionRecord] = []

    for event in events:
        if event.event_type == EventType.TOOL_REQUESTED:
            payload = event.payload
            if not isinstance(payload, ToolRequestedPayload):
                raise ReplayConsistencyError(
                    f"Expected ToolRequestedPayload at seq={event.seq}."
                )
            requested.append(
                ToolRequestRecord(event_id=event.event_id, seq=event.seq, payload=payload)
            )
        if event.event_type == EventType.TOOL_COMPLETED:
            payload = event.payload
            if not isinstance(payload, ToolCompletedPayload):
                raise ReplayConsistencyError(
                    f"Expected ToolCompletedPayload at seq={event.seq}."
                )
            completion_record = ToolCompletionRecord(
                event_id=event.event_id,
                seq=event.seq,
                payload=payload,
            )
            if payload.request_id is None:
                legacy_completions.append(completion_record)
            else:
                completions_by_request_id[payload.request_id] = completion_record

    requested_ids = {record.event_id for record in requested}
    dangling_completion_ids = set(completions_by_request_id) - requested_ids
    if dangling_completion_ids:
        raise ReplayConsistencyError(
            "Found tool_completed event with request_id that does not map to tool_requested."
        )

    resolutions: list[ToolResolution] = []
    legacy_index = 0
    for request_record in requested:
        completion = completions_by_request_id.get(request_record.event_id)
        if completion is None and legacy_index < len(legacy_completions):
            completion = legacy_completions[legacy_index]
            legacy_index += 1
        resolutions.append(
            ToolResolution(
                request=request_record,
                completion=completion,
            )
        )

    if legacy_index != len(legacy_completions):
        raise ReplayConsistencyError(
            "Found legacy tool_completed events that exceed tool_requested events."
        )

    return resolutions
