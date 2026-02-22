from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from artana._kernel.types import ReplayConsistencyError
from artana.events import KernelEvent, ToolCompletedPayload, ToolRequestedPayload


@dataclass(frozen=True, slots=True)
class ToolRequestRecord:
    event_id: str
    payload: ToolRequestedPayload


@dataclass(frozen=True, slots=True)
class ToolResolution:
    request: ToolRequestRecord
    completion: ToolCompletedPayload | None


def resolve_tool_resolutions(events: Sequence[KernelEvent]) -> list[ToolResolution]:
    requested: list[ToolRequestRecord] = []
    completions_by_request_id: dict[str, ToolCompletedPayload] = {}
    legacy_completions: list[ToolCompletedPayload] = []

    for event in events:
        if event.event_type == "tool_requested":
            payload = event.payload
            if not isinstance(payload, ToolRequestedPayload):
                raise ReplayConsistencyError(
                    f"Expected ToolRequestedPayload at seq={event.seq}."
                )
            requested.append(ToolRequestRecord(event_id=event.event_id, payload=payload))
        if event.event_type == "tool_completed":
            payload = event.payload
            if not isinstance(payload, ToolCompletedPayload):
                raise ReplayConsistencyError(
                    f"Expected ToolCompletedPayload at seq={event.seq}."
                )
            if payload.request_id is None:
                legacy_completions.append(payload)
            else:
                completions_by_request_id[payload.request_id] = payload

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
