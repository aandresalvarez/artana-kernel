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
                raise ReplayConsistencyError(
                    "tool_completed payload is missing request_id; "
                    "legacy completions are unsupported."
                )
            completions_by_request_id[payload.request_id] = completion_record

    requested_ids = {record.event_id for record in requested}
    dangling_completion_ids = set(completions_by_request_id) - requested_ids
    if dangling_completion_ids:
        raise ReplayConsistencyError(
            "Found tool_completed event with request_id that does not map to tool_requested."
        )

    resolutions: list[ToolResolution] = []
    for request_record in requested:
        completion = completions_by_request_id.get(request_record.event_id)
        resolutions.append(
            ToolResolution(
                request=request_record,
                completion=completion,
            )
        )

    return resolutions
