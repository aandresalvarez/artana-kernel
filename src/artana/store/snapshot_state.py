from __future__ import annotations

import json
from datetime import datetime, timezone

from artana.events import (
    EventType,
    HarnessFailedPayload,
    HarnessSleepPayload,
    HarnessStagePayload,
    KernelEvent,
    ModelCompletedPayload,
    PauseRequestedPayload,
    ReplayedWithDriftPayload,
    RunSummaryPayload,
    ToolCompletedPayload,
)
from artana.store.base import RunStateSnapshotRecord


def initialize_run_state_snapshot(*, event: KernelEvent) -> RunStateSnapshotRecord:
    base = RunStateSnapshotRecord(
        run_id=event.run_id,
        tenant_id=event.tenant_id,
        last_event_seq=0,
        last_event_type="",
        updated_at=_coerce_timestamp(event.timestamp),
        status="active",
        blocked_on=None,
        failure_reason=None,
        last_step_key=None,
        drift_count=0,
        last_stage=None,
        last_tool=None,
        model_cost_total=0.0,
        open_pause_count=0,
        explain_status="completed",
        explain_failure_reason=None,
        explain_failure_step=None,
    )
    return apply_event_to_run_state_snapshot(snapshot=base, event=event)


def apply_event_to_run_state_snapshot(
    *,
    snapshot: RunStateSnapshotRecord,
    event: KernelEvent,
) -> RunStateSnapshotRecord:
    if event.run_id != snapshot.run_id:
        raise ValueError(
            f"Snapshot run_id mismatch: {snapshot.run_id!r} vs event {event.run_id!r}."
        )
    if event.tenant_id != snapshot.tenant_id:
        raise ValueError(
            "Snapshot tenant_id mismatch: "
            f"{snapshot.tenant_id!r} vs event {event.tenant_id!r}."
        )

    next_step_key = _next_step_key(current=snapshot.last_step_key, event=event)
    status = snapshot.status
    blocked_on = snapshot.blocked_on
    failure_reason = snapshot.failure_reason
    drift_count = snapshot.drift_count
    last_stage = snapshot.last_stage
    last_tool = snapshot.last_tool
    model_cost_total = snapshot.model_cost_total
    open_pause_count = snapshot.open_pause_count
    explain_status = snapshot.explain_status
    explain_failure_reason = snapshot.explain_failure_reason
    explain_failure_step = snapshot.explain_failure_step

    if event.event_type == EventType.RUN_STARTED:
        status = "active"
        blocked_on = None
        failure_reason = None

    elif event.event_type == EventType.PAUSE_REQUESTED and isinstance(
        event.payload, PauseRequestedPayload
    ):
        open_pause_count = max(0, open_pause_count) + 1
        status = "paused"
        blocked_on = _pause_blocked_on(event.payload)
        failure_reason = None

    elif event.event_type == EventType.RESUME_REQUESTED:
        if open_pause_count > 0:
            open_pause_count -= 1
        if open_pause_count == 0:
            blocked_on = None
            if status not in {"failed", "completed"}:
                status = "active"
        else:
            status = "paused"

    elif event.event_type == EventType.HARNESS_FAILED and isinstance(
        event.payload, HarnessFailedPayload
    ):
        status = "failed"
        failure_reason = event.payload.error_type
        explain_status = "failed"
        explain_failure_reason = event.payload.error_type
        explain_failure_step = event.payload.last_step_key

    elif event.event_type == EventType.HARNESS_SLEEP and isinstance(
        event.payload, HarnessSleepPayload
    ):
        if event.payload.status == "failed":
            status = "failed"
            failure_reason = event.payload.execution_error_type or event.payload.sleep_error_type
            explain_status = "failed"
        else:
            if open_pause_count == 0:
                status = "completed"
                blocked_on = None
            else:
                status = "paused"
            failure_reason = None
            explain_status = "completed"

    elif event.event_type == EventType.MODEL_COMPLETED and isinstance(
        event.payload, ModelCompletedPayload
    ):
        model_cost_total += event.payload.cost_usd

    elif event.event_type == EventType.REPLAYED_WITH_DRIFT and isinstance(
        event.payload, ReplayedWithDriftPayload
    ):
        drift_count += 1

    elif event.event_type == EventType.HARNESS_STAGE and isinstance(
        event.payload, HarnessStagePayload
    ):
        last_stage = event.payload.stage

    elif event.event_type == EventType.RUN_SUMMARY and isinstance(
        event.payload, RunSummaryPayload
    ):
        if last_stage is None and event.payload.summary_type == "trace::round":
            stage_value = _trace_round_stage(event.payload.summary_json)
            if stage_value is not None:
                last_stage = stage_value

    elif event.event_type == EventType.TOOL_COMPLETED and isinstance(
        event.payload, ToolCompletedPayload
    ):
        last_tool = event.payload.tool_name

    return RunStateSnapshotRecord(
        run_id=snapshot.run_id,
        tenant_id=snapshot.tenant_id,
        last_event_seq=event.seq,
        last_event_type=event.event_type.value,
        updated_at=_coerce_timestamp(event.timestamp),
        status=status,
        blocked_on=blocked_on,
        failure_reason=failure_reason,
        last_step_key=next_step_key,
        drift_count=drift_count,
        last_stage=last_stage,
        last_tool=last_tool,
        model_cost_total=model_cost_total,
        open_pause_count=open_pause_count,
        explain_status=explain_status,
        explain_failure_reason=explain_failure_reason,
        explain_failure_step=explain_failure_step,
    )


def _pause_blocked_on(payload: PauseRequestedPayload) -> str | None:
    if payload.context_json is None:
        return None
    try:
        context_payload: object = json.loads(payload.context_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(context_payload, dict):
        return None
    approval_key = context_payload.get("approval_key")
    if isinstance(approval_key, str) and approval_key != "":
        return f"approval:{approval_key}"
    unblock_key = context_payload.get("unblock_key")
    if isinstance(unblock_key, str) and unblock_key != "":
        return f"unblock:{unblock_key}"
    return None


def _next_step_key(*, current: str | None, event: KernelEvent) -> str | None:
    payload_step_key = getattr(event.payload, "step_key", None)
    if isinstance(payload_step_key, str):
        return payload_step_key
    if isinstance(event.parent_step_key, str):
        return event.parent_step_key
    return current


def _trace_round_stage(summary_json: str) -> str | None:
    try:
        parsed: object = json.loads(summary_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    stage_obj = parsed.get("stage")
    if isinstance(stage_obj, str):
        return stage_obj
    return None


def _coerce_timestamp(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)

