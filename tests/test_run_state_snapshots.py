from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.events import (
    EventPayload,
    EventType,
    HarnessFailedPayload,
    HarnessSleepPayload,
    HarnessStagePayload,
    KernelEvent,
    ModelCompletedPayload,
    PauseRequestedPayload,
    ReplayedWithDriftPayload,
    ResumeRequestedPayload,
    RunStartedPayload,
    RunSummaryPayload,
    ToolCompletedPayload,
)
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore
from artana.store.base import EventStore, SupportsModelCostAggregation, SupportsRunIndexing

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class UnusedModelPort:
    async def complete(
        self,
        request: ModelRequest[OutputModelT],
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0),
        )


def _tenant(*, tenant_id: str = "org_snapshots") -> TenantContext:
    return TenantContext(
        tenant_id=tenant_id,
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )


class _ApprovalContext(BaseModel):
    approval_key: str


class SnapshotBlindStore(EventStore, SupportsRunIndexing, SupportsModelCostAggregation):
    def __init__(self, inner: SQLiteStore) -> None:
        self._inner = inner

    async def append_event(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
        parent_step_key: str | None = None,
    ) -> KernelEvent:
        return await self._inner.append_event(
            run_id=run_id,
            tenant_id=tenant_id,
            event_type=event_type,
            payload=payload,
            parent_step_key=parent_step_key,
        )

    async def get_events_for_run(self, run_id: str) -> list[KernelEvent]:
        return await self._inner.get_events_for_run(run_id)

    async def get_latest_run_summary(
        self,
        run_id: str,
        summary_type: str,
    ) -> RunSummaryPayload | None:
        return await self._inner.get_latest_run_summary(run_id, summary_type)

    async def verify_run_chain(self, run_id: str) -> bool:
        return await self._inner.verify_run_chain(run_id)

    async def close(self) -> None:
        await self._inner.close()

    async def list_run_ids(
        self,
        *,
        tenant_id: str | None = None,
        since: datetime | None = None,
    ) -> list[str]:
        return await self._inner.list_run_ids(tenant_id=tenant_id, since=since)

    async def get_model_cost_sum_for_run(self, run_id: str) -> float:
        return await self._inner.get_model_cost_sum_for_run(run_id)


@pytest.mark.asyncio
async def test_sqlite_run_state_snapshots_track_transitions(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    run_id = "run_snapshot_transition"
    tenant = _tenant(tenant_id="tenant_snapshot_transition")
    try:
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.RUN_STARTED,
            payload=RunStartedPayload(),
        )
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.PAUSE_REQUESTED,
            payload=PauseRequestedPayload(
                reason="need approval",
                context_json='{"approval_key":"mgr_1"}',
                step_key="pause_1",
            ),
        )
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.RESUME_REQUESTED,
            payload=ResumeRequestedPayload(),
        )
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.MODEL_COMPLETED,
            payload=ModelCompletedPayload(
                model="gpt-4o-mini",
                output_json='{"ok":true}',
                prompt_tokens=10,
                completion_tokens=5,
                cost_usd=0.5,
            ),
        )
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.REPLAYED_WITH_DRIFT,
            payload=ReplayedWithDriftPayload(
                step_key="drift_1",
                model="gpt-4o-mini",
                drift_fields=["prompt"],
                source_model_requested_event_id="event_req_1",
                source_model_completed_seq=2,
                replay_policy="allow_prompt_drift",
            ),
        )
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.HARNESS_STAGE,
            payload=HarnessStagePayload(stage="work", round=1),
        )
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.TOOL_COMPLETED,
            payload=ToolCompletedPayload(
                tool_name="send_invoice",
                result_json='{"ok":true}',
            ),
        )
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.HARNESS_FAILED,
            payload=HarnessFailedPayload(
                error_type="RuntimeError",
                message="failed",
                last_step_key="failed_step",
            ),
        )
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.HARNESS_SLEEP,
            payload=HarnessSleepPayload(status="completed"),
        )

        snapshot = await store.get_run_state_snapshot(run_id=run_id)
        assert snapshot is not None
        assert snapshot.status == "completed"
        assert snapshot.blocked_on is None
        assert snapshot.open_pause_count == 0
        assert snapshot.drift_count == 1
        assert snapshot.last_stage == "work"
        assert snapshot.last_tool == "send_invoice"
        assert snapshot.model_cost_total == pytest.approx(0.5)
        assert snapshot.explain_status == "completed"
        assert snapshot.explain_failure_reason == "RuntimeError"
        assert snapshot.explain_failure_step == "failed_step"

        await store.append_event(
            run_id="run_snapshot_active",
            tenant_id=tenant.tenant_id,
            event_type=EventType.RUN_STARTED,
            payload=RunStartedPayload(),
        )
        active = await store.list_run_state_snapshots(
            tenant_id=tenant.tenant_id,
            status="active",
        )
        assert [item.run_id for item in active] == ["run_snapshot_active"]

        recent = await store.list_run_state_snapshots(
            tenant_id=tenant.tenant_id,
            since=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        assert {item.run_id for item in recent} == {
            "run_snapshot_transition",
            "run_snapshot_active",
        }
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_kernel_run_state_reads_match_snapshot_and_fallback_paths(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    snapshot_kernel = ArtanaKernel(store=store, model_port=UnusedModelPort())
    fallback_kernel = ArtanaKernel(store=SnapshotBlindStore(store), model_port=UnusedModelPort())
    tenant = _tenant(tenant_id="tenant_snapshot_kernel")
    try:
        await snapshot_kernel.start_run(tenant=tenant, run_id="run_snapshot_a")
        await snapshot_kernel.pause(
            run_id="run_snapshot_a",
            tenant=tenant,
            reason="needs approval",
            context=_ApprovalContext(approval_key="approve_1"),
            step_key="pause_1",
        )
        await snapshot_kernel.resume(
            run_id="run_snapshot_a",
            tenant=tenant,
        )
        await store.append_event(
            run_id="run_snapshot_a",
            tenant_id=tenant.tenant_id,
            event_type=EventType.HARNESS_STAGE,
            payload=HarnessStagePayload(stage="work", round=1),
        )
        await store.append_event(
            run_id="run_snapshot_a",
            tenant_id=tenant.tenant_id,
            event_type=EventType.MODEL_COMPLETED,
            payload=ModelCompletedPayload(
                model="gpt-4o-mini",
                output_json='{"ok":true}',
                prompt_tokens=1,
                completion_tokens=1,
                cost_usd=0.25,
            ),
        )
        await store.append_event(
            run_id="run_snapshot_a",
            tenant_id=tenant.tenant_id,
            event_type=EventType.REPLAYED_WITH_DRIFT,
            payload=ReplayedWithDriftPayload(
                step_key="drift_1",
                model="gpt-4o-mini",
                drift_fields=["prompt"],
                source_model_requested_event_id="event_req_1",
                source_model_completed_seq=2,
                replay_policy="allow_prompt_drift",
            ),
        )
        await store.append_event(
            run_id="run_snapshot_a",
            tenant_id=tenant.tenant_id,
            event_type=EventType.TOOL_COMPLETED,
            payload=ToolCompletedPayload(
                tool_name="lookup_weather",
                result_json='{"ok":true}',
            ),
        )

        await snapshot_kernel.start_run(tenant=tenant, run_id="run_snapshot_b")
        await store.append_event(
            run_id="run_snapshot_b",
            tenant_id=tenant.tenant_id,
            event_type=EventType.HARNESS_SLEEP,
            payload=HarnessSleepPayload(status="completed"),
        )

        await snapshot_kernel.start_run(tenant=tenant, run_id="run_snapshot_c")
        await store.append_event(
            run_id="run_snapshot_c",
            tenant_id=tenant.tenant_id,
            event_type=EventType.HARNESS_FAILED,
            payload=HarnessFailedPayload(
                error_type="RuntimeError",
                message="failed",
                last_step_key="failed_step",
            ),
        )
        await store.append_event(
            run_id="run_snapshot_c",
            tenant_id=tenant.tenant_id,
            event_type=EventType.HARNESS_SLEEP,
            payload=HarnessSleepPayload(status="completed"),
        )

        for run_id in ("run_snapshot_a", "run_snapshot_b", "run_snapshot_c"):
            assert await snapshot_kernel.get_run_status(
                run_id=run_id
            ) == await fallback_kernel.get_run_status(run_id=run_id)
            assert await snapshot_kernel.resume_point(
                run_id=run_id
            ) == await fallback_kernel.resume_point(run_id=run_id)
            assert await snapshot_kernel.explain_run(
                run_id
            ) == await fallback_kernel.explain_run(run_id)

        assert await snapshot_kernel.list_active_runs(
            tenant_id=tenant.tenant_id
        ) == await fallback_kernel.list_active_runs(tenant_id=tenant.tenant_id)
        assert await snapshot_kernel.list_active_runs(
            tenant_id=tenant.tenant_id,
            status="completed",
        ) == await fallback_kernel.list_active_runs(
            tenant_id=tenant.tenant_id,
            status="completed",
        )
    finally:
        await fallback_kernel.close()
        await snapshot_kernel.close()
