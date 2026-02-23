from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import ArtanaKernel
from artana.events import EventType, ToolRequestedPayload
from artana.harness import (
    HarnessContext,
    HarnessStateError,
    IncrementalTaskHarness,
    TaskProgressValidationError,
    TaskUnit,
)
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class UnusedModelPort:
    async def complete(
        self,
        request: ModelRequest[OutputModelT],
    ) -> ModelResult[OutputModelT]:
        raise AssertionError("Model should not be called in harness lifecycle tests.")


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_harness",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )


@pytest.mark.asyncio
async def test_incremental_harness_emits_wake_reorientation_and_sleep_summary(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=UnusedModelPort())
    harness = IncrementalTaskHarness(kernel=kernel)
    tenant = _tenant()
    run_id = "run_harness_reorient"

    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        await harness.set_task_progress(
            run_id=run_id,
            tenant=tenant,
            units=(
                TaskUnit(id="t1", description="Collect requirements", state="pending"),
                TaskUnit(id="t2", description="Implement patch", state="pending"),
            ),
            step_key="task_progress_init",
        )
        await harness.set_artifact(
            run_id=run_id,
            tenant=tenant,
            key="session_note",
            value={"status": "ready"},
            step_key="artifact_note",
        )

        result = await harness.run(run_id=run_id, tenant=tenant)
        assert [unit.id for unit in result] == ["t1", "t2"]

        events = await store.get_events_for_run(run_id)
        wake_payload = next(
            event.payload
            for event in events
            if event.event_type == EventType.RUN_SUMMARY
            and event.payload.summary_type == "wake_reorientation"
        )
        wake_summary = json.loads(wake_payload.summary_json)
        assert wake_summary["latest_run_summary"]["summary_type"] == "artifact::session_note"
        assert wake_summary["task_progress"][0]["id"] == "t1"
        assert wake_summary["task_progress"][1]["id"] == "t2"
        assert any(
            event.event_type == EventType.RUN_SUMMARY
            and event.payload.summary_type == "harness_sleep"
            for event in events
        )
    finally:
        await kernel.close()


class DoubleDoneHarness(IncrementalTaskHarness):
    async def step(self, *, context: HarnessContext) -> tuple[TaskUnit, ...]:
        initial = await self.get_task_progress(run_id=context.run_id)
        if initial is None:
            raise AssertionError("Expected initialized task_progress.")
        await self.set_task_progress(
            run_id=context.run_id,
            tenant=context.tenant,
            units=(
                TaskUnit(id="t1", description="First", state="done"),
                TaskUnit(id="t2", description="Second", state="pending"),
            ),
            step_key="task_done_1",
            verified_done_unit_ids=frozenset({"t1"}),
        )
        await self.set_task_progress(
            run_id=context.run_id,
            tenant=context.tenant,
            units=(
                TaskUnit(id="t1", description="First", state="done"),
                TaskUnit(id="t2", description="Second", state="done"),
            ),
            step_key="task_done_2",
            verified_done_unit_ids=frozenset({"t2"}),
        )
        return await super().step(context=context)


@pytest.mark.asyncio
async def test_incremental_harness_allows_only_one_done_transition_per_session(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=UnusedModelPort())
    harness = DoubleDoneHarness(kernel=kernel)
    tenant = _tenant()
    run_id = "run_harness_done_limit"

    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        await harness.set_task_progress(
            run_id=run_id,
            tenant=tenant,
            units=(
                TaskUnit(id="t1", description="First", state="pending"),
                TaskUnit(id="t2", description="Second", state="pending"),
            ),
            step_key="task_seed",
        )

        with pytest.raises(
            TaskProgressValidationError,
            match="Only one TaskUnit can transition to done",
        ):
            await harness.run(run_id=run_id, tenant=tenant)
    finally:
        await kernel.close()


class UnverifiedDoneHarness(IncrementalTaskHarness):
    async def step(self, *, context: HarnessContext) -> tuple[TaskUnit, ...]:
        await self.set_task_progress(
            run_id=context.run_id,
            tenant=context.tenant,
            units=(TaskUnit(id="t1", description="Only task", state="done"),),
            step_key="task_unverified_done",
        )
        return ()


@pytest.mark.asyncio
async def test_incremental_harness_requires_verification_for_done_transition(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=UnusedModelPort())
    harness = UnverifiedDoneHarness(kernel=kernel)
    tenant = _tenant()
    run_id = "run_harness_verify_done"

    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        await harness.set_task_progress(
            run_id=run_id,
            tenant=tenant,
            units=(TaskUnit(id="t1", description="Only task", state="pending"),),
            step_key="task_seed",
        )
        with pytest.raises(
            TaskProgressValidationError,
            match="requires explicit verification",
        ):
            await harness.run(run_id=run_id, tenant=tenant)
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_harness_sleep_requires_clean_state_without_pending_tool_requests(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=UnusedModelPort())
    harness = IncrementalTaskHarness(kernel=kernel)
    tenant = _tenant()
    run_id = "run_harness_clean_state"

    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.TOOL_REQUESTED,
            payload=ToolRequestedPayload(
                tool_name="noop",
                arguments_json="{}",
                idempotency_key="pending-idempotency",
            ),
        )

        with pytest.raises(
            HarnessStateError,
            match="Cannot sleep with unresolved tool requests",
        ):
            await harness.run(run_id=run_id, tenant=tenant)
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_kernel_latest_summary_aliases_return_latest_payload(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=UnusedModelPort())
    harness = IncrementalTaskHarness(kernel=kernel)
    tenant = _tenant()
    run_id = "run_kernel_latest_summary"

    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        await harness.emit_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type="task_progress",
            payload={"units": [{"id": "t1", "description": "Task", "state": "pending"}]},
            step_key="task_1",
        )
        await harness.emit_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type="task_progress",
            payload={"units": [{"id": "t1", "description": "Task", "state": "done"}]},
            step_key="task_2",
        )

        latest_a = await kernel.get_latest_run_summary(
            run_id=run_id,
            summary_type="task_progress",
        )
        latest_b = await kernel.get_latest_summary(
            run_id=run_id,
            summary_type="task_progress",
        )
        assert latest_a is not None
        assert latest_b is not None
        assert latest_a.summary_json == latest_b.summary_json
        assert latest_a.step_key == "task_2"
        assert json.loads(latest_a.summary_json)["units"][0]["state"] == "done"
    finally:
        await kernel.close()
