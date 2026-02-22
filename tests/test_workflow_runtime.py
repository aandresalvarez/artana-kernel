from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.events import PauseRequestedPayload, WorkflowStepCompletedPayload
from artana.kernel import (
    ArtanaKernel,
    JsonValue,
    ReplayConsistencyError,
    WorkflowContext,
    json_step_serde,
    pydantic_step_serde,
)
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class NeverCalledModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        raise AssertionError("Model port should not be called in workflow runtime tests.")


class StepState(BaseModel):
    value: int


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_workflow",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


@pytest.mark.asyncio
async def test_run_workflow_replays_completed_step_with_pydantic_serde(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"
    compute_attempts = [0]

    async def compute_step() -> StepState:
        compute_attempts[0] += 1
        return StepState(value=7)

    async def workflow(context: WorkflowContext) -> int:
        state = await context.step(
            name="compute",
            action=compute_step,
            serde=pydantic_step_serde(StepState),
        )
        return state.value

    first_store = SQLiteStore(str(database_path))
    first_kernel = ArtanaKernel(store=first_store, model_port=NeverCalledModelPort())
    try:
        first_result = await first_kernel.run_workflow(
            run_id="run_workflow_replay",
            tenant=_tenant(),
            workflow=workflow,
        )
        assert first_result.status == "complete"
        assert first_result.output == 7
    finally:
        await first_kernel.close()

    second_store = SQLiteStore(str(database_path))
    second_kernel = ArtanaKernel(store=second_store, model_port=NeverCalledModelPort())
    try:
        second_result = await second_kernel.run_workflow(
            run_id="run_workflow_replay",
            tenant=_tenant(),
            workflow=workflow,
        )
        assert second_result.status == "complete"
        assert second_result.output == 7
        assert compute_attempts[0] == 1

        events = await second_store.get_events_for_run("run_workflow_replay")
        assert [event.event_type for event in events] == [
            "workflow_step_requested",
            "workflow_step_completed",
        ]
        assert isinstance(events[1].payload, WorkflowStepCompletedPayload)
        assert events[1].payload.step_name == "compute"
    finally:
        await second_kernel.close()


@pytest.mark.asyncio
async def test_run_workflow_pause_then_resume_continues_after_completed_steps(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"
    human_approved = [False]
    prepare_attempts = [0]
    finalize_attempts = [0]

    async def prepare_step() -> JsonValue:
        prepare_attempts[0] += 1
        return "prepared"

    async def finalize_step() -> JsonValue:
        finalize_attempts[0] += 1
        return "done"

    async def workflow(context: WorkflowContext) -> str:
        await context.step(
            name="prepare",
            action=prepare_step,
            serde=json_step_serde(),
        )
        if not human_approved[0]:
            await context.pause_for_human("manual approval required")
        finalize_result = await context.step(
            name="finalize",
            action=finalize_step,
            serde=json_step_serde(),
        )
        assert isinstance(finalize_result, str)
        return finalize_result

    first_store = SQLiteStore(str(database_path))
    first_kernel = ArtanaKernel(store=first_store, model_port=NeverCalledModelPort())
    try:
        paused = await first_kernel.run_workflow(
            run_id="run_workflow_pause",
            tenant=_tenant(),
            workflow=workflow,
        )
        assert paused.status == "paused"
        assert paused.output is None
        assert paused.pause_ticket is not None
    finally:
        await first_kernel.close()

    human_approved[0] = True

    second_store = SQLiteStore(str(database_path))
    second_kernel = ArtanaKernel(store=second_store, model_port=NeverCalledModelPort())
    try:
        resumed = await second_kernel.run_workflow(
            run_id="run_workflow_pause",
            tenant=_tenant(),
            workflow=workflow,
        )
        assert resumed.status == "complete"
        assert resumed.output == "done"
        assert resumed.pause_ticket is None
        assert prepare_attempts[0] == 1
        assert finalize_attempts[0] == 1

        events = await second_store.get_events_for_run("run_workflow_pause")
        assert [event.event_type for event in events] == [
            "workflow_step_requested",
            "workflow_step_completed",
            "pause_requested",
            "workflow_step_requested",
            "workflow_step_completed",
        ]
        assert isinstance(events[2].payload, PauseRequestedPayload)
        assert events[2].payload.reason == "manual approval required"
    finally:
        await second_kernel.close()


@pytest.mark.asyncio
async def test_run_workflow_retries_pending_step_after_crash(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"
    step_attempts = [0]

    async def flaky_step() -> JsonValue:
        step_attempts[0] += 1
        if step_attempts[0] == 1:
            raise RuntimeError("simulated step crash")
        return "ok"

    async def workflow(context: WorkflowContext) -> str:
        flaky_result = await context.step(
            name="flaky",
            action=flaky_step,
            serde=json_step_serde(),
        )
        assert isinstance(flaky_result, str)
        return flaky_result

    first_store = SQLiteStore(str(database_path))
    first_kernel = ArtanaKernel(store=first_store, model_port=NeverCalledModelPort())
    with pytest.raises(RuntimeError, match="simulated step crash"):
        await first_kernel.run_workflow(
            run_id="run_workflow_crash",
            tenant=_tenant(),
            workflow=workflow,
        )
    try:
        pending_state = await first_kernel.resume(run_id="run_workflow_crash")
        assert pending_state.status == "ready"

        first_events = await first_store.get_events_for_run("run_workflow_crash")
        assert [event.event_type for event in first_events] == [
            "workflow_step_requested",
        ]
    finally:
        await first_kernel.close()

    second_store = SQLiteStore(str(database_path))
    second_kernel = ArtanaKernel(store=second_store, model_port=NeverCalledModelPort())
    try:
        resumed = await second_kernel.run_workflow(
            run_id="run_workflow_crash",
            tenant=_tenant(),
            workflow=workflow,
        )
        assert resumed.status == "complete"
        assert resumed.output == "ok"
        assert step_attempts[0] == 2

        second_events = await second_store.get_events_for_run("run_workflow_crash")
        assert [event.event_type for event in second_events] == [
            "workflow_step_requested",
            "workflow_step_completed",
        ]
    finally:
        await second_kernel.close()


@pytest.mark.asyncio
async def test_run_workflow_detects_step_name_replay_mismatch(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=NeverCalledModelPort())

    async def alpha_step() -> JsonValue:
        return "alpha"

    async def beta_step() -> JsonValue:
        return "beta"

    async def workflow_alpha(context: WorkflowContext) -> str:
        alpha_result = await context.step(
            name="alpha",
            action=alpha_step,
            serde=json_step_serde(),
        )
        assert isinstance(alpha_result, str)
        return alpha_result

    async def workflow_beta(context: WorkflowContext) -> str:
        beta_result = await context.step(
            name="beta",
            action=beta_step,
            serde=json_step_serde(),
        )
        assert isinstance(beta_result, str)
        return beta_result

    try:
        first_result = await kernel.run_workflow(
            run_id="run_workflow_mismatch",
            tenant=_tenant(),
            workflow=workflow_alpha,
        )
        assert first_result.status == "complete"
        assert first_result.output == "alpha"

        with pytest.raises(ReplayConsistencyError):
            await kernel.run_workflow(
                run_id="run_workflow_mismatch",
                tenant=_tenant(),
                workflow=workflow_beta,
            )
    finally:
        await kernel.close()
