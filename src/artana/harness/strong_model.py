from __future__ import annotations

from typing import cast

from artana.harness.base import HarnessContext
from artana.harness.incremental import IncrementalTaskHarness, TaskUnit
from artana.harness.state import HarnessOutcome, HarnessTaskState, WorkspaceState
from artana.kernel import TraceLevel
from artana.models import TenantContext


class StrongModelHarness(IncrementalTaskHarness):
    __test__ = False

    async def run(  # type: ignore[override]
        self,
        run_id: str,
        *,
        tenant: TenantContext | None = None,
        model: str | None = None,
        trace_level: TraceLevel | None = None,
    ) -> HarnessOutcome:
        return cast(
            HarnessOutcome,
            await super().run(
                run_id=run_id,
                tenant=tenant,
                model=model,
                trace_level=trace_level,
            ),
        )

    async def step(self, *, context: HarnessContext) -> HarnessOutcome:  # type: ignore[override]
        existing_task_progress = await self.get_task_progress(
            run_id=context.run_id,
            tenant=context.tenant,
        )
        if existing_task_progress is not None:
            pre_workspace_state = await self.build_workspace_state(
                context=context,
                task_progress=existing_task_progress,
            )
            await self.set_workspace_state(
                workspace_state=pre_workspace_state,
                step_key=self._next_step_key(prefix="workspace_state_pre"),
            )
        task_progress = await self.advance_tasks(context=context)
        workspace_state = await self.build_workspace_state(
            context=context,
            task_progress=task_progress,
        )
        await self.set_workspace_state(workspace_state=workspace_state)
        outcome = await self.build_outcome(
            context=context,
            task_progress=task_progress,
            workspace_state=workspace_state,
        )
        await self.set_harness_outcome(outcome=outcome)
        return outcome

    async def build_workspace_state(
        self,
        *,
        context: HarnessContext,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        return await self.snapshot_workspace_state(
            domain=type(self).__name__,
            active_plan=f"{type(self).__name__} staged harness",
            open_tasks=[
                unit.description
                for unit in task_progress
                if unit.state != "done"
            ],
            notes={"task_progress": _task_progress_payload(task_progress)},
            run_id=context.run_id,
            tenant=context.tenant,
        )

    async def build_outcome(
        self,
        *,
        context: HarnessContext,
        task_progress: tuple[TaskUnit, ...],
        workspace_state: WorkspaceState,
    ) -> HarnessOutcome:
        completed = all(unit.state == "done" for unit in task_progress)
        next_action = next(
            (
                unit.description
                for unit in task_progress
                if unit.state != "done"
            ),
            None,
        )
        return HarnessOutcome(
            status="completed" if completed else "in_progress",
            confidence=1.0 if completed else None,
            gates_passed=["all_tasks_completed"] if completed else [],
            artifacts_produced=sorted(workspace_state.artifacts),
            next_recommended_action=next_action,
            human_review_needed=False,
            workspace_state=workspace_state,
            task_progress=_task_states(task_progress),
            details={"task_count": len(task_progress)},
        )


def _task_progress_payload(task_progress: tuple[TaskUnit, ...]) -> list[dict[str, str]]:
    return [
        {
            "id": unit.id,
            "description": unit.description,
            "state": unit.state,
        }
        for unit in task_progress
    ]


def _task_states(task_progress: tuple[TaskUnit, ...]) -> list[HarnessTaskState]:
    return [
        HarnessTaskState(
            id=unit.id,
            description=unit.description,
            state=unit.state,
        )
        for unit in task_progress
    ]


__all__ = ["StrongModelHarness"]
