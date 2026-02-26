from __future__ import annotations

from pydantic import BaseModel

from artana.harness.base import HarnessContext
from artana.harness.incremental import IncrementalTaskHarness, TaskUnit
from artana.models import TenantContext
from artana.ports.model import ModelCallOptions


class TestAdjudication(BaseModel):
    passed: bool
    reasoning: str


class ExecuteTestArgs(BaseModel):
    command: str


class TestDrivenHarness(IncrementalTaskHarness):
    __test__ = False

    async def step(self, *, context: HarnessContext) -> tuple[TaskUnit, ...]:
        if not self._uses_structured_flow:
            existing = await self.get_task_progress(run_id=context.run_id)
            if existing is None:
                return ()
            return existing

        task_progress = await self._ensure_task_progress_initialized(context=context)
        pending_task = next((unit for unit in task_progress if unit.state == "pending"), None)
        if pending_task is None:
            return task_progress

        await self.transition_task_unit(
            run_id=context.run_id,
            tenant=context.tenant,
            unit_id=pending_task.id,
            new_state="in_progress",
            step_key=f"task_{pending_task.id}_in_progress",
        )

        try:
            await self._invoke_work_on(task=pending_task, context=context)
        except Exception:
            await self.transition_task_unit(
                run_id=context.run_id,
                tenant=context.tenant,
                unit_id=pending_task.id,
                new_state="pending",
                step_key=f"task_{pending_task.id}_reset_pending",
            )
            raise

        latest = await self.get_task_progress(run_id=context.run_id)
        if latest is None:
            return ()
        current = next((unit for unit in latest if unit.id == pending_task.id), None)
        if current is not None and current.state == "in_progress":
            await self.transition_task_unit(
                run_id=context.run_id,
                tenant=context.tenant,
                unit_id=pending_task.id,
                new_state="pending",
                step_key=f"task_{pending_task.id}_verification_pending",
            )
            latest = await self.get_task_progress(run_id=context.run_id)
        if latest is None:
            return ()
        return latest

    async def verify_and_commit(
        self,
        *,
        task_id: str,
        test_command: str,
        step_key_prefix: str = "verify",
        run_id: str | None = None,
        tenant: TenantContext | None = None,
        model_options: ModelCallOptions | None = None,
        parent_step_key: str | None = None,
    ) -> bool:
        result = await self.run_tool(
            tool_name="execute_local_test",
            arguments=ExecuteTestArgs(command=test_command),
            step_key=f"{step_key_prefix}_{task_id}_exec",
            run_id=run_id,
            tenant=tenant,
            parent_step_key=parent_step_key,
        )

        adjudication = await self.run_verify_model(
            prompt=(
                "Review this test output and determine if the task is complete:\n"
                f"{result.result_json}"
            ),
            output_schema=TestAdjudication,
            step_key=f"{step_key_prefix}_{task_id}_adjudicate",
            run_id=run_id,
            tenant=tenant,
            model_options=model_options,
            parent_step_key=parent_step_key,
        )

        if adjudication.output.passed:
            await self.transition_task_unit(
                run_id=run_id,
                tenant=tenant,
                unit_id=task_id,
                new_state="done",
                step_key=f"{step_key_prefix}_{task_id}_done",
                verification_passed=True,
            )
            return True

        return False


__all__ = ["ExecuteTestArgs", "TestAdjudication", "TestDrivenHarness"]
