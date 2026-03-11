from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from _live_example_utils import (
    friendly_exit,
    print_example_header,
    print_summary,
    require_openai_api_key,
    resolve_model,
)
from pydantic import BaseModel

from artana import (
    AcceptanceSpec,
    ArtanaKernel,
    ContextBuilder,
    StepKey,
    TenantContext,
    ToolGate,
)
from artana.harness import CodingHarness, HarnessOutcome, TaskUnit, WorkspaceState
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class PatchPlan(BaseModel):
    summary: str
    changed_files: list[str]
    verification_notes: list[str]


class CheckoutCodingHarness(CodingHarness):
    def __init__(
        self,
        *,
        kernel: ArtanaKernel,
        tenant: TenantContext,
        model_name: str,
    ) -> None:
        super().__init__(
            kernel=kernel,
            tenant=tenant,
            default_model=model_name,
            draft_model=model_name,
            verify_model=model_name,
            replay_policy="strict",
            context_builder=ContextBuilder(
                progressive_skills=False,
                task_category="coding",
            ),
            acceptance=AcceptanceSpec(
                gates=(ToolGate(tool="run_tests", must_pass=True),),
            ),
            agent_system_prompt=(
                "You are a coding harness agent. Gather grounded repo context, "
                "produce a concise patch plan, and finish only after the verification tool passes."
            ),
            max_iterations=6,
        )

    async def define_tasks(self) -> list[TaskUnit]:
        return [
            TaskUnit(
                id="plan_fix",
                description="Produce a patch plan for the checkout retry regression",
            )
        ]

    async def work_on(self, task: TaskUnit) -> None:
        step = StepKey(namespace=f"{self._resolve_run_id(run_id=None)}_coding")
        prompt = (
            "Review the checkout retry regression and produce a coding plan.\n"
            "Use the available tools to gather the bug report, repo notes, "
            "and verification status.\n"
            "Return only JSON matching PatchPlan.\n"
            "Changed files must be concrete paths from the repo map.\n"
            "verification_notes must explain why the change is safe to land.\n"
        )
        result = await self.run_agent(
            prompt=prompt,
            output_schema=PatchPlan,
        )
        await self.set_artifact(
            key="patch_plan",
            value=result.model_dump(mode="json"),
            step_key=step.next("artifact_patch_plan"),
        )

    async def coding_goal(self) -> str:
        return "Fix the checkout retry regression without regressing the auth flow."

    async def coding_artifact_keys(self) -> tuple[str, ...]:
        return ("patch_plan",)

    async def coding_constraints(self) -> tuple[str, ...]:
        return (
            "changed_files must come from the repo map",
            "verification must pass before completion",
            "keep the plan concise and implementation-oriented",
        )

    async def coding_allowed_tool_names(self) -> tuple[str, ...]:
        return (
            "read_bug_report",
            "read_repo_map",
            "read_repo_notes",
            "run_tests",
        )

    async def build_workspace_state(
        self,
        *,
        context,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        state = await super().build_workspace_state(
            context=context,
            task_progress=task_progress,
        )
        return state.model_copy(
            update={
                "active_plan": (
                    "Gather bug context, produce a patch plan, and require a "
                    "verification pass before marking the task complete."
                ),
                "memory_summary": "Coding harness for a checkout retry regression.",
            }
        )

    async def build_outcome(
        self,
        *,
        context,
        task_progress: tuple[TaskUnit, ...],
        workspace_state: WorkspaceState,
    ) -> HarnessOutcome:
        outcome = await super().build_outcome(
            context=context,
            task_progress=task_progress,
            workspace_state=workspace_state,
        )
        patch_plan = await self.get_artifact(
            run_id=context.run_id,
            tenant=context.tenant,
            key="patch_plan",
        )
        if isinstance(patch_plan, dict):
            return outcome.model_copy(
                update={
                    "artifacts_produced": sorted(workspace_state.artifacts),
                    "details": {
                        "run_id": context.run_id,
                        "changed_files": patch_plan.get("changed_files", []),
                    },
                }
            )
        return outcome


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_coding_harness",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )


async def main() -> None:
    require_openai_api_key(script_name="10_live_manual_agent_harness.py")
    model_name = resolve_model(
        env_var="ARTANA_CODING_HARNESS_MODEL",
        default="openai/gpt-5.4",
    )
    print_example_header(
        title="10 - Coding Harness (GPT-5.4)",
        models={"coding_harness": model_name},
    )

    database_path = Path("examples/.state_live_coding_harness.db")
    scratch_root = Path("examples/.tmp_live_coding_harness")
    if database_path.exists():
        database_path.unlink()
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)

    kernel = ArtanaKernel(
        store=SQLiteStore(str(database_path)),
        model_port=LiteLLMAdapter(timeout_seconds=30.0, max_retries=1),
        middleware=ArtanaKernel.default_middleware_stack(),
    )

    @kernel.tool()
    async def read_bug_report() -> str:
        return json.dumps(
            {
                "service": "checkout-service",
                "issue": "Retry path can double-submit after a timeout race",
                "severity": "high",
                "goal": "Make retries idempotent and keep auth unchanged",
            }
        )

    @kernel.tool()
    async def read_repo_map() -> str:
        return json.dumps(
            {
                "candidate_files": [
                    "src/checkout/retry.py",
                    "src/checkout/idempotency.py",
                    "tests/test_checkout_retry.py",
                ]
            }
        )

    @kernel.tool()
    async def read_repo_notes() -> str:
        return json.dumps(
            {
                "notes": [
                    "The retry flow already writes an idempotency token.",
                    "Auth middleware should not change.",
                    "Regression coverage belongs in tests/test_checkout_retry.py.",
                ]
            }
        )

    @kernel.tool()
    async def run_tests() -> str:
        return json.dumps(
            {
                "passed": True,
                "suite": "checkout_retry",
                "status": "passed",
            }
        )

    tenant = _tenant()
    run_id = "coding_harness_run"

    try:
        harness = CheckoutCodingHarness(
            kernel=kernel,
            tenant=tenant,
            model_name=model_name,
        )
        outcome = await harness.run(run_id=run_id)
        patch_plan = await harness.get_artifact(run_id=run_id, tenant=tenant, key="patch_plan")
        workspace_state = await harness.get_workspace_state(run_id=run_id, tenant=tenant)

        print_summary(
            payload={
                "run_id": run_id,
                "model": model_name,
                "outcome": outcome.model_dump(mode="json"),
                "workspace_state": (
                    workspace_state.model_dump(mode="json")
                    if workspace_state is not None
                    else None
                ),
                "patch_plan": patch_plan,
            }
        )
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()
        if scratch_root.exists():
            shutil.rmtree(scratch_root)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        raise friendly_exit(
            script_name="10_live_manual_agent_harness.py",
            error=exc,
        ) from exc
