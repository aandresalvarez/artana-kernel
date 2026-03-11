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
    TenantContext,
    ToolGate,
)
from artana.harness import HarnessOutcome, SupportHarness, TaskUnit, WorkspaceState
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class SupportResolution(BaseModel):
    resolution_summary: str
    recommended_action: str
    customer_message: str
    escalation_needed: bool


class CustomerSupportHarness(SupportHarness):
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
                task_category="support",
            ),
            acceptance=AcceptanceSpec(
                gates=(ToolGate(tool="check_refund_policy", must_pass=True),),
            ),
            agent_system_prompt=(
                "You are a support harness agent. Read customer history, follow policy, "
                "and produce a grounded resolution with a clear customer-facing response."
            ),
            max_iterations=6,
        )

    async def define_tasks(self) -> list[TaskUnit]:
        return [
            TaskUnit(
                id="resolve_ticket",
                description="Produce a grounded resolution for the current support ticket",
            )
        ]

    async def work_on(self, task: TaskUnit) -> None:
        resolution = await self.run_agent(
            prompt=(
                "Resolve the active support ticket.\n"
                "Use the available tools to inspect the customer profile, ticket history, "
                "and refund policy.\n"
                "Return only JSON matching SupportResolution."
            ),
            output_schema=SupportResolution,
        )
        await self.set_artifact(
            key="support_resolution",
            value=resolution.model_dump(mode="json"),
            step_key="artifact_support_resolution",
        )

    async def support_task(self) -> str:
        return "Resolve a damaged-item complaint without violating refund policy."

    async def support_customer_summary(self) -> str | None:
        payload = await self.get_artifact(key="customer_profile")
        if not isinstance(payload, dict):
            return None
        tier = payload.get("tier")
        sentiment = payload.get("sentiment")
        return f"tier={tier}; sentiment={sentiment}"

    async def support_ticket_history(self) -> str | None:
        payload = await self.get_artifact(key="ticket_history")
        if not isinstance(payload, dict):
            return None
        events = payload.get("events")
        if not isinstance(events, list):
            return None
        return f"{len(events)} prior ticket events recorded"

    async def support_artifact_keys(self) -> tuple[str, ...]:
        return (
            "customer_profile",
            "ticket_history",
            "policy_snapshot",
            "support_resolution",
        )

    async def support_constraints(self) -> tuple[str, ...]:
        return (
            "customer response must stay within refund policy",
            "escalation is required for any exception handling",
        )

    async def support_allowed_tool_names(self) -> tuple[str, ...]:
        return (
            "read_customer_profile",
            "read_ticket_history",
            "check_refund_policy",
        )

    async def build_workspace_state(
        self,
        *,
        context,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        if await self.get_artifact(key="customer_profile") is None:
            customer = await self.run_tool(
                tool_name="read_customer_profile",
                arguments=NoArgs(),
                step_key="seed_customer_profile",
            )
            await self.set_artifact(
                key="customer_profile",
                value=json.loads(customer.result_json),
                step_key="artifact_customer_profile",
            )
        if await self.get_artifact(key="ticket_history") is None:
            history = await self.run_tool(
                tool_name="read_ticket_history",
                arguments=NoArgs(),
                step_key="seed_ticket_history",
            )
            await self.set_artifact(
                key="ticket_history",
                value=json.loads(history.result_json),
                step_key="artifact_ticket_history",
            )
        if await self.get_artifact(key="policy_snapshot") is None:
            policy = await self.run_tool(
                tool_name="check_refund_policy",
                arguments=NoArgs(),
                step_key="seed_policy_snapshot",
            )
            await self.set_artifact(
                key="policy_snapshot",
                value=json.loads(policy.result_json),
                step_key="artifact_policy_snapshot",
            )
        state = await super().build_workspace_state(
            context=context,
            task_progress=task_progress,
        )
        return state.model_copy(
            update={
                "active_plan": (
                    "Load customer context, check policy, then propose a grounded "
                    "resolution and customer reply."
                )
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
        resolution = await self.get_artifact(
            run_id=context.run_id,
            tenant=context.tenant,
            key="support_resolution",
        )
        if not isinstance(resolution, dict):
            return outcome
        return outcome.model_copy(
            update={
                "human_review_needed": bool(resolution.get("escalation_needed")),
                "details": {
                    "run_id": context.run_id,
                    "recommended_action": resolution.get("recommended_action"),
                },
            }
        )


class NoArgs(BaseModel):
    pass


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_support_harness",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )


async def main() -> None:
    require_openai_api_key(script_name="13_support_strong_model_harness.py")
    model_name = resolve_model(
        env_var="ARTANA_SUPPORT_HARNESS_MODEL",
        default="openai/gpt-5.4",
    )
    print_example_header(
        title="13 - Support Strong-Model Harness (GPT-5.4)",
        models={"support_harness": model_name},
    )

    database_path = Path("examples/.state_support_strong_model_harness.db")
    scratch_root = Path("examples/.tmp_support_strong_model_harness")
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
    async def read_customer_profile() -> str:
        return json.dumps(
            {
                "customer_id": "cust_108",
                "tier": "gold",
                "sentiment": "frustrated",
            }
        )

    @kernel.tool()
    async def read_ticket_history() -> str:
        return json.dumps(
            {
                "events": [
                    "customer reported a damaged package",
                    "warehouse confirmed box damage during transit",
                ]
            }
        )

    @kernel.tool()
    async def check_refund_policy() -> str:
        return json.dumps(
            {
                "passed": True,
                "status": "passed",
                "policy": "Gold customers can receive replacement or refund for transit damage.",
            }
        )

    tenant = _tenant()
    run_id = "support_strong_model_harness"

    try:
        harness = CustomerSupportHarness(
            kernel=kernel,
            tenant=tenant,
            model_name=model_name,
        )
        outcome = await harness.run(run_id=run_id)
        resolution = await harness.get_artifact(
            run_id=run_id,
            tenant=tenant,
            key="support_resolution",
        )

        print_summary(
            payload={
                "run_id": run_id,
                "model": model_name,
                "outcome": outcome.model_dump(mode="json"),
                "support_resolution": resolution,
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
            script_name="13_support_strong_model_harness.py",
            error=exc,
        ) from exc
