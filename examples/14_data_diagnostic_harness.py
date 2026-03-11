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

from artana import ArtanaKernel, ContextBuilder, TenantContext
from artana.harness import DataHarness, HarnessOutcome, TaskUnit, WorkspaceState
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class DataDiagnosis(BaseModel):
    issue_summary: str
    likely_root_cause: str
    supporting_evidence: list[str]
    suggested_fix: str


class EtlDiagnosticHarness(DataHarness):
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
                task_category="data",
            ),
            agent_system_prompt=(
                "You are a data diagnostic harness agent. Inspect logs, schema summaries, "
                "and quality rules, then produce a grounded diagnosis."
            ),
            max_iterations=6,
        )

    async def define_tasks(self) -> list[TaskUnit]:
        return [
            TaskUnit(
                id="diagnose_etl",
                description="Diagnose the nightly ETL failure and propose a fix",
            )
        ]

    async def work_on(self, task: TaskUnit) -> None:
        diagnosis = await self.run_agent(
            prompt=(
                "Diagnose the active ETL failure.\n"
                "Use the available tools to inspect logs, schema, and quality rules.\n"
                "Each supporting_evidence item must be a clean single-sentence fact.\n"
                "Do not include correction notes, schema labels, "
                "or JSON fragments inside strings.\n"
                "Return only JSON matching DataDiagnosis."
            ),
            output_schema=DataDiagnosis,
        )
        await self.set_artifact(
            key="data_diagnosis",
            value=diagnosis.model_dump(mode="json"),
            step_key="artifact_data_diagnosis",
        )

    async def data_problem(self) -> str:
        return "Find why the nightly orders ETL failed after a schema change."

    async def data_schema_summary(self) -> str | None:
        payload = await self.get_artifact(key="schema_snapshot")
        if not isinstance(payload, dict):
            return None
        return str(payload.get("summary"))

    async def data_logs_summary(self) -> str | None:
        payload = await self.get_artifact(key="log_snapshot")
        if not isinstance(payload, dict):
            return None
        return str(payload.get("summary"))

    async def data_artifact_keys(self) -> tuple[str, ...]:
        return ("schema_snapshot", "log_snapshot", "quality_rules", "data_diagnosis")

    async def data_constraints(self) -> tuple[str, ...]:
        return (
            "suggested fixes must be grounded in the observed logs or schema",
            "quality rule violations must be cited explicitly",
        )

    async def data_allowed_tool_names(self) -> tuple[str, ...]:
        return (
            "read_etl_logs",
            "read_schema_snapshot",
            "read_quality_rules",
        )

    async def build_workspace_state(
        self,
        *,
        context,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        if await self.get_artifact(key="schema_snapshot") is None:
            schema = await self.run_tool(
                tool_name="read_schema_snapshot",
                arguments=NoArgs(),
                step_key="seed_schema_snapshot",
            )
            await self.set_artifact(
                key="schema_snapshot",
                value=json.loads(schema.result_json),
                step_key="artifact_schema_snapshot",
            )
        if await self.get_artifact(key="log_snapshot") is None:
            logs = await self.run_tool(
                tool_name="read_etl_logs",
                arguments=NoArgs(),
                step_key="seed_log_snapshot",
            )
            await self.set_artifact(
                key="log_snapshot",
                value=json.loads(logs.result_json),
                step_key="artifact_log_snapshot",
            )
        if await self.get_artifact(key="quality_rules") is None:
            rules = await self.run_tool(
                tool_name="read_quality_rules",
                arguments=NoArgs(),
                step_key="seed_quality_rules",
            )
            await self.set_artifact(
                key="quality_rules",
                value=json.loads(rules.result_json),
                step_key="artifact_quality_rules",
            )
        state = await super().build_workspace_state(
            context=context,
            task_progress=task_progress,
        )
        return state.model_copy(
            update={
                "active_plan": (
                    "Load schema, logs, and quality rules before proposing the ETL fix."
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
        diagnosis = await self.get_artifact(
            run_id=context.run_id,
            tenant=context.tenant,
            key="data_diagnosis",
        )
        if not isinstance(diagnosis, dict):
            return outcome
        return outcome.model_copy(
            update={
                "details": {
                    "run_id": context.run_id,
                    "likely_root_cause": diagnosis.get("likely_root_cause"),
                }
            }
        )


class NoArgs(BaseModel):
    pass


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_data_harness",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )


async def main() -> None:
    require_openai_api_key(script_name="14_data_diagnostic_harness.py")
    model_name = resolve_model(
        env_var="ARTANA_DATA_HARNESS_MODEL",
        default="openai/gpt-5.4",
    )
    print_example_header(
        title="14 - Data Diagnostic Harness (GPT-5.4)",
        models={"data_harness": model_name},
    )

    database_path = Path("examples/.state_data_diagnostic_harness.db")
    scratch_root = Path("examples/.tmp_data_diagnostic_harness")
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
    async def read_etl_logs() -> str:
        return json.dumps(
            {
                "summary": "Nightly job failed with a null customer_id after schema update.",
                "tail": [
                    "column customer_id became required",
                    "transform step still emits nullable customer_id",
                ],
            }
        )

    @kernel.tool()
    async def read_schema_snapshot() -> str:
        return json.dumps(
            {
                "summary": "orders_clean.customer_id changed from nullable to required",
                "version": "2026.03.11",
            }
        )

    @kernel.tool()
    async def read_quality_rules() -> str:
        return json.dumps(
            {
                "rules": [
                    "customer_id must be non-null",
                    "every orders_clean row must map to a customer dimension row",
                ]
            }
        )

    tenant = _tenant()
    run_id = "data_diagnostic_harness"

    try:
        harness = EtlDiagnosticHarness(
            kernel=kernel,
            tenant=tenant,
            model_name=model_name,
        )
        outcome = await harness.run(run_id=run_id)
        diagnosis = await harness.get_artifact(
            run_id=run_id,
            tenant=tenant,
            key="data_diagnosis",
        )

        print_summary(
            payload={
                "run_id": run_id,
                "model": model_name,
                "outcome": outcome.model_dump(mode="json"),
                "data_diagnosis": diagnosis,
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
            script_name="14_data_diagnostic_harness.py",
            error=exc,
        ) from exc
