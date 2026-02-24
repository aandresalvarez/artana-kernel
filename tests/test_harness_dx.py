from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import ArtanaKernel
from artana.events import EventType, ModelRequestedPayload, RunSummaryPayload
from artana.harness import BaseHarness, HarnessContext, IncrementalTaskHarness, TaskUnit
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class EchoArgs(BaseModel):
    email: str


class StaticDecisionModelPort:
    async def complete(
        self,
        request: ModelRequest[OutputModelT],
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate(
            {"approved": True, "reason": "policy_ok"}
        )
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=2, completion_tokens=1, cost_usd=0.01),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_harness_dx",
        capabilities=frozenset({"reports"}),
        budget_usd_limit=5.0,
    )


class DeveloperFriendlyHarness(BaseHarness[dict[str, object]]):
    async def step(self, *, context: HarnessContext) -> dict[str, object]:
        await self.write_summary("developer_state", {"phase": "running"})
        model_result = await self.run_model(
            prompt="approve this",
            output_schema=Decision,
        )
        tool_result = await self.run_tool(
            tool_name="echo_public",
            arguments=EchoArgs(email="alice@example.com"),
        )
        tools = self.list_tools()
        summary_payload = await self.read_summary("developer_state")
        return {
            "approved": model_result.output.approved,
            "tool_echo": json.loads(tool_result.result_json)["echo"],
            "visible_tools": [tool.name for tool in tools],
            "summary": summary_payload,
        }


class TaskListHarness(IncrementalTaskHarness):
    def __init__(
        self,
        *,
        kernel: ArtanaKernel,
        tenant: TenantContext | None = None,
    ) -> None:
        super().__init__(kernel=kernel, tenant=tenant)
        self.completed: list[str] = []

    async def define_tasks(self) -> list[TaskUnit]:
        return [
            TaskUnit(id="collect", description="Collect source material"),
            TaskUnit(id="summarize", description="Write summary"),
        ]

    async def work_on(self, task: TaskUnit) -> None:
        self.completed.append(task.id)
        await self.set_artifact(
            key=f"completed_{task.id}",
            value=True,
        )


@pytest.mark.asyncio
async def test_base_harness_helpers_allow_run_without_explicit_tenant(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=StaticDecisionModelPort())
    tenant = _tenant()

    @kernel.tool()
    async def echo_public(email: str) -> str:
        return json.dumps({"echo": email, "status": "ok"})

    @kernel.tool(requires_capability="payments")
    async def restricted_transfer(account_id: str) -> str:
        return json.dumps({"ok": True, "account_id": account_id})

    harness = DeveloperFriendlyHarness(kernel=kernel, tenant=tenant)
    try:
        result = await harness.run(run_id="run_harness_dx")
        assert result["approved"] is True
        assert result["tool_echo"] == "alice@example.com"
        assert result["summary"] == {"phase": "running"}
        assert result["visible_tools"] == ["echo_public"]

        events = await store.get_events_for_run("run_harness_dx")
        run_summaries = [
            event.payload
            for event in events
            if event.event_type == EventType.RUN_SUMMARY
            and isinstance(event.payload, RunSummaryPayload)
        ]
        assert any(
            payload.summary_type == "developer_state"
            for payload in run_summaries
        )
        assert any(event.event_type == EventType.MODEL_REQUESTED for event in events)
        assert any(event.event_type == EventType.TOOL_COMPLETED for event in events)
        model_requested = [
            event.payload
            for event in events
            if event.event_type == EventType.MODEL_REQUESTED
            and isinstance(event.payload, ModelRequestedPayload)
        ]
        assert any(
            payload.step_key == "developerfriendlyharness_model_1"
            for payload in model_requested
        )
        run_summaries = [
            event.payload
            for event in events
            if event.event_type == EventType.RUN_SUMMARY
            and isinstance(event.payload, RunSummaryPayload)
        ]
        assert any(
            payload.step_key == "developerfriendlyharness_summary_developer_state_1"
            for payload in run_summaries
        )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_incremental_task_harness_default_flow_runs_one_task_per_session(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=StaticDecisionModelPort())
    tenant = _tenant()
    harness = TaskListHarness(kernel=kernel, tenant=tenant)
    run_id = "run_incremental_dx"

    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)

        first = await harness.run(run_id=run_id)
        assert [unit.id for unit in first] == ["collect", "summarize"]
        assert [unit.state for unit in first] == ["done", "pending"]
        assert harness.completed == ["collect"]
        assert await harness.get_artifact(run_id=run_id, key="completed_collect") is True

        second = await harness.run(run_id=run_id)
        assert [unit.state for unit in second] == ["done", "done"]
        assert harness.completed == ["collect", "summarize"]
        assert await harness.get_artifact(run_id=run_id, key="completed_summarize") is True

        third = await harness.run(run_id=run_id)
        assert [unit.state for unit in third] == ["done", "done"]
        assert harness.completed == ["collect", "summarize"]

        events = await store.get_events_for_run(run_id)
        run_summaries = [
            event.payload
            for event in events
            if event.event_type == EventType.RUN_SUMMARY
            and isinstance(event.payload, RunSummaryPayload)
        ]
        assert any(
            payload.summary_type == "task_progress"
            for payload in run_summaries
        )
    finally:
        await kernel.close()
