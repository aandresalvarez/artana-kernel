from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import ArtanaKernel
from artana.events import EventType, ModelRequestedPayload, RunSummaryPayload
from artana.harness import (
    ActionHarness,
    BaseHarness,
    DataHarness,
    HarnessContext,
    IncrementalTaskHarness,
    StrongModelAgentHarness,
    StrongModelHarness,
    SupportHarness,
    TaskUnit,
    WorkspaceState,
)
from artana.models import TenantContext
from artana.ports.model import ModelCallOptions, ModelRequest, ModelResult, ModelUsage
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


class WorkspaceAwareAgentResult(BaseModel):
    summary: str


class WorkspaceCaptureModelPort:
    def __init__(self) -> None:
        self.system_messages: list[str] = []

    async def complete(
        self,
        request: ModelRequest[OutputModelT],
    ) -> ModelResult[OutputModelT]:
        self.system_messages.append(request.messages[0].content)
        output = request.output_schema.model_validate(
            {"summary": "workspace snapshot observed"}
        )
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=3, completion_tokens=2, cost_usd=0.01),
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


class DraftVerifyHarness(BaseHarness[tuple[str, str]]):
    async def step(self, *, context: HarnessContext) -> tuple[str, str]:
        draft = await self.run_draft_model(
            prompt="draft answer",
            output_schema=Decision,
            model_options=ModelCallOptions(
                api_mode="responses",
                reasoning_effort="none",
                verbosity="low",
            ),
            step_key="draft_step",
        )
        verify = await self.run_verify_model(
            prompt="verify answer",
            output_schema=Decision,
            model_options=ModelCallOptions(api_mode="chat"),
            step_key="verify_step",
        )
        return (draft.output.reason, verify.output.reason)


class WorkspaceAwareHarness(StrongModelHarness):
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
            TaskUnit(id="collect", description="Collect evidence"),
            TaskUnit(id="review", description="Review evidence"),
        ]

    async def work_on(self, task: TaskUnit) -> None:
        self.completed.append(task.id)
        await self.set_artifact(
            key=f"note_{task.id}",
            value={"done": task.id},
        )

    async def build_workspace_state(
        self,
        *,
        context: HarnessContext,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        completed_artifacts = tuple(
            f"note_{unit.id}" for unit in task_progress if unit.state == "done"
        )
        return await self.snapshot_workspace_state(
            domain="review",
            question="Is the evidence package ready for publish?",
            artifact_keys=completed_artifacts,
            constraints=("review before publish",),
            open_tasks=[
                unit.description for unit in task_progress if unit.state != "done"
            ],
            notes={"completed_ids": list(self.completed)},
            run_id=context.run_id,
            tenant=context.tenant,
        )


class AgentWorkspaceHarness(StrongModelAgentHarness):
    async def define_tasks(self) -> list[TaskUnit]:
        return [TaskUnit(id="analyze", description="Analyze workspace state")]

    async def work_on(self, task: TaskUnit) -> None:
        result = await self.run_agent(
            prompt="Summarize the current workspace state.",
            output_schema=WorkspaceAwareAgentResult,
            max_iterations=1,
        )
        await self.set_artifact(key="agent_summary", value=result.model_dump(mode="json"))

    async def build_workspace_state(
        self,
        *,
        context: HarnessContext,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        return await self.snapshot_workspace_state(
            domain="research",
            question="What should the agent notice before acting?",
            artifact_keys=("seed_note", "agent_summary"),
            open_tasks=[
                unit.description for unit in task_progress if unit.state != "done"
            ],
            allowed_tool_names=(),
            notes={"task_progress": [unit.id for unit in task_progress]},
            run_id=context.run_id,
            tenant=context.tenant,
        )


class MultiSessionAgentHarness(StrongModelAgentHarness):
    async def define_tasks(self) -> list[TaskUnit]:
        return [
            TaskUnit(id="collect", description="Collect evidence"),
            TaskUnit(id="summarize", description="Write summary"),
        ]

    async def work_on(self, task: TaskUnit) -> None:
        result = await self.run_agent(
            run_id=f"{self._resolve_run_id(run_id=None)}::{task.id}",
            prompt=f"Complete the {task.id} task.",
            output_schema=WorkspaceAwareAgentResult,
            max_iterations=1,
            workspace_aware=False,
        )
        await self.set_artifact(
            key=f"agent_{task.id}",
            value=result.model_dump(mode="json"),
        )


class SupportWorkspaceHarness(SupportHarness):
    async def define_tasks(self) -> list[TaskUnit]:
        return [TaskUnit(id="respond", description="Respond to customer")]

    async def work_on(self, task: TaskUnit) -> None:
        await self.set_artifact(key="support_note", value={"ok": True})

    async def support_task(self) -> str:
        return "Resolve the customer complaint."

    async def support_customer_summary(self) -> str | None:
        return "tier=gold"

    async def support_ticket_history(self) -> str | None:
        return "two prior interactions"


class DataWorkspaceHarness(DataHarness):
    async def define_tasks(self) -> list[TaskUnit]:
        return [TaskUnit(id="diagnose", description="Diagnose ETL")]

    async def work_on(self, task: TaskUnit) -> None:
        await self.set_artifact(key="data_note", value={"ok": True})

    async def data_problem(self) -> str:
        return "Find the ETL failure."

    async def data_schema_summary(self) -> str | None:
        return "schema changed"

    async def data_logs_summary(self) -> str | None:
        return "logs show null keys"


class ActionWorkspaceHarness(ActionHarness):
    async def define_tasks(self) -> list[TaskUnit]:
        return [TaskUnit(id="execute", description="Execute action")]

    async def work_on(self, task: TaskUnit) -> None:
        await self.set_artifact(key="action_note", value={"ok": True})

    async def action_goal(self) -> str:
        return "Send the invoice."

    async def action_subject_summary(self) -> str | None:
        return "account=acct_1"

    async def action_limits_summary(self) -> str | None:
        return "daily_limit=1"


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


@pytest.mark.asyncio
async def test_domain_templates_produce_domain_specific_workspace_shapes(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=StaticDecisionModelPort())
    tenant = _tenant()

    try:
        support = await SupportWorkspaceHarness(kernel=kernel, tenant=tenant).run(
            run_id="run_support_template"
        )
        data = await DataWorkspaceHarness(kernel=kernel, tenant=tenant).run(
            run_id="run_data_template"
        )
        action = await ActionWorkspaceHarness(kernel=kernel, tenant=tenant).run(
            run_id="run_action_template"
        )

        assert support.workspace_state is not None
        assert support.workspace_state.domain == "support"
        assert support.workspace_state.memory_summary == "tier=gold"
        assert support.workspace_state.graph_summary == "two prior interactions"

        assert data.workspace_state is not None
        assert data.workspace_state.domain == "data"
        assert data.workspace_state.active_plan == "schema changed"
        assert data.workspace_state.graph_summary == "logs show null keys"

        assert action.workspace_state is not None
        assert action.workspace_state.domain == "action"
        assert action.workspace_state.memory_summary == "account=acct_1"
        assert action.workspace_state.graph_summary == "daily_limit=1"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_base_harness_draft_and_verify_wrappers_use_dedicated_models(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=StaticDecisionModelPort())
    tenant = _tenant()
    harness = DraftVerifyHarness(
        kernel=kernel,
        tenant=tenant,
        draft_model="gpt-5-mini",
        verify_model="gpt-5.4",
    )
    try:
        reasons = await harness.run(run_id="run_draft_verify")
        assert reasons == ("policy_ok", "policy_ok")

        events = await store.get_events_for_run("run_draft_verify")
        model_requested = [
            event.payload
            for event in events
            if event.event_type == EventType.MODEL_REQUESTED
            and isinstance(event.payload, ModelRequestedPayload)
        ]
        assert [payload.model for payload in model_requested] == [
            "gpt-5-mini",
            "gpt-5.4",
        ]
        assert model_requested[0].api_mode == "responses"
        assert model_requested[0].reasoning_effort == "none"
        assert model_requested[0].verbosity == "low"
        assert model_requested[1].api_mode == "chat"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_strong_model_harness_persists_workspace_state_and_outcome(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=StaticDecisionModelPort())
    tenant = _tenant()
    harness = WorkspaceAwareHarness(kernel=kernel, tenant=tenant)
    run_id = "run_strong_model_harness"

    try:
        first = await harness.run(run_id=run_id)
        assert first.status == "in_progress"
        assert first.workspace_state is not None
        assert first.workspace_state.question == "Is the evidence package ready for publish?"
        assert first.workspace_state.artifacts == {"note_collect": {"done": "collect"}}
        stored_workspace = await harness.get_workspace_state(run_id=run_id, tenant=tenant)
        assert stored_workspace == first.workspace_state
        stored_outcome = await harness.get_harness_outcome(run_id=run_id, tenant=tenant)
        assert stored_outcome == first

        second = await harness.run(run_id=run_id)
        assert second.status == "completed"
        assert second.gates_passed == ["all_tasks_completed"]
        assert second.workspace_state is not None
        assert second.workspace_state.open_tasks == []
        assert second.workspace_state.artifacts == {
            "note_collect": {"done": "collect"},
            "note_review": {"done": "review"},
        }

        events = await store.get_events_for_run(run_id)
        wake_summaries = [
            json.loads(event.payload.summary_json)
            for event in events
            if event.event_type == EventType.RUN_SUMMARY
            and isinstance(event.payload, RunSummaryPayload)
            and event.payload.summary_type == "wake_reorientation"
        ]
        assert any(
            summary["workspace_state"]["question"]
            == "Is the evidence package ready for publish?"
            and summary["harness_outcome"]["status"] == "in_progress"
            for summary in wake_summaries
            if summary["workspace_state"] is not None
            and summary["harness_outcome"] is not None
        )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_strong_model_agent_harness_injects_workspace_state_into_agent_context(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = WorkspaceCaptureModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    tenant = _tenant()
    harness = AgentWorkspaceHarness(kernel=kernel, tenant=tenant)
    run_id = "run_agent_workspace_harness"

    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        await harness.set_artifact(
            run_id=run_id,
            tenant=tenant,
            key="seed_note",
            value={"status": "ready"},
            step_key="artifact_seed_note",
        )
        outcome = await harness.run(run_id=run_id)
        assert outcome.status == "completed"
        assert model_port.system_messages
        assert "[WORKSPACE STATE SNAPSHOT]" in model_port.system_messages[0]
        assert (
            "Question: What should the agent notice before acting?"
            in model_port.system_messages[0]
        )
        assert "Artifacts: seed_note" in model_port.system_messages[0]
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_strong_model_agent_harness_allows_distinct_agent_run_ids_per_task(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = WorkspaceCaptureModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    tenant = _tenant()
    harness = MultiSessionAgentHarness(kernel=kernel, tenant=tenant)
    run_id = "run_multi_session_agent_harness"

    try:
        first = await harness.run(run_id=run_id)
        assert first.status == "in_progress"
        assert await harness.get_artifact(run_id=run_id, key="agent_collect") == {
            "summary": "workspace snapshot observed"
        }

        second = await harness.run(run_id=run_id)
        assert second.status == "completed"
        assert await harness.get_artifact(run_id=run_id, key="agent_summarize") == {
            "summary": "workspace snapshot observed"
        }

        child_collect_events = await store.get_events_for_run(f"{run_id}::collect")
        child_summarize_events = await store.get_events_for_run(f"{run_id}::summarize")
        assert any(
            event.event_type == EventType.MODEL_REQUESTED
            for event in child_collect_events
        )
        assert any(
            event.event_type == EventType.MODEL_REQUESTED
            for event in child_summarize_events
        )
    finally:
        await kernel.close()
