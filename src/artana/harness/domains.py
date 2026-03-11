from __future__ import annotations

from collections.abc import Sequence

from artana.harness.agentic import StrongModelAgentHarness
from artana.harness.base import HarnessContext
from artana.harness.incremental import TaskUnit
from artana.harness.state import HarnessOutcome, WorkspaceState


class ResearchHarness(StrongModelAgentHarness):
    __test__ = False

    async def research_question(self) -> str:
        raise NotImplementedError

    async def research_artifact_keys(self) -> tuple[str, ...]:
        return ()

    async def research_graph_summary(self) -> str | None:
        return None

    async def research_evidence_count(self) -> int | None:
        return None

    async def research_contradictions(self) -> tuple[str, ...]:
        return ()

    async def research_constraints(self) -> tuple[str, ...]:
        return ()

    async def research_allowed_tool_names(self) -> Sequence[str] | None:
        return None

    async def build_workspace_state(
        self,
        *,
        context: HarnessContext,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        return await self.snapshot_workspace_state(
            domain="research",
            question=await self.research_question(),
            graph_summary=await self.research_graph_summary(),
            evidence_count=await self.research_evidence_count(),
            artifact_keys=await self.research_artifact_keys(),
            constraints=await self.research_constraints(),
            open_tasks=[
                f"{unit.id}: {unit.description}"
                for unit in task_progress
                if unit.state != "done"
            ],
            unresolved_contradictions=await self.research_contradictions(),
            allowed_tool_names=await self.research_allowed_tool_names(),
            notes={"task_progress": _task_notes(task_progress)},
            run_id=context.run_id,
            tenant=context.tenant,
        )


class CurationHarness(StrongModelAgentHarness):
    __test__ = False

    async def curation_goal(self) -> str:
        raise NotImplementedError

    async def curation_artifact_keys(self) -> tuple[str, ...]:
        return ()

    async def curation_constraints(self) -> tuple[str, ...]:
        return ()

    async def curation_allowed_tool_names(self) -> Sequence[str] | None:
        return None

    async def build_workspace_state(
        self,
        *,
        context: HarnessContext,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        return await self.snapshot_workspace_state(
            domain="curation",
            question=await self.curation_goal(),
            artifact_keys=await self.curation_artifact_keys(),
            constraints=await self.curation_constraints(),
            open_tasks=[
                f"{unit.id}: {unit.description}"
                for unit in task_progress
                if unit.state != "done"
            ],
            allowed_tool_names=await self.curation_allowed_tool_names(),
            notes={"task_progress": _task_notes(task_progress)},
            run_id=context.run_id,
            tenant=context.tenant,
        )


class CodingHarness(StrongModelAgentHarness):
    __test__ = False

    async def coding_goal(self) -> str:
        raise NotImplementedError

    async def coding_artifact_keys(self) -> tuple[str, ...]:
        return ()

    async def coding_constraints(self) -> tuple[str, ...]:
        return ()

    async def coding_allowed_tool_names(self) -> Sequence[str] | None:
        return None

    async def build_workspace_state(
        self,
        *,
        context: HarnessContext,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        return await self.snapshot_workspace_state(
            domain="coding",
            question=await self.coding_goal(),
            artifact_keys=await self.coding_artifact_keys(),
            constraints=await self.coding_constraints(),
            open_tasks=[
                f"{unit.id}: {unit.description}"
                for unit in task_progress
                if unit.state != "done"
            ],
            allowed_tool_names=await self.coding_allowed_tool_names(),
            notes={"task_progress": _task_notes(task_progress)},
            run_id=context.run_id,
            tenant=context.tenant,
        )


class SupportHarness(StrongModelAgentHarness):
    __test__ = False

    async def support_task(self) -> str:
        raise NotImplementedError

    async def support_customer_summary(self) -> str | None:
        return None

    async def support_ticket_history(self) -> str | None:
        return None

    async def support_artifact_keys(self) -> tuple[str, ...]:
        return ()

    async def support_constraints(self) -> tuple[str, ...]:
        return ()

    async def support_allowed_tool_names(self) -> Sequence[str] | None:
        return None

    async def build_workspace_state(
        self,
        *,
        context: HarnessContext,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        return await self.snapshot_workspace_state(
            domain="support",
            question=await self.support_task(),
            memory_summary=await self.support_customer_summary(),
            graph_summary=await self.support_ticket_history(),
            artifact_keys=await self.support_artifact_keys(),
            constraints=await self.support_constraints(),
            open_tasks=[
                f"{unit.id}: {unit.description}"
                for unit in task_progress
                if unit.state != "done"
            ],
            allowed_tool_names=await self.support_allowed_tool_names(),
            notes={"task_progress": _task_notes(task_progress)},
            run_id=context.run_id,
            tenant=context.tenant,
        )


class DataHarness(StrongModelAgentHarness):
    __test__ = False

    async def data_problem(self) -> str:
        raise NotImplementedError

    async def data_schema_summary(self) -> str | None:
        return None

    async def data_logs_summary(self) -> str | None:
        return None

    async def data_artifact_keys(self) -> tuple[str, ...]:
        return ()

    async def data_constraints(self) -> tuple[str, ...]:
        return ()

    async def data_allowed_tool_names(self) -> Sequence[str] | None:
        return None

    async def build_workspace_state(
        self,
        *,
        context: HarnessContext,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        return await self.snapshot_workspace_state(
            domain="data",
            question=await self.data_problem(),
            active_plan=await self.data_schema_summary(),
            graph_summary=await self.data_logs_summary(),
            artifact_keys=await self.data_artifact_keys(),
            constraints=await self.data_constraints(),
            open_tasks=[
                f"{unit.id}: {unit.description}"
                for unit in task_progress
                if unit.state != "done"
            ],
            allowed_tool_names=await self.data_allowed_tool_names(),
            notes={"task_progress": _task_notes(task_progress)},
            run_id=context.run_id,
            tenant=context.tenant,
        )


class ActionHarness(StrongModelAgentHarness):
    __test__ = False

    async def action_goal(self) -> str:
        raise NotImplementedError

    async def action_subject_summary(self) -> str | None:
        return None

    async def action_limits_summary(self) -> str | None:
        return None

    async def action_artifact_keys(self) -> tuple[str, ...]:
        return ()

    async def action_constraints(self) -> tuple[str, ...]:
        return ()

    async def action_allowed_tool_names(self) -> Sequence[str] | None:
        return None

    async def build_workspace_state(
        self,
        *,
        context: HarnessContext,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        return await self.snapshot_workspace_state(
            domain="action",
            question=await self.action_goal(),
            memory_summary=await self.action_subject_summary(),
            graph_summary=await self.action_limits_summary(),
            artifact_keys=await self.action_artifact_keys(),
            constraints=await self.action_constraints(),
            open_tasks=[
                f"{unit.id}: {unit.description}"
                for unit in task_progress
                if unit.state != "done"
            ],
            allowed_tool_names=await self.action_allowed_tool_names(),
            notes={"task_progress": _task_notes(task_progress)},
            run_id=context.run_id,
            tenant=context.tenant,
        )


class ReviewHarness(StrongModelAgentHarness):
    __test__ = False

    async def review_question(self) -> str:
        raise NotImplementedError

    async def review_artifact_keys(self) -> tuple[str, ...]:
        return ()

    async def review_constraints(self) -> tuple[str, ...]:
        return ()

    async def review_blockers(self) -> tuple[str, ...]:
        return ()

    async def review_allowed_tool_names(self) -> Sequence[str] | None:
        return None

    async def build_workspace_state(
        self,
        *,
        context: HarnessContext,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        return await self.snapshot_workspace_state(
            domain="review",
            question=await self.review_question(),
            artifact_keys=await self.review_artifact_keys(),
            constraints=await self.review_constraints(),
            open_tasks=[
                f"{unit.id}: {unit.description}"
                for unit in task_progress
                if unit.state != "done"
            ],
            unresolved_contradictions=await self.review_blockers(),
            allowed_tool_names=await self.review_allowed_tool_names(),
            notes={"task_progress": _task_notes(task_progress)},
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
        outcome = await super().build_outcome(
            context=context,
            task_progress=task_progress,
            workspace_state=workspace_state,
        )
        if workspace_state.unresolved_contradictions:
            return outcome.model_copy(
                update={
                    "status": "needs_review",
                    "human_review_needed": True,
                }
            )
        return outcome


def _task_notes(task_progress: tuple[TaskUnit, ...]) -> list[dict[str, str]]:
    return [
        {
            "id": unit.id,
            "description": unit.description,
            "state": unit.state,
        }
        for unit in task_progress
    ]


__all__ = [
    "ActionHarness",
    "CodingHarness",
    "CurationHarness",
    "DataHarness",
    "ResearchHarness",
    "ReviewHarness",
    "SupportHarness",
]
