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
from artana.harness import HarnessOutcome, ResearchHarness, TaskUnit, WorkspaceState
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class EvidenceTable(BaseModel):
    confirmed_claims: list[str]
    contradictions: list[str]
    next_queries: list[str]


class ResearchBrief(BaseModel):
    question: str
    summary: str
    evidence_strength: str
    contradictions: list[str]
    next_actions: list[str]


class Med13ResearchHarness(ResearchHarness):
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
                task_category="research",
            ),
            agent_system_prompt=(
                "You are a research harness agent. Gather grounded evidence, "
                "track contradictions explicitly, and produce concise structured outputs."
            ),
            max_iterations=6,
        )

    async def define_tasks(self) -> list[TaskUnit]:
        return [
            TaskUnit(
                id="gather_evidence",
                description="Gather evidence for the MED13 transcription question",
            ),
            TaskUnit(
                id="write_brief",
                description="Write a grounded research brief for the current evidence",
            ),
        ]

    async def work_on(self, task: TaskUnit) -> None:
        if task.id == "gather_evidence":
            await self._gather_evidence()
            return
        if task.id == "write_brief":
            await self._write_brief()
            return
        raise ValueError(f"Unknown task id={task.id!r}.")

    async def _gather_evidence(self) -> None:
        graph_snapshot = await self.run_tool(
            tool_name="read_graph_snapshot",
            arguments=NoArgs(),
            step_key="collect_graph_snapshot",
        )
        await self.set_artifact(
            key="graph_snapshot",
            value=json.loads(graph_snapshot.result_json),
            step_key="artifact_graph_snapshot",
        )
        result = await self.run_agent(
            prompt=(
                "Gather grounded evidence for the active research question.\n"
                "Use the available tools to inspect literature snippets and graph context.\n"
                "Return only JSON matching EvidenceTable."
            ),
            output_schema=EvidenceTable,
        )
        await self.set_artifact(
            key="evidence_table",
            value=result.model_dump(mode="json"),
            step_key="artifact_evidence_table",
        )

    async def _write_brief(self) -> None:
        evidence = EvidenceTable.model_validate(await self.get_artifact(key="evidence_table"))
        result = await self.run_agent(
            run_id=f"{self._resolve_run_id(run_id=None)}::write_brief",
            prompt=(
                "Write a grounded research brief using the current workspace state.\n"
                f"Evidence table: {evidence.model_dump_json()}\n"
                "Return only JSON matching ResearchBrief."
            ),
            output_schema=ResearchBrief,
            workspace_aware=False,
        )
        await self.set_artifact(
            key="research_brief",
            value=result.model_dump(mode="json"),
            step_key="artifact_research_brief",
        )

    async def research_question(self) -> str:
        return "What evidence links MED13 to transcription regulation and disease risk?"

    async def research_artifact_keys(self) -> tuple[str, ...]:
        return ("graph_snapshot", "evidence_table", "research_brief")

    async def research_graph_summary(self) -> str | None:
        graph_obj = await self.get_artifact(key="graph_snapshot")
        if not isinstance(graph_obj, dict):
            return None
        summary = graph_obj.get("summary")
        return str(summary) if isinstance(summary, str) else None

    async def research_evidence_count(self) -> int | None:
        evidence = await self.get_artifact(key="evidence_table")
        if not isinstance(evidence, dict):
            return None
        claims = evidence.get("confirmed_claims")
        if not isinstance(claims, list):
            return None
        return len(claims)

    async def research_contradictions(self) -> tuple[str, ...]:
        evidence = await self.get_artifact(key="evidence_table")
        if not isinstance(evidence, dict):
            return ()
        contradictions = evidence.get("contradictions")
        if not isinstance(contradictions, list):
            return ()
        return tuple(str(item) for item in contradictions)

    async def research_constraints(self) -> tuple[str, ...]:
        return (
            "Only report claims grounded in the provided evidence tools",
            "Contradictions must be carried into the final brief",
        )

    async def research_allowed_tool_names(self) -> tuple[str, ...]:
        return (
            "search_literature",
            "read_graph_snapshot",
            "score_evidence",
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
                    "Search evidence, compare against the current graph snapshot, "
                    "then produce a grounded brief."
                ),
                "memory_summary": "MED13 research session with contradiction tracking.",
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
        brief = await self.get_artifact(
            run_id=context.run_id,
            tenant=context.tenant,
            key="research_brief",
        )
        if isinstance(brief, dict):
            return outcome.model_copy(
                update={
                    "artifacts_produced": sorted(workspace_state.artifacts),
                    "details": {
                        "run_id": context.run_id,
                        "evidence_strength": brief.get("evidence_strength"),
                    },
                }
            )
        return outcome


class NoArgs(BaseModel):
    pass


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_research_harness",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )


async def main() -> None:
    require_openai_api_key(script_name="12_research_strong_model_harness.py")
    model_name = resolve_model(
        env_var="ARTANA_RESEARCH_HARNESS_MODEL",
        default="openai/gpt-5.4",
    )
    print_example_header(
        title="12 - Research Strong-Model Harness (GPT-5.4)",
        models={"research_harness": model_name},
    )

    database_path = Path("examples/.state_research_strong_model_harness.db")
    scratch_root = Path("examples/.tmp_research_strong_model_harness")
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
    async def search_literature(query: str) -> str:
        return json.dumps(
            {
                "query": query,
                "snippets": [
                    "MED13 is a Mediator complex subunit involved in transcription control.",
                    "Rare MED13 variants have been associated with developmental phenotypes.",
                    "Some reports emphasize limited mechanistic certainty for "
                    "specific disease links.",
                ],
            }
        )

    @kernel.tool()
    async def read_graph_snapshot() -> str:
        return json.dumps(
            {
                "summary": (
                    "Graph contains MED13 -> transcription regulation and MED13 -> "
                    "developmental disorder candidate edges."
                ),
                "node_count": 12,
                "edge_count": 19,
            }
        )

    @kernel.tool()
    async def score_evidence(claim: str) -> str:
        score = 0.82 if "transcription" in claim.lower() else 0.58
        return json.dumps({"claim": claim, "score": score})

    tenant = _tenant()
    run_id = "research_strong_model_harness"

    try:
        harness = Med13ResearchHarness(
            kernel=kernel,
            tenant=tenant,
            model_name=model_name,
        )

        session_outcomes: list[dict[str, object]] = []
        expected_task_count = len(await harness.define_tasks())
        completed_task_count = 0
        session_index = 0

        while completed_task_count < expected_task_count:
            session_index += 1
            outcome = await harness.run(run_id=run_id)
            completed_task_count = sum(
                1 for unit in outcome.task_progress if unit.state == "done"
            )
            session_outcomes.append(
                {
                    "session": session_index,
                    "status": outcome.status,
                    "open_tasks": (
                        outcome.workspace_state.open_tasks
                        if outcome.workspace_state is not None
                        else []
                    ),
                }
            )

        evidence_table = await harness.get_artifact(
            run_id=run_id,
            tenant=tenant,
            key="evidence_table",
        )
        research_brief = await harness.get_artifact(
            run_id=run_id,
            tenant=tenant,
            key="research_brief",
        )
        workspace_state = await harness.get_workspace_state(run_id=run_id, tenant=tenant)
        harness_outcome = await harness.get_harness_outcome(run_id=run_id, tenant=tenant)

        print_summary(
            payload={
                "run_id": run_id,
                "model": model_name,
                "sessions": session_outcomes,
                "evidence_table": evidence_table,
                "research_brief": research_brief,
                "workspace_state": (
                    workspace_state.model_dump(mode="json")
                    if workspace_state is not None
                    else None
                ),
                "harness_outcome": (
                    harness_outcome.model_dump(mode="json")
                    if harness_outcome is not None
                    else None
                ),
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
            script_name="12_research_strong_model_harness.py",
            error=exc,
        ) from exc
