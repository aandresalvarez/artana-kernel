from __future__ import annotations

import asyncio
import json
import re
import shutil
from functools import cache
from pathlib import Path
from typing import Literal

from _live_example_utils import (
    friendly_exit,
    print_example_header,
    print_summary,
    require_openai_api_key,
    resolve_model,
)
from pydantic import BaseModel

from artana import ArtanaKernel, ModelCallOptions, TenantContext
from artana.harness import HarnessOutcome, ReviewHarness, TaskUnit, WorkspaceState
from artana.ports.model import LiteLLMAdapter
from artana.ports.tool import ToolExecutionContext
from artana.safety import IntentPlanRecord
from artana.store import SQLiteStore

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "openai_docs" / "prompts.md"


class EmptyArgs(BaseModel):
    pass


class DependencyStatusArgs(BaseModel):
    dependency: str


class PlanChecklistItem(BaseModel):
    name: str
    done_when: str


class ReleasePlan(BaseModel):
    objective: str
    publish_title: str
    checklist: list[PlanChecklistItem]


class EvidenceSummary(BaseModel):
    confirmed_facts: list[str]
    key_risks: list[str]
    blockers: list[str]


class ReleaseBrief(BaseModel):
    title: str
    executive_summary: str
    recommendation: Literal["ship", "hold"]
    key_risks: list[str]
    next_actions: list[str]


class ReleaseVerification(BaseModel):
    approved: bool
    reasoning: str
    missing_items: list[str]


class DeterministicVerification(BaseModel):
    approved: bool
    failed_rules: list[str]


class PublishBriefArgs(BaseModel):
    title: str
    executive_summary: str
    recommendation: Literal["ship", "hold"]
    key_risks: list[str]
    next_actions: list[str]


@cache
def _prompt_content() -> str:
    return PROMPTS_PATH.read_text(encoding="utf-8")


def _prompt_block(tag: str) -> str:
    content = _prompt_content()
    match = re.search(rf"(<{tag}>.*?</{tag}>)", content, re.DOTALL)
    if match is None:
        raise RuntimeError(f"Prompt block <{tag}> not found in {PROMPTS_PATH}.")
    return match.group(1).strip()


def _prompt_blocks(*tags: str) -> str:
    return "\n\n".join(_prompt_block(tag) for tag in tags)


class ReleaseReadinessHarness(ReviewHarness):
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
        )

    async def define_tasks(self) -> list[TaskUnit]:
        return [
            TaskUnit(
                id="plan",
                description="Draft a human-readable release checklist for reviewers",
            ),
            TaskUnit(id="collect_evidence", description="Gather release and dependency evidence"),
            TaskUnit(id="draft_brief", description="Draft the release brief"),
            TaskUnit(id="verify_publish", description="Verify and publish the release brief"),
        ]

    async def work_on(self, task: TaskUnit) -> None:
        if task.id == "plan":
            await self._plan_release()
            return
        if task.id == "collect_evidence":
            await self._collect_evidence()
            return
        if task.id == "draft_brief":
            await self._draft_brief()
            return
        if task.id == "verify_publish":
            await self._verify_and_publish()
            return
        raise ValueError(f"Unknown task id={task.id!r}.")

    async def _plan_release(self) -> None:
        plan_prompt = (
            "You are drafting a human-readable release checklist for a durable staged harness.\n"
            "Task: prepare a reviewer-facing checklist for checkout-service v2026.03.10.\n"
            "Return only JSON matching ReleasePlan.\n"
            "This checklist is an artifact for humans. It does not define harness control flow.\n"
            "Use a concise checklist that covers evidence review, release risks, verification, "
            "and publish readiness.\n\n"
            f"{_prompt_blocks('output_contract', 'completeness_contract')}"
        )
        result = await self.run_draft_model(
            prompt=plan_prompt,
            output_schema=ReleasePlan,
            step_key="release_plan_draft",
            model_options=ModelCallOptions(
                api_mode="chat",
                reasoning_effort="none",
                verbosity="low",
            ),
        )
        await self.set_artifact(
            key="plan",
            value=result.output.model_dump(mode="json"),
            step_key="artifact_plan",
        )

    async def _collect_evidence(self) -> None:
        plan = ReleasePlan.model_validate(await self.get_artifact(key="plan"))

        release_inputs = await self.run_tool(
            tool_name="read_release_inputs",
            arguments=EmptyArgs(),
            step_key="collect_release_inputs",
        )
        dependency_payments = await self.run_tool(
            tool_name="read_dependency_status",
            arguments=DependencyStatusArgs(dependency="payments-api"),
            step_key="collect_dependency_payments",
        )
        dependency_risk = await self.run_tool(
            tool_name="read_dependency_status",
            arguments=DependencyStatusArgs(dependency="fraud-worker"),
            step_key="collect_dependency_risk",
        )

        evidence_prompt = (
            "Summarize the grounded release evidence.\n"
            "Return only JSON matching EvidenceSummary.\n\n"
            f"Reviewer checklist: {plan.model_dump_json()}\n"
            f"Release inputs: {release_inputs.result_json}\n"
            f"Dependency payments-api: {dependency_payments.result_json}\n"
            f"Dependency fraud-worker: {dependency_risk.result_json}\n\n"
            f"{_prompt_blocks('grounding_rules', 'completeness_contract', 'output_contract')}"
        )
        result = await self.run_model(
            prompt=evidence_prompt,
            output_schema=EvidenceSummary,
            step_key="collect_evidence_summary",
            model_options=ModelCallOptions(
                api_mode="chat",
                reasoning_effort="low",
                verbosity="low",
            ),
        )
        await self.set_artifact(
            key="evidence_summary",
            value=result.output.model_dump(mode="json"),
            step_key="artifact_evidence_summary",
        )

    async def _draft_brief(self) -> None:
        plan = ReleasePlan.model_validate(await self.get_artifact(key="plan"))
        evidence = EvidenceSummary.model_validate(await self.get_artifact(key="evidence_summary"))

        brief_prompt = (
            "Draft the final release brief for a human reviewer.\n"
            "Return only JSON matching ReleaseBrief.\n\n"
            f"Reviewer checklist: {plan.model_dump_json()}\n"
            f"Evidence: {evidence.model_dump_json()}\n\n"
            "Rules:\n"
            "- key_risks must be copied verbatim from EvidenceSummary.key_risks or blockers.\n"
            "- If blockers are present, recommendation must be 'hold'.\n"
            "- When recommendation is 'hold', next_actions must directly address the blockers.\n"
            "- Do not invent risks that are not present in the evidence.\n\n"
            f"{_prompt_blocks('output_contract', 'default_follow_through_policy')}"
        )
        result = await self.run_model(
            prompt=brief_prompt,
            output_schema=ReleaseBrief,
            step_key="draft_release_brief",
            model_options=ModelCallOptions(
                api_mode="chat",
                reasoning_effort="low",
                verbosity="low",
            ),
        )
        await self.set_artifact(
            key="release_brief",
            value=result.output.model_dump(mode="json"),
            step_key="artifact_release_brief",
        )

    async def _verify_and_publish(self) -> None:
        brief = ReleaseBrief.model_validate(await self.get_artifact(key="release_brief"))
        evidence = EvidenceSummary.model_validate(await self.get_artifact(key="evidence_summary"))

        verification_prompt = (
            "Verify whether the release brief is complete relative to the collected "
            "evidence package, grounded, and safe to publish as a reviewer brief.\n"
            "Return only JSON matching ReleaseVerification.\n\n"
            f"Brief: {brief.model_dump_json()}\n"
            f"Evidence: {evidence.model_dump_json()}\n\n"
            "Rules:\n"
            "- Judge completeness only against the provided evidence package.\n"
            "- Do not require external release artifacts, sign-offs, or staging data that "
            "were not part of the collected evidence.\n"
            "- Reject the brief only if it invents facts, omits evidence-backed blockers, "
            "or recommends ship despite blockers.\n\n"
            f"{_prompt_blocks(
                'verification_loop',
                'grounding_rules',
                'output_contract',
            )}"
        )
        verification = await self.run_verify_model(
            prompt=verification_prompt,
            output_schema=ReleaseVerification,
            step_key="verify_release_brief",
            model_options=ModelCallOptions(
                api_mode="chat",
                reasoning_effort="high",
                verbosity="low",
            ),
        )
        await self.set_artifact(
            key="verification",
            value=verification.output.model_dump(mode="json"),
            step_key="artifact_verification",
        )
        deterministic_verification = self._deterministic_verification(
            brief=brief,
            evidence=evidence,
        )
        await self.set_artifact(
            key="deterministic_verification",
            value=deterministic_verification.model_dump(mode="json"),
            step_key="artifact_deterministic_verification",
        )
        if not verification.output.approved:
            missing = ", ".join(verification.output.missing_items) or "unspecified"
            raise RuntimeError(
                "Release brief verification failed before publish. "
                f"Reason: {verification.output.reasoning}. Missing: {missing}."
            )
        if not deterministic_verification.approved:
            failed_rules = ", ".join(deterministic_verification.failed_rules)
            raise RuntimeError(
                "Release brief failed deterministic verification before publish. "
                f"Failed rules: {failed_rules}."
            )

        run_id = self._resolve_run_id(run_id=None)
        tenant = self._resolve_tenant(tenant=None)
        await self.kernel.record_intent_plan(
            run_id=run_id,
            tenant=tenant,
            intent=IntentPlanRecord(
                intent_id="publish_release_brief_v2026_03_10",
                goal="Publish release brief for checkout-service v2026.03.10",
                why="Share a grounded release recommendation with stakeholders",
                success_criteria="Release brief stored exactly once with its idempotency key",
                assumed_state="Release inputs and dependency evidence have been collected",
                applies_to_tools=("publish_release_brief",),
            ),
            step_key="publish_release_intent",
        )

        published = await self.run_tool(
            tool_name="publish_release_brief",
            arguments=PublishBriefArgs(
                title=brief.title,
                executive_summary=brief.executive_summary,
                recommendation=brief.recommendation,
                key_risks=brief.key_risks,
                next_actions=brief.next_actions,
            ),
            step_key="publish_release_brief",
        )
        await self.set_artifact(
            key="publish_receipt",
            value=json.loads(published.result_json),
            step_key="artifact_publish_receipt",
        )

    def _deterministic_verification(
        self,
        *,
        brief: ReleaseBrief,
        evidence: EvidenceSummary,
    ) -> DeterministicVerification:
        failed_rules = [
            *self._rule_brief_risks_are_grounded(brief=brief, evidence=evidence),
            *self._rule_ship_requires_no_blockers(brief=brief, evidence=evidence),
            *self._rule_title_is_present_and_concise(brief=brief),
            *self._rule_hold_requires_next_actions(brief=brief),
        ]

        return DeterministicVerification(
            approved=len(failed_rules) == 0,
            failed_rules=failed_rules,
        )

    def _rule_brief_risks_are_grounded(
        self,
        *,
        brief: ReleaseBrief,
        evidence: EvidenceSummary,
    ) -> list[str]:
        evidence_items = tuple((*evidence.key_risks, *evidence.blockers))
        normalized_evidence_items = {_normalize_text(item) for item in evidence_items}
        failures: list[str] = []
        for risk in brief.key_risks:
            if _normalize_text(risk) not in normalized_evidence_items:
                failures.append(f"brief risk missing from evidence summary: {risk}")
        return failures

    def _rule_ship_requires_no_blockers(
        self,
        *,
        brief: ReleaseBrief,
        evidence: EvidenceSummary,
    ) -> list[str]:
        if brief.recommendation == "ship" and evidence.blockers:
            return ["recommendation 'ship' is not allowed when blockers exist"]
        return []

    def _rule_title_is_present_and_concise(
        self,
        *,
        brief: ReleaseBrief,
    ) -> list[str]:
        failures: list[str] = []
        title = brief.title.strip()
        if title == "":
            failures.append("title must be non-empty")
        if len(title) > 80:
            failures.append("title must be concise (80 chars max)")
        return failures

    def _rule_hold_requires_next_actions(
        self,
        *,
        brief: ReleaseBrief,
    ) -> list[str]:
        if brief.recommendation == "hold" and not brief.next_actions:
            return ["next_actions must be present when recommendation is 'hold'"]
        return []

    async def build_workspace_state(
        self,
        *,
        context,
        task_progress: tuple[TaskUnit, ...],
    ) -> WorkspaceState:
        evidence_payload = await self.get_artifact(
            run_id=context.run_id,
            tenant=context.tenant,
            key="evidence_summary",
        )
        blockers: list[str] = []
        evidence_count: int | None = None
        if isinstance(evidence_payload, dict):
            blockers_obj = evidence_payload.get("blockers")
            confirmed_facts_obj = evidence_payload.get("confirmed_facts")
            if isinstance(blockers_obj, list):
                blockers = [str(item) for item in blockers_obj]
            if isinstance(confirmed_facts_obj, list):
                evidence_count = len(confirmed_facts_obj)

        return await self.snapshot_workspace_state(
            domain="release_review",
            question="Should checkout-service v2026.03.10 ship?",
            active_plan=(
                "Plan release checklist, gather evidence, draft the brief, "
                "verify deterministically, then publish once."
            ),
            evidence_count=evidence_count,
            artifact_keys=(
                "plan",
                "evidence_summary",
                "release_brief",
                "verification",
                "deterministic_verification",
                "publish_receipt",
            ),
            constraints=(
                "publish requires release:publish capability",
                "publish requires a recorded intent plan",
                "publish must remain idempotent across replay",
            ),
            open_tasks=[
                f"{unit.id}: {unit.description}"
                for unit in task_progress
                if unit.state != "done"
            ],
            unresolved_contradictions=blockers,
            allowed_tool_names=(
                "read_release_inputs",
                "read_dependency_status",
                "publish_release_brief",
            ),
            notes={
                "task_progress": [
                    {
                        "id": unit.id,
                        "description": unit.description,
                        "state": unit.state,
                    }
                    for unit in task_progress
                ]
            },
            run_id=context.run_id,
            tenant=context.tenant,
        )

    async def build_outcome(
        self,
        *,
        context,
        task_progress: tuple[TaskUnit, ...],
        workspace_state: WorkspaceState,
    ) -> HarnessOutcome:
        verification_payload = await self.get_artifact(
            run_id=context.run_id,
            tenant=context.tenant,
            key="verification",
        )
        deterministic_payload = await self.get_artifact(
            run_id=context.run_id,
            tenant=context.tenant,
            key="deterministic_verification",
        )
        publish_receipt = await self.get_artifact(
            run_id=context.run_id,
            tenant=context.tenant,
            key="publish_receipt",
        )
        release_brief = await self.get_artifact(
            run_id=context.run_id,
            tenant=context.tenant,
            key="release_brief",
        )

        verification_approved = (
            bool(verification_payload.get("approved"))
            if isinstance(verification_payload, dict)
            else False
        )
        deterministic_approved = (
            bool(deterministic_payload.get("approved"))
            if isinstance(deterministic_payload, dict)
            else False
        )
        human_review_needed = (
            isinstance(release_brief, dict)
            and release_brief.get("recommendation") == "hold"
        )
        completed = all(unit.state == "done" for unit in task_progress)
        gates_passed: list[str] = []
        gates_failed: list[str] = []
        if verification_payload is not None:
            target = gates_passed if verification_approved else gates_failed
            target.append("model_verification")
        if deterministic_payload is not None:
            target = gates_passed if deterministic_approved else gates_failed
            target.append("deterministic_verification")
        if publish_receipt is not None:
            gates_passed.append("publish_receipt_recorded")

        if completed and verification_approved and deterministic_approved and publish_receipt:
            status = "completed"
        elif human_review_needed:
            status = "needs_review"
        else:
            status = "in_progress"

        next_action = next(
            (
                f"Advance task {unit.id}: {unit.description}"
                for unit in task_progress
                if unit.state != "done"
            ),
            None,
        )
        return HarnessOutcome(
            status=status,
            confidence=0.94 if status == "completed" else None,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            artifacts_produced=sorted(workspace_state.artifacts),
            next_recommended_action=next_action,
            human_review_needed=human_review_needed,
            workspace_state=workspace_state,
            task_progress=[
                {
                    "id": unit.id,
                    "description": unit.description,
                    "state": unit.state,
                }
                for unit in task_progress
            ],
            details={
                "run_id": context.run_id,
                "publish_recorded": publish_receipt is not None,
            },
        )


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_durable_release_harness",
        capabilities=frozenset({"release:publish"}),
        budget_usd_limit=10.0,
    )


async def main() -> None:
    require_openai_api_key(script_name="11_durable_release_harness.py")
    model_name = resolve_model(
        env_var="ARTANA_DURABLE_HARNESS_MODEL",
        default="openai/gpt-5.4",
    )
    print_example_header(
        title="11 - Durable Release Harness (GPT-5.4)",
        models={"harness": model_name},
    )

    database_path = Path("examples/.state_durable_release_harness.db")
    publish_root = Path("examples/.tmp_durable_release_harness")
    publish_path = publish_root / "release_brief.json"

    if database_path.exists():
        database_path.unlink()
    if publish_root.exists():
        shutil.rmtree(publish_root)
    publish_root.mkdir(parents=True, exist_ok=True)

    kernel = ArtanaKernel(
        store=SQLiteStore(str(database_path)),
        model_port=LiteLLMAdapter(timeout_seconds=45.0, max_retries=1),
        middleware=ArtanaKernel.default_middleware_stack(),
    )
    tenant = _tenant()
    run_id = "durable_release_harness"

    @kernel.tool()
    async def read_release_inputs() -> str:
        return json.dumps(
            {
                "service": "checkout-service",
                "release_version": "v2026.03.10",
                "change_summary": [
                    "cart tax bugfix",
                    "fraud timeout tuning",
                    "new payment retries",
                ],
                "known_constraints": [
                    "must keep checkout latency under 350ms",
                    "must not regress payment success rate",
                ],
                "release_window": "2026-03-10 18:00 UTC",
            }
        )

    @kernel.tool()
    async def read_dependency_status(dependency: str) -> str:
        dependency_map = {
            "payments-api": {
                "dependency": "payments-api",
                "status": "healthy",
                "notes": "error rate stable for 24h; p95 latency 180ms",
            },
            "fraud-worker": {
                "dependency": "fraud-worker",
                "status": "degraded",
                "notes": "retry queue elevated after 15:00 UTC deploy; mitigation in progress",
            },
        }
        payload = dependency_map.get(
            dependency,
            {
                "dependency": dependency,
                "status": "unknown",
                "notes": "no dependency snapshot available",
            },
        )
        return json.dumps(payload)

    @kernel.tool(requires_capability="release:publish", side_effect=True)
    async def publish_release_brief(
        title: str,
        executive_summary: str,
        recommendation: Literal["ship", "hold"],
        key_risks: list[str],
        next_actions: list[str],
        artana_context: ToolExecutionContext,
    ) -> str:
        payload = {
            "title": title,
            "executive_summary": executive_summary,
            "recommendation": recommendation,
            "key_risks": key_risks,
            "next_actions": next_actions,
            "idempotency_key": artana_context.idempotency_key,
        }
        publish_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return json.dumps(
            {
                "ok": True,
                "path": str(publish_path),
                "idempotency_key": artana_context.idempotency_key,
            }
        )

    try:
        harness = ReleaseReadinessHarness(
            kernel=kernel,
            tenant=tenant,
            model_name=model_name,
        )

        session_snapshots: list[dict[str, object]] = []
        expected_task_count = len(await harness.define_tasks())
        completed_task_count = 0
        session_index = 0

        while completed_task_count < expected_task_count:
            session_index += 1
            outcome = await harness.run(run_id=run_id)
            next_completed_task_count = sum(
                1 for unit in outcome.task_progress if unit.state == "done"
            )
            session_snapshots.append(
                {
                    "session": session_index,
                    "status": outcome.status,
                    "task_states": [
                        {"id": unit.id, "state": unit.state}
                        for unit in outcome.task_progress
                    ],
                    "workspace_open_tasks": outcome.workspace_state.open_tasks
                    if outcome.workspace_state is not None
                    else [],
                }
            )
            if next_completed_task_count <= completed_task_count:
                raise RuntimeError(
                    "Example harness did not make forward progress. "
                    "This example expects at least one additional completed task per session."
                )
            completed_task_count = next_completed_task_count

        final_plan = await kernel.get_artifact(run_id=run_id, tenant=tenant, key="plan")
        final_evidence = await kernel.get_artifact(
            run_id=run_id,
            tenant=tenant,
            key="evidence_summary",
        )
        final_brief = await kernel.get_artifact(
            run_id=run_id,
            tenant=tenant,
            key="release_brief",
        )
        verification = await kernel.get_artifact(
            run_id=run_id,
            tenant=tenant,
            key="verification",
        )
        deterministic_verification = await kernel.get_artifact(
            run_id=run_id,
            tenant=tenant,
            key="deterministic_verification",
        )
        publish_receipt = await kernel.get_artifact(
            run_id=run_id,
            tenant=tenant,
            key="publish_receipt",
        )
        workspace_state = await harness.get_workspace_state(run_id=run_id, tenant=tenant)
        harness_outcome = await harness.get_harness_outcome(run_id=run_id, tenant=tenant)

        print_summary(
            payload={
                "run_id": run_id,
                "model": model_name,
                "prompt_blocks": [
                    "output_contract",
                    "default_follow_through_policy",
                    "completeness_contract",
                    "verification_loop",
                    "grounding_rules",
                ],
                "sessions": session_snapshots,
                "plan": final_plan,
                "evidence_summary": final_evidence,
                "release_brief": final_brief,
                "verification": verification,
                "deterministic_verification": deterministic_verification,
                "publish_receipt": publish_receipt,
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
                "published_file_exists": publish_path.exists(),
            }
        )
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()
        if publish_root.exists():
            shutil.rmtree(publish_root)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        raise friendly_exit(
            script_name="11_durable_release_harness.py",
            error=exc,
        ) from exc
