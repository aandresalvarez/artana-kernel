from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from artana.agent import SingleStepModelClient
from artana.kernel import ArtanaKernel, KernelPolicy, WorkflowContext, pydantic_step_serde
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Triplet(BaseModel):
    src: str
    relation: str
    dst: str


class FactExtractionResult(BaseModel):
    facts: list[Triplet]


class DerivedGraphResult(BaseModel):
    derived_relations: list[Triplet]


class CommitResult(BaseModel):
    status: str
    total_relations: int
    derived_relations: list[Triplet]


class FactExtractionModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        usage = ModelUsage(prompt_tokens=84, completion_tokens=112, cost_usd=0.0)
        output = request.output_schema.model_validate(
            {
                "facts": [
                    {"src": "DrugA", "relation": "inhibits", "dst": "GeneB"},
                    {"src": "GeneB", "relation": "activates", "dst": "DiseaseC"},
                    {"src": "DrugD", "relation": "inhibits", "dst": "GeneE"},
                    {"src": "GeneE", "relation": "activates", "dst": "DiseaseF"},
                ]
            }
        )
        return ModelResult(
            output=output,
            usage=usage,
        )


def _derive_relations(facts: list[Triplet]) -> list[Triplet]:
    inhibitions: dict[str, set[str]] = {}
    activations: dict[str, set[str]] = {}

    for fact in facts:
        relation = fact.relation.lower()
        if relation == "inhibits":
            inhibitions.setdefault(fact.src, set()).add(fact.dst)
        elif relation == "activates":
            activations.setdefault(fact.src, set()).add(fact.dst)

    derived: list[Triplet] = []
    for source, blocked_targets in inhibitions.items():
        for blocked in blocked_targets:
            for downstream in activations.get(blocked, set()):
                derived.append(Triplet(src=source, relation="REDUCES", dst=downstream))
    derived.sort(key=lambda triplet: (triplet.src, triplet.dst))
    return derived


async def _run_workflow_with_human_gate(
    kernel: ArtanaKernel,
    run_id: str,
    tenant: TenantContext,
    prompt: str,
) -> None:
    chat = SingleStepModelClient(kernel=kernel)
    should_pause = [True]

    async def workflow(context: WorkflowContext) -> CommitResult:
        async def extract_action() -> FactExtractionResult:
            result = await chat.step(
                run_id=run_id,
                tenant=tenant,
                model="triplet-extractor-demo",
                prompt=prompt,
                output_schema=FactExtractionResult,
                step_key="facts_step",
            )
            return result.output

        extracted = await context.step(
            name="extract_facts",
            action=extract_action,
            serde=pydantic_step_serde(FactExtractionResult),
        )

        async def derive_action() -> DerivedGraphResult:
            derived = _derive_relations(extracted.facts)
            return DerivedGraphResult(derived_relations=derived)

        derived = await context.step(
            name="derive_graph",
            action=derive_action,
            serde=pydantic_step_serde(DerivedGraphResult),
        )

        if should_pause[0]:
            await context.pause(
                reason="Please verify derived relationships before committing to DB.",
                context=derived,
                step_key="scientist_review",
            )

        async def commit_action() -> CommitResult:
            return CommitResult(
                status="committed",
                total_relations=len(derived.derived_relations),
                derived_relations=derived.derived_relations,
            )

        return await context.step(
            name="persist_graph",
            action=commit_action,
            serde=pydantic_step_serde(CommitResult),
        )

    result = await kernel.run_workflow(
        run_id=run_id,
        tenant=tenant,
        workflow=workflow,
    )
    if result.status != "paused":
        raise RuntimeError(
            f"Expected workflow to pause for human review, got status={result.status!r}."
        )
    print("Workflow paused:", result.pause_ticket.ticket_id)

    should_pause[0] = False
    resumed = await kernel.run_workflow(
        run_id=run_id,
        tenant=tenant,
        workflow=workflow,
    )
    if resumed.output is None:
        raise RuntimeError("Expected workflow to complete with a CommitResult.")
    print("Workflow complete:")
    print(resumed.output.model_dump_json(indent=2))


async def main() -> None:
    db_path = Path("examples/.state_hard_triplets_workflow.db")
    if db_path.exists():
        db_path.unlink()

    article = (
        "DrugA inhibits GeneB. "
        "GeneB activates DiseaseC. "
        "DrugD inhibits GeneE. "
        "GeneE activates DiseaseF."
    )
    store = SQLiteStore(str(db_path))
    kernel = ArtanaKernel(
        store=store,
        model_port=FactExtractionModelPort(),
        middleware=ArtanaKernel.default_middleware_stack(),
        policy=KernelPolicy.enforced(),
    )
    tenant = TenantContext(
        tenant_id="science_team",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    prompt = (
        "Extract explicitly stated relationship triplets from the text:\n\n"
        + article
        + "\n\nReturn only JSON matching the schema."
    )
    try:
        await _run_workflow_with_human_gate(
            kernel=kernel,
            run_id="paper_analysis_001",
            tenant=tenant,
            prompt=prompt,
        )
    finally:
        await kernel.close()
        if db_path.exists():
            db_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
