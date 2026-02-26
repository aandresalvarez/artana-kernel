from __future__ import annotations

import asyncio
import json
from pathlib import Path

from _live_example_utils import (
    friendly_exit,
    print_example_header,
    print_summary,
    require_openai_api_key,
    resolve_model,
)
from pydantic import BaseModel, Field

from artana.agent import AutonomousAgent, ContextBuilder
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.model import LiteLLMAdapter
from artana.ports.tool import ToolExecutionContext
from artana.store import SQLiteStore


class Triplet(BaseModel):
    src: str
    relation: str
    dst: str


class FactResult(BaseModel):
    facts: list[Triplet] = Field(default_factory=list)


class Adjudication(BaseModel):
    triplet: Triplet
    is_valid: bool
    reasoning: str


class AdjudicationResult(BaseModel):
    evaluations: list[Adjudication] = Field(default_factory=list)


class FinalReport(BaseModel):
    explicit_facts: list[Triplet] = Field(default_factory=list)
    verified_derived_relations: list[Triplet] = Field(default_factory=list)


def _create_kernel(
    db_path: Path,
    *,
    extractor_model: str,
    adjudicator_model: str,
) -> ArtanaKernel:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kernel = ArtanaKernel(
        store=SQLiteStore(str(db_path)),
        model_port=LiteLLMAdapter(),
        middleware=ArtanaKernel.default_middleware_stack(),
    )

    @kernel.tool(requires_capability="spawn_extractor")
    async def run_extractor_agent(
        text: str,
        artana_context: ToolExecutionContext,
    ) -> str:
        """Sub-agent: strict fact extraction from text."""
        print("\n  🕵️‍♂️ [Extractor Sub-Agent] Waking up to extract facts...")

        tenant = TenantContext(
            tenant_id=artana_context.tenant_id,
            budget_usd_limit=5.0,
            capabilities=frozenset(),
        )
        agent = AutonomousAgent(
            kernel,
            context_builder=ContextBuilder(progressive_skills=False),
        )

        result = await agent.run(
            run_id=f"{artana_context.run_id}_ext_{artana_context.idempotency_key[:6]}",
            tenant=tenant,
            model=extractor_model,
            system_prompt=(
                "You extract explicitly stated relationships. "
                "Allowed relations: INHIBITS, ACTIVATES, PART_OF."
            ),
            prompt=f"Text to analyze: {text}",
            output_schema=FactResult,
        )
        print(
            f"  🕵️‍♂️ [Extractor Sub-Agent] Found {len(result.facts)} facts."
        )
        return result.model_dump_json()

    @kernel.tool(requires_capability="run_math")
    async def run_graph_math(facts_json: str) -> str:
        """Pure Python deterministic graph closure."""
        print("\n  🧮 [Math Tool] Computing multi-hop graph closure...")

        try:
            facts_payload = json.loads(facts_json)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid facts_json format."})

        raw_facts = facts_payload.get("facts") if isinstance(facts_payload, dict) else None
        if not isinstance(raw_facts, list):
            return json.dumps({"error": "Invalid facts payload shape."})

        try:
            facts = [Triplet.model_validate(item) for item in raw_facts]
        except Exception as exc:
            return json.dumps({"error": f"Invalid fact schema: {exc!s}"})

        inferred: list[Triplet] = []
        for first in facts:
            for second in facts:
                if (
                    first.dst == second.src
                    and first.src != second.dst
                    and first.relation.upper() == "INHIBITS"
                    and second.relation.upper() == "ACTIVATES"
                ):
                    inferred.append(
                        Triplet(src=first.src, relation="REDUCES", dst=second.dst)
                    )

        print(f"  🧮 [Math Tool] Derived {len(inferred)} mathematical relations.")
        return json.dumps({"derived_relations": [item.model_dump() for item in inferred]})

    @kernel.tool(requires_capability="spawn_adjudicator")
    async def run_adjudicator_agent(
        text: str,
        derived_json: str,
        artana_context: ToolExecutionContext,
    ) -> str:
        """Sub-agent: LLM-based adjudication of inferred math."""
        print(
            "\n  👩‍🔬 [Adjudicator Sub-Agent] Reviewing derived math against context..."
        )

        tenant = TenantContext(
            tenant_id=artana_context.tenant_id,
            budget_usd_limit=5.0,
            capabilities=frozenset(),
        )
        agent = AutonomousAgent(
            kernel,
            context_builder=ContextBuilder(progressive_skills=False),
        )

        prompt = (
            f"Original text: {text}\n\n"
            f"Mathematically derived relations: {derived_json}\n\n"
            "Evaluate if these derived relations make biological sense based strictly on the text."
        )

        result = await agent.run(
            run_id=f"{artana_context.run_id}_adj_{artana_context.idempotency_key[:6]}",
            tenant=tenant,
            model=adjudicator_model,
            system_prompt=(
                "You are a strict Biological Adjudicator. "
                "Reject relations that ignore nuance."
            ),
            prompt=prompt,
            output_schema=AdjudicationResult,
            max_iterations=5,
        )
        valid_relations = [entry for entry in result.evaluations if entry.is_valid]
        print(
            f"  👩‍🔬 [Adjudicator Sub-Agent] Approved {len(valid_relations)} relations."
        )
        return json.dumps(
            {"verified_relations": [item.triplet.model_dump() for item in valid_relations]}
        )

    return kernel


async def main() -> None:
    require_openai_api_key(script_name="06_triplets_swarm.py")
    lead_model = resolve_model(env_var="ARTANA_MODEL_LEAD", default="gpt-4o")
    extractor_model = resolve_model(env_var="ARTANA_MODEL_EXTRACTOR", default="gpt-4o-mini")
    adjudicator_model = resolve_model(env_var="ARTANA_MODEL_ADJUDICATOR", default="gpt-4o")
    print_example_header(
        title="06 - Triplets Swarm (Sub-Agent Runtime)",
        models={
            "lead": lead_model,
            "extractor": extractor_model,
            "adjudicator": adjudicator_model,
        },
    )

    db_path = Path("examples/.state_triplets_swarm.db")
    if db_path.exists():
        db_path.unlink()

    kernel = _create_kernel(
        db_path,
        extractor_model=extractor_model,
        adjudicator_model=adjudicator_model,
    )

    tenant = TenantContext(
        tenant_id="science_team",
        budget_usd_limit=5.00,
        capabilities=frozenset({"spawn_extractor", "run_math", "spawn_adjudicator"}),
    )
    lead_agent = AutonomousAgent(
        kernel,
        context_builder=ContextBuilder(progressive_skills=False),
    )

    article_text = (
        "DrugA strongly INHIBITS GeneB. In most patients, GeneB ACTIVATES DiseaseC. "
        "However, this text does not account for mutant variants."
    )

    system_prompt = (
        "You are the Lead Biomedical Analyst. You must coordinate a team to analyze a paper.\n"
        "Step 1: Use run_extractor_agent to get explicit facts.\n"
        "Step 2: Pass those facts to run_graph_math to find derived relations.\n"
        "Step 3: Pass the original text and the derived relations to run_adjudicator_agent.\n"
        "Step 4: Output the FinalReport combining the facts and verified relations."
    )

    print("👔 [Lead Agent] Starting orchestration...\n")
    try:
        report = await lead_agent.run(
            run_id="paper_analysis_swarm_01",
            tenant=tenant,
            model=lead_model,
            system_prompt=system_prompt,
            prompt=f"Please analyze this text: {article_text}",
            output_schema=FinalReport,
            max_iterations=16,
        )
        print_summary(
            payload={
                "run_id": "paper_analysis_swarm_01",
                "lead_model": lead_model,
                "extractor_model": extractor_model,
                "adjudicator_model": adjudicator_model,
                "explicit_fact_count": len(report.explicit_facts),
                "verified_relation_count": len(report.verified_derived_relations),
                "report": report.model_dump(),
            }
        )
    finally:
        await kernel.close()
        if db_path.exists():
            db_path.unlink()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        raise friendly_exit(script_name="06_triplets_swarm.py", error=exc) from exc
