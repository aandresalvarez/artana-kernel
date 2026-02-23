"""
Single-step fact extraction from articles as subject–predicate–object triplets.

Uses one model call: article text + extraction instructions → structured triplets.
All execution is event-sourced and replay-safe via the Artana kernel.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel

from artana import ArtanaKernel, KernelModelClient, KernelPolicy, TenantContext
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore


class Triplet(BaseModel):
    """A single fact: subject – predicate – object (e.g. A is connected to B)."""

    subject: str
    predicate: str
    object: str


class ExtractedFacts(BaseModel):
    """Structured output: list of triplets extracted from the article."""

    triplets: list[Triplet]


EXTRACTION_INSTRUCTIONS = """Extract factual triplets from the article below.
Each triplet has:
- subject: the entity or concept that the fact is about
- predicate: a short verb phrase describing the relation (e.g. "is connected to", "works at", "located in")
- object: the other entity or value

Output only valid facts that are explicitly stated or clearly implied. One triplet per fact.
"""

SAMPLE_ARTICLE = """
Berlin is the capital of Germany. The city was divided during the Cold War; the Berlin Wall
fell in 1989. Angela Merkel grew up in East Germany and later became Chancellor of Germany.
She studied physics at the University of Leipzig. The European Union has its roots in the
European Coal and Steel Community, founded in 1951. Brussels serves as the de facto capital
of the European Union.
"""


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Load environment variables first.")

    database_path = Path("examples/.state_03_fact_extraction.db")
    if database_path.exists():
        database_path.unlink()

    store = SQLiteStore(str(database_path))
    kernel = ArtanaKernel(
        store=store,
        model_port=LiteLLMAdapter(
            timeout_seconds=30.0,
            max_retries=1,
            fail_on_unknown_cost=True,
        ),
        middleware=ArtanaKernel.default_middleware_stack(),
        policy=KernelPolicy.enforced(),
    )

    tenant = TenantContext(
        tenant_id="org_fact_extraction",
        capabilities=frozenset(),
        budget_usd_limit=0.20,
    )

    try:
        run = await kernel.start_run(tenant=tenant)
        prompt = f"{EXTRACTION_INSTRUCTIONS}\n\n---\n\nArticle:\n{SAMPLE_ARTICLE.strip()}"

        result = await KernelModelClient(kernel=kernel).chat(
            run_id=run.run_id,
            tenant=tenant,
            model="gpt-4o-mini",
            prompt=prompt,
            output_schema=ExtractedFacts,
            step_key="extract_facts",
        )

        print("Run id:", run.run_id)
        print("Extracted triplets:")
        for i, t in enumerate(result.output.triplets, 1):
            print(f"  {i}. ({t.subject!r} -- {t.predicate!r} --> {t.object!r})")
        print(
            "Usage:",
            {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "cost_usd": result.usage.cost_usd,
            },
        )
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
