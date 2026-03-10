from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

from _live_example_utils import (
    friendly_exit,
    print_example_header,
    print_summary,
    require_openai_api_key,
    resolve_model,
)
from pydantic import BaseModel

from artana import ArtanaKernel, KernelModelClient, ModelCallOptions, TenantContext
from artana.harness import BaseHarness, HarnessContext
from artana.ports.model import LiteLLMAdapter
from artana.store import SQLiteStore

GENE_DB: dict[str, str] = {
    "MED13": "MED13 is a Mediator complex subunit involved in transcription regulation.",
    "TP53": "TP53 is a tumor suppressor gene involved in genome integrity.",
    "BRCA1": "BRCA1 is associated with DNA repair.",
}


class LookupGeneArgs(BaseModel):
    gene_name: str


class ConciseAnswer(BaseModel):
    answer: str


class HarnessCaseResult(BaseModel):
    name: str
    passed: bool
    output: str
    missing_terms: list[str]


@dataclass(frozen=True, slots=True)
class TestCase:
    name: str
    user_input: str
    must_include: tuple[str, ...]


def _detect_gene(text: str) -> str | None:
    for token in re.findall(r"[A-Za-z0-9]+", text.upper()):
        if token in GENE_DB:
            return token
    return None


def _missing_tool_context() -> dict[str, object]:
    return {
        "gene_name": None,
        "found": False,
        "summary": "Missing curated gene lookup context.",
    }


def _answer_prompt(*, user_input: str, tool_context: dict[str, object]) -> str:
    return (
        "Return only JSON for schema {answer:string}.\n"
        "You are a concise biomedical assistant.\n"
        "Use the provided tool context when it is relevant.\n"
        "When tool_context.found is false, the answer must contain the word 'missing'.\n"
        "When tool_context.found is true, reuse tool_context.gene_name exactly.\n\n"
        f"User question: {user_input}\n"
        f"Tool context: {json.dumps(tool_context, sort_keys=True)}"
    )


async def _ensure_run_exists(
    *,
    kernel: ArtanaKernel,
    run_id: str,
    tenant: TenantContext,
) -> None:
    try:
        await kernel.load_run(run_id=run_id, tenant=tenant)
    except ValueError:
        await kernel.start_run(tenant=tenant, run_id=run_id)


async def run_manual_agent(
    *,
    kernel: ArtanaKernel,
    tenant: TenantContext,
    run_id: str,
    model: str,
    user_input: str,
) -> ConciseAnswer:
    await _ensure_run_exists(kernel=kernel, run_id=run_id, tenant=tenant)

    tool_context = _missing_tool_context()
    gene_name = _detect_gene(user_input)
    if gene_name is not None:
        tool_result = await kernel.step_tool(
            run_id=run_id,
            tenant=tenant,
            tool_name="lookup_gene",
            arguments=LookupGeneArgs(gene_name=gene_name),
            step_key=f"manual_lookup_{gene_name.lower()}",
        )
        tool_context = json.loads(tool_result.result_json)

    model_result = await KernelModelClient(kernel).step(
        run_id=run_id,
        tenant=tenant,
        model=model,
        prompt=_answer_prompt(user_input=user_input, tool_context=tool_context),
        output_schema=ConciseAnswer,
        step_key="manual_agent_answer",
        model_options=ModelCallOptions(api_mode="auto"),
    )
    return model_result.output


class GeneLookupHarness(BaseHarness[list[HarnessCaseResult]]):
    def __init__(
        self,
        kernel: ArtanaKernel,
        *,
        cases: tuple[TestCase, ...],
        tenant: TenantContext,
        default_model: str,
    ) -> None:
        super().__init__(
            kernel=kernel,
            tenant=tenant,
            default_model=default_model,
            replay_policy="strict",
        )
        self._cases = cases

    async def step(self, *, context: HarnessContext) -> list[HarnessCaseResult]:
        results: list[HarnessCaseResult] = []

        for index, case in enumerate(self._cases, start=1):
            step_prefix = f"case_{index}"
            tool_context = _missing_tool_context()
            gene_name = _detect_gene(case.user_input)

            if gene_name is not None:
                tool_result = await self.run_tool(
                    tool_name="lookup_gene",
                    arguments=LookupGeneArgs(gene_name=gene_name),
                    step_key=f"{step_prefix}_lookup",
                )
                tool_context = json.loads(tool_result.result_json)

            answer_result = await self.run_model(
                prompt=_answer_prompt(
                    user_input=case.user_input,
                    tool_context=tool_context,
                ),
                output_schema=ConciseAnswer,
                step_key=f"{step_prefix}_answer",
                model_options=ModelCallOptions(api_mode="auto"),
            )
            answer = answer_result.output.answer
            missing_terms = [
                term for term in case.must_include if term.lower() not in answer.lower()
            ]
            results.append(
                HarnessCaseResult(
                    name=case.name,
                    passed=len(missing_terms) == 0,
                    output=answer,
                    missing_terms=missing_terms,
                )
            )

        return results


async def main() -> None:
    require_openai_api_key(script_name="10_live_manual_agent_harness.py")
    model_name = resolve_model(
        env_var="ARTANA_HARNESS_MODEL",
        default="openai/gpt-5.4",
    )
    print_example_header(
        title="10 - Manual Agent + Harness (GPT-5.4)",
        models={"manual_agent": model_name, "harness": model_name},
    )

    database_path = Path("examples/.state_live_manual_agent_harness.db")
    if database_path.exists():
        database_path.unlink()

    store = SQLiteStore(str(database_path))
    kernel = ArtanaKernel(
        store=store,
        model_port=LiteLLMAdapter(timeout_seconds=30.0, max_retries=1),
        middleware=ArtanaKernel.default_middleware_stack(),
    )

    @kernel.tool()
    async def lookup_gene(gene_name: str) -> str:
        canonical = gene_name.upper()
        summary = GENE_DB.get(canonical)
        if summary is None:
            return json.dumps(
                {
                    "gene_name": canonical,
                    "found": False,
                    "summary": f"Missing curated lookup for {canonical}.",
                }
            )
        return json.dumps(
            {
                "gene_name": canonical,
                "found": True,
                "summary": summary,
            }
        )

    tenant = TenantContext(
        tenant_id="org_gene_harness",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )
    cases = (
        TestCase(
            name="MED13 lookup",
            user_input="What is MED13?",
            must_include=("MED13",),
        ),
        TestCase(
            name="TP53 lookup",
            user_input="Tell me about TP53 in one sentence.",
            must_include=("TP53",),
        ),
        TestCase(
            name="Unknown concept",
            user_input="What is ABCXYZ123?",
            must_include=("missing",),
        ),
    )

    try:
        manual_agent_output = await run_manual_agent(
            kernel=kernel,
            tenant=tenant,
            run_id="manual_agent_run",
            model=model_name,
            user_input="What is MED13?",
        )

        harness = GeneLookupHarness(
            kernel=kernel,
            cases=cases,
            tenant=tenant,
            default_model=model_name,
        )
        harness_results = await harness.run(run_id="gene_harness_run")
        passed = sum(1 for result in harness_results if result.passed)

        print_summary(
            payload={
                "model": model_name,
                "manual_agent": manual_agent_output.model_dump(),
                "harness_score": f"{passed}/{len(harness_results)}",
                "harness_results": [
                    result.model_dump(mode="json") for result in harness_results
                ],
            }
        )
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        raise friendly_exit(
            script_name="10_live_manual_agent_harness.py",
            error=exc,
        ) from exc
