import asyncio
from pydantic import BaseModel

from artana.kernel import ArtanaKernel, ModelInput
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore


class Report(BaseModel):
    summary: str


class DemoModelPort:
    async def complete(self, request: ModelRequest[Report]) -> ModelResult[Report]:
        return ModelResult(
            output=Report(summary="generated report"),
            usage=ModelUsage(prompt_tokens=4, completion_tokens=3, cost_usd=0.0),
        )


async def generate_report(workflow_id: str, account_id: str) -> str:
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter4_step5.db"),
        model_port=DemoModelPort(),
    )

    tenant = TenantContext(
        tenant_id=account_id,
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    await kernel.start_run(tenant=tenant, run_id=workflow_id)

    result = await kernel.step_model(
        run_id=workflow_id,
        tenant=tenant,
        model="demo-model",
        input=ModelInput.from_prompt(f"Generate report for {account_id}"),
        output_schema=Report,
        step_key="report_step",
    )

    await kernel.close()
    return result.output.summary


async def main():
    print(await generate_report("workflow_123", "acct_42"))


asyncio.run(main())