import asyncio

from pydantic import BaseModel

from artana.kernel import ArtanaKernel, WorkflowContext, json_step_serde
from artana.agent import SingleStepModelClient
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore


class Intent(BaseModel):
    question: str


class Email(BaseModel):
    body: str


class HybridModel:
    async def complete(self, request: ModelRequest[BaseModel]) -> ModelResult[BaseModel]:
        if "question" in request.output_schema.model_fields:
            output = request.output_schema.model_validate({"question": "What is revenue?"})
        else:
            output = request.output_schema.model_validate({"body": "Revenue is $8.3M."})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=5, completion_tokens=5, cost_usd=0.0),
        )


async def heavy_math():
    return {"revenue": 8300000}


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter3_step4.db"),
        model_port=HybridModel(),
    )

    tenant = TenantContext(
        tenant_id="finance",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    client = SingleStepModelClient(kernel=kernel)

    async def workflow(ctx: WorkflowContext):
        intent = await client.step(
            run_id=ctx.run_id,
            tenant=ctx.tenant,
            model="demo-model",
            prompt="Extract intent",
            output_schema=Intent,
            step_key="intent",
        )

        math = await ctx.step(
            name="compute",
            action=heavy_math,
            serde=json_step_serde(),
        )

        email = await client.step(
            run_id=ctx.run_id,
            tenant=ctx.tenant,
            model="demo-model",
            prompt=f"{intent.output.question}. Revenue: {math['revenue']}",
            output_schema=Email,
            step_key="email",
        )

        return email.output.body

    result = await kernel.run_workflow(
        run_id="hybrid_run",
        tenant=tenant,
        workflow=workflow,
    )

    print(result.output)
    await kernel.close()


asyncio.run(main())