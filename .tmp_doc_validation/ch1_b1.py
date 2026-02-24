import asyncio
from typing import TypeVar

from pydantic import BaseModel

from artana.agent import SingleStepModelClient
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputT = TypeVar("OutputT", bound=BaseModel)


class HelloResult(BaseModel):
    message: str


class DemoModelPort:
    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        output = request.output_schema.model_validate(
            {"message": "Hello from Artana!"}
        )
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=5, completion_tokens=5, cost_usd=0.0),
        )


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("step1.db"),
        model_port=DemoModelPort(),
    )

    tenant = TenantContext(
        tenant_id="demo_user",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    client = SingleStepModelClient(kernel=kernel)

    result = await client.step(
        run_id="hello_run",
        tenant=tenant,
        model="demo-model",
        prompt="Say hello",
        output_schema=HelloResult,
        step_key="hello_step",  # 🔑 required for replay safety
    )

    print(result.output)
    await kernel.close()


asyncio.run(main())