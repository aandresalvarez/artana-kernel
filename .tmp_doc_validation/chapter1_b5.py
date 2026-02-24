import asyncio
from pydantic import BaseModel

from artana.agent import AutonomousAgent
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore


class Report(BaseModel):
    text: str


class DemoModelPort:
    async def complete(self, request):
        return type(request).output_schema.model_validate({"text": "Demo report"})


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("agent.db"),
        model_port=DemoModelPort(),
    )

    tenant = TenantContext(
        tenant_id="agent_user",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    agent = AutonomousAgent(kernel=kernel)

    result = await agent.run(
        run_id="agent_run",
        tenant=tenant,
        model="demo-model",
        prompt="Write a short report",
        output_schema=Report,
    )

    print(result.text)
    await kernel.close()


asyncio.run(main())