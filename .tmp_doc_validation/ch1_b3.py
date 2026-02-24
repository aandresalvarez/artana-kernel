import asyncio
import json

from pydantic import BaseModel

from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.store import SQLiteStore
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.ports.tool import ToolExecutionContext


class Decision(BaseModel):
    ok: bool


class TransferArgs(BaseModel):
    amount: int
    to_user: str


class DemoModelPort:
    async def complete(self, request: ModelRequest[Decision]) -> ModelResult[Decision]:
        return ModelResult(
            output=Decision(ok=True),
            usage=ModelUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.0),
        )


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("step2.db"),
        model_port=DemoModelPort(),
    )

    @kernel.tool()
    async def transfer_money(
        amount: int,
        to_user: str,
        artana_context: ToolExecutionContext,
    ) -> str:
        return json.dumps({
            "amount": amount,
            "to_user": to_user,
            "idempotency_key": artana_context.idempotency_key
        })

    tenant = TenantContext(
        tenant_id="demo_user",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    await kernel.start_run(tenant=tenant, run_id="tool_run")

    result = await kernel.step_tool(
        run_id="tool_run",
        tenant=tenant,
        tool_name="transfer_money",
        arguments=TransferArgs(amount=10, to_user="alice"),
        step_key="transfer_step",
    )

    print(result.result_json)
    await kernel.close()


asyncio.run(main())