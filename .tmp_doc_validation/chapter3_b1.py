import asyncio
import json

from pydantic import BaseModel

from artana.kernel import ArtanaKernel, ToolExecutionFailedError
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.ports.tool import ToolExecutionContext, ToolUnknownOutcomeError
from artana.store import SQLiteStore


class NoopOutput(BaseModel):
    ok: bool


class NoopModelPort:
    async def complete(self, request: ModelRequest[NoopOutput]) -> ModelResult[NoopOutput]:
        return ModelResult(
            output=NoopOutput(ok=True),
            usage=ModelUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.0),
        )


class ChargeArgs(BaseModel):
    amount_cents: int
    card_id: str


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter3_step1.db"),
        model_port=NoopModelPort(),
    )

    provider_state = {"first_attempt": True}

    @kernel.tool(requires_capability="payments:charge")
    async def charge_credit_card(
        amount_cents: int,
        card_id: str,
        artana_context: ToolExecutionContext,
    ) -> str:
        if provider_state["first_attempt"]:
            provider_state["first_attempt"] = False
            raise ToolUnknownOutcomeError("network timeout after provider accepted charge")

        return json.dumps({
            "status": "charged",
            "idempotency_key": artana_context.idempotency_key,
        })

    tenant = TenantContext(
        tenant_id="billing_team",
        capabilities=frozenset({"payments:charge"}),
        budget_usd_limit=5.0,
    )

    await kernel.start_run(tenant=tenant, run_id="payment_run")

    args = ChargeArgs(amount_cents=1000, card_id="card_123")

    try:
        await kernel.step_tool(
            run_id="payment_run",
            tenant=tenant,
            tool_name="charge_credit_card",
            arguments=args,
            step_key="charge_step",
        )
    except ToolExecutionFailedError:
        print("Reconciliation required")

    result = await kernel.reconcile_tool(
        run_id="payment_run",
        tenant=tenant,
        tool_name="charge_credit_card",
        arguments=args,
        step_key="charge_step",
    )

    print("Reconciled:", result)
    await kernel.close()


asyncio.run(main())