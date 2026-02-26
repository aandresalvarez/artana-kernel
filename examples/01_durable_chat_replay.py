from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from artana import ArtanaKernel, KernelModelClient, TenantContext
from artana.middleware import (
    CapabilityGuardMiddleware,
    PIIScrubberMiddleware,
    QuotaMiddleware,
)
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.ports.tool import ToolExecutionContext
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class TransferDecision(BaseModel):
    approved: bool
    reason: str


class TransferArgs(BaseModel):
    account_id: str
    amount: str


class DemoModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        output = request.output_schema.model_validate(
            {"approved": True, "reason": "Balance check passed."}
        )
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=12, completion_tokens=6, cost_usd=0.01),
        )


async def main() -> None:
    database_path = Path("examples/.state_first_example.db")
    if database_path.exists():
        database_path.unlink()

    store = SQLiteStore(str(database_path))
    model_port = DemoModelPort()
    kernel = ArtanaKernel(
        store=store,
        model_port=model_port,
        middleware=[
            PIIScrubberMiddleware(),
            QuotaMiddleware(),
            CapabilityGuardMiddleware(),
        ],
    )
    transfer_tool_calls = [0]

    @kernel.tool(requires_capability="finance:write", side_effect=True)
    async def submit_transfer(
        account_id: str,
        amount: str,
        artana_context: ToolExecutionContext,
    ) -> str:
        transfer_tool_calls[0] += 1
        return (
            '{"status":"submitted","account_id":"'
            + account_id
            + '","amount":"'
            + amount
            + '","idempotency_key":"'
            + artana_context.idempotency_key
            + '"}'
        )

    tenant = TenantContext(
        tenant_id="org_demo",
        capabilities=frozenset({"finance:write"}),
        budget_usd_limit=1.0,
    )

    try:
        client = KernelModelClient(kernel)
        first = await client.step(
            run_id="example_run_1",
            prompt="Transfer 10 from acc_1. My email is user@example.com",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=TransferDecision,
        )
        second = await client.step(
            run_id="example_run_1",
            prompt="Transfer 10 from acc_1. My email is user@example.com",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=TransferDecision,
        )
        first_tool = await kernel.step_tool(
            run_id="example_run_1",
            tenant=tenant,
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
            step_key="submit_transfer_acc_1_10",
        )
        second_tool = await kernel.step_tool(
            run_id="example_run_1",
            tenant=tenant,
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
            step_key="submit_transfer_acc_1_10",
        )
        events = await store.get_events_for_run("example_run_1")

        print("First model replayed:", first.replayed)
        print("Second model replayed:", second.replayed)
        print("First tool replayed:", first_tool.replayed)
        print("Second tool replayed:", second_tool.replayed)
        print("Model calls:", model_port.calls)
        print("Tool calls:", transfer_tool_calls[0])
        print("Decision:", first.output.model_dump())
        print("Transfer result:", first_tool.result_json)
        print("Event types:", [event.event_type for event in events])

        if not second.replayed:
            raise AssertionError("Expected second model step to replay.")
        if second_tool.replayed is not True:
            raise AssertionError("Expected second tool step to replay.")
        if transfer_tool_calls[0] != 1:
            raise AssertionError("Replay should not execute duplicate tool calls.")
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
