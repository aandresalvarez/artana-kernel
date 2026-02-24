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
from artana.ports.model import ModelRequest, ModelResult, ModelUsage, ToolCall
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class TransferDecision(BaseModel):
    approved: bool
    reason: str


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
            tool_calls=(
                ToolCall(
                    tool_name="submit_transfer",
                    arguments_json='{"account_id":"acc_1","amount":"10"}',
                    tool_call_id="submit_transfer_call_1",
                ),
            ),
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

    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        transfer_tool_calls[0] += 1
        return (
            '{"status":"submitted","account_id":"'
            + account_id
            + '","amount":"'
            + amount
            + '"}'
        )

    tenant = TenantContext(
        tenant_id="org_demo",
        capabilities=frozenset({"finance:write"}),
        budget_usd_limit=1.0,
    )

    try:
        first = await KernelModelClient(kernel=kernel).step(
            run_id="example_run_1",
            prompt="Transfer 10 from acc_1. My email is user@example.com",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=TransferDecision,
        )
        second = await KernelModelClient(kernel=kernel).step(
            run_id="example_run_1",
            prompt="Transfer 10 from acc_1. My email is user@example.com",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=TransferDecision,
        )
        events = await store.get_events_for_run("example_run_1")

        print("First call replayed:", first.replayed)
        print("Second call replayed:", second.replayed)
        print("Model calls:", model_port.calls)
        print("Tool calls:", transfer_tool_calls[0])
        print("Decision:", first.output.model_dump())
        print("Event types:", [event.event_type for event in events])
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
