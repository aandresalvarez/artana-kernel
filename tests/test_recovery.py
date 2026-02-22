from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.events import ToolCompletedPayload
from artana.kernel import ArtanaKernel, ToolExecutionFailedError
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class TransferArgs(BaseModel):
    account_id: str
    amount: str


class PlainModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=12, completion_tokens=4, cost_usd=0.01),
        )


def register_submit_transfer_tool(kernel: ArtanaKernel, attempts: list[int]) -> None:
    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        attempts[0] += 1
        if attempts[0] == 1:
            raise RuntimeError("simulated crash during tool execution")
        return f'{{"account_id":"{account_id}","amount":"{amount}","status":"submitted"}}'


@pytest.mark.asyncio
async def test_unknown_tool_outcome_halts_replay_and_requires_reconciliation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"
    tenant = TenantContext(
        tenant_id="org_recovery",
        capabilities=frozenset({"finance:write"}),
        budget_usd_limit=1.0,
    )
    tool_attempts = [0]

    first_store = SQLiteStore(str(database_path))
    first_kernel = ArtanaKernel(store=first_store, model_port=PlainModelPort())
    register_submit_transfer_tool(first_kernel, tool_attempts)

    await first_kernel.start_run(tenant=tenant, run_id="run_recovery")
    with pytest.raises(ToolExecutionFailedError, match="unknown outcome"):
        await first_kernel.step_tool(
            run_id="run_recovery",
            tenant=tenant,
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
        )

    first_events = await first_store.get_events_for_run("run_recovery")
    assert [event.event_type for event in first_events] == [
        "run_started",
        "tool_requested",
        "tool_completed",
    ]
    completed_payload = first_events[-1].payload
    assert isinstance(completed_payload, ToolCompletedPayload)
    assert completed_payload.outcome == "unknown_outcome"
    assert tool_attempts[0] == 1
    await first_kernel.close()

    second_store = SQLiteStore(str(database_path))
    second_kernel = ArtanaKernel(store=second_store, model_port=PlainModelPort())
    register_submit_transfer_tool(second_kernel, tool_attempts)

    try:
        with pytest.raises(ToolExecutionFailedError, match="outcome='unknown_outcome'"):
            await second_kernel.step_tool(
                run_id="run_recovery",
                tenant=tenant,
                tool_name="submit_transfer",
                arguments=TransferArgs(account_id="acc_1", amount="10"),
            )
        assert tool_attempts[0] == 1

        events = await second_store.get_events_for_run("run_recovery")
        assert [event.event_type for event in events] == [
            "run_started",
            "tool_requested",
            "tool_completed",
        ]
    finally:
        await second_kernel.close()
