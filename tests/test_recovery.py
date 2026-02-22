from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.events import ToolCompletedPayload
from artana.kernel import ArtanaKernel, ToolExecutionFailedError
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage, ToolCall
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class ToolCallingModelPort:
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
            tool_calls=(
                ToolCall(
                    tool_name="submit_transfer",
                    arguments_json='{"account_id":"acc_1","amount":"10"}',
                ),
            ),
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
    model_port = ToolCallingModelPort()
    tool_attempts = [0]

    first_store = SQLiteStore(str(database_path))
    first_kernel = ArtanaKernel(store=first_store, model_port=model_port)
    register_submit_transfer_tool(first_kernel, tool_attempts)

    with pytest.raises(ToolExecutionFailedError, match="unknown outcome"):
        await first_kernel.chat(
            run_id="run_recovery",
            prompt="Transfer funds",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )

    first_events = await first_store.get_events_for_run("run_recovery")
    assert [event.event_type for event in first_events] == [
        "model_requested",
        "model_completed",
        "tool_requested",
        "tool_completed",
    ]
    completed_payload = first_events[-1].payload
    assert isinstance(completed_payload, ToolCompletedPayload)
    assert completed_payload.outcome == "unknown_outcome"
    assert model_port.calls == 1
    assert tool_attempts[0] == 1
    await first_kernel.close()

    second_store = SQLiteStore(str(database_path))
    second_kernel = ArtanaKernel(store=second_store, model_port=model_port)
    register_submit_transfer_tool(second_kernel, tool_attempts)

    try:
        with pytest.raises(ToolExecutionFailedError, match="outcome='unknown_outcome'"):
            await second_kernel.chat(
                run_id="run_recovery",
                prompt="Transfer funds",
                model="gpt-4o-mini",
                tenant=tenant,
                output_schema=Decision,
            )
        assert model_port.calls == 1
        assert tool_attempts[0] == 1

        events = await second_store.get_events_for_run("run_recovery")
        assert [event.event_type for event in events] == [
            "model_requested",
            "model_completed",
            "tool_requested",
            "tool_completed",
        ]
    finally:
        await second_kernel.close()
