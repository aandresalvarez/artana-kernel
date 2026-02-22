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


class TransferArgs(BaseModel):
    account_id: str
    amount: str


class PlainModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({"ok": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_unknown",
        capabilities=frozenset({"finance:write"}),
        budget_usd_limit=1.0,
    )


def register_failing_submit_transfer(kernel: ArtanaKernel, attempts: list[int]) -> None:
    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        attempts[0] += 1
        raise RuntimeError("network disconnect after submit")


def register_success_submit_transfer(kernel: ArtanaKernel, attempts: list[int]) -> None:
    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        attempts[0] += 1
        return '{"status":"submitted"}'


@pytest.mark.asyncio
async def test_unknown_tool_outcome_is_recorded_and_not_retried_on_replay(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"
    first_store = SQLiteStore(str(database_path))
    first_kernel = ArtanaKernel(store=first_store, model_port=PlainModelPort())
    tool_attempts = [0]
    register_failing_submit_transfer(first_kernel, tool_attempts)

    await first_kernel.start_run(tenant=_tenant(), run_id="run_unknown_tool_outcome")
    with pytest.raises(ToolExecutionFailedError, match="unknown outcome"):
        await first_kernel.step_tool(
            run_id="run_unknown_tool_outcome",
            tenant=_tenant(),
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
        )
    try:
        events = await first_store.get_events_for_run("run_unknown_tool_outcome")
        assert [event.event_type for event in events] == [
            "run_started",
            "tool_requested",
            "tool_completed",
        ]
        tool_completed_payload = events[-1].payload
        assert isinstance(tool_completed_payload, ToolCompletedPayload)
        assert tool_completed_payload.outcome == "unknown_outcome"
    finally:
        await first_kernel.close()

    second_store = SQLiteStore(str(database_path))
    second_kernel = ArtanaKernel(store=second_store, model_port=PlainModelPort())
    register_success_submit_transfer(second_kernel, tool_attempts)

    try:
        with pytest.raises(ToolExecutionFailedError, match="outcome='unknown_outcome'"):
            await second_kernel.step_tool(
                run_id="run_unknown_tool_outcome",
                tenant=_tenant(),
                tool_name="submit_transfer",
                arguments=TransferArgs(account_id="acc_1", amount="10"),
            )

        assert tool_attempts[0] == 1

        reconciled = await second_kernel.reconcile_tool(
            run_id="run_unknown_tool_outcome",
            tenant=_tenant(),
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
        )
        assert '"status":"submitted"' in reconciled
        assert tool_attempts[0] == 2

        replayed_after_reconcile = await second_kernel.step_tool(
            run_id="run_unknown_tool_outcome",
            tenant=_tenant(),
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
        )
        assert replayed_after_reconcile.replayed is True
        assert tool_attempts[0] == 2

        replay_events = await second_store.get_events_for_run("run_unknown_tool_outcome")
        assert len(replay_events) == 4
        payload = replay_events[-1].payload
        assert isinstance(payload, ToolCompletedPayload)
        assert payload.outcome == "success"
    finally:
        await second_kernel.close()
