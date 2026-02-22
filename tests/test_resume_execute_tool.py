from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.events import ResumeRequestedPayload, ToolCompletedPayload
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


class HumanInput(BaseModel):
    note: str


class PlainModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=5, completion_tokens=2, cost_usd=0.01),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_exec",
        capabilities=frozenset({"finance:write"}),
        budget_usd_limit=1.0,
    )


def register_failing_submit_transfer(kernel: ArtanaKernel, attempts: list[int]) -> None:
    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        attempts[0] += 1
        raise RuntimeError("simulated tool failure")


def register_success_submit_transfer(kernel: ArtanaKernel, attempts: list[int]) -> None:
    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        attempts[0] += 1
        return f'{{"account_id":"{account_id}","amount":"{amount}","status":"submitted"}}'


@pytest.mark.asyncio
async def test_step_tool_replays_completed_call_without_reexecution(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=PlainModelPort())
    executions = 0

    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        nonlocal executions
        executions += 1
        return f'{{"account_id":"{account_id}","amount":"{amount}","status":"submitted"}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_exec")
        first = await kernel.step_tool(
            run_id="run_exec",
            tenant=_tenant(),
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
            step_key="transfer",
        )
        second = await kernel.step_tool(
            run_id="run_exec",
            tenant=_tenant(),
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
            step_key="transfer",
        )
        assert first.result_json == second.result_json
        assert first.replayed is False
        assert second.replayed is True
        assert first.seq == second.seq
        assert executions == 1
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_step_tool_records_unknown_outcome_and_halts_replay(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    attempts = [0]

    first_store = SQLiteStore(str(database_path))
    first_kernel = ArtanaKernel(store=first_store, model_port=PlainModelPort())
    register_failing_submit_transfer(first_kernel, attempts)

    try:
        await first_kernel.start_run(tenant=_tenant(), run_id="run_exec_pending")
        with pytest.raises(ToolExecutionFailedError, match="unknown outcome"):
            await first_kernel.step_tool(
                run_id="run_exec_pending",
                tenant=_tenant(),
                tool_name="submit_transfer",
                arguments=TransferArgs(account_id="acc_2", amount="20"),
                step_key="transfer",
            )
    finally:
        await first_kernel.close()

    second_store = SQLiteStore(str(database_path))
    second_kernel = ArtanaKernel(store=second_store, model_port=PlainModelPort())
    register_success_submit_transfer(second_kernel, attempts)

    try:
        with pytest.raises(ToolExecutionFailedError, match="outcome='unknown_outcome'"):
            await second_kernel.step_tool(
                run_id="run_exec_pending",
                tenant=_tenant(),
                tool_name="submit_transfer",
                arguments=TransferArgs(account_id="acc_2", amount="20"),
                step_key="transfer",
            )
        assert attempts[0] == 1

        events = await second_store.get_events_for_run("run_exec_pending")
        payload = events[-1].payload
        assert isinstance(payload, ToolCompletedPayload)
        assert payload.outcome == "unknown_outcome"
    finally:
        await second_kernel.close()


@pytest.mark.asyncio
async def test_resume_appends_boundary_event(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=PlainModelPort())

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_resume")
        resumed = await kernel.resume(
            run_id="run_resume",
            tenant=_tenant(),
            human_input=HumanInput(note="approved"),
        )
        assert resumed.run_id == "run_resume"
        assert resumed.tenant_id == _tenant().tenant_id

        events = await store.get_events_for_run("run_resume")
        payload = events[-1].payload
        assert isinstance(payload, ResumeRequestedPayload)
        assert payload.human_input_json == '{"note":"approved"}'
    finally:
        await kernel.close()
