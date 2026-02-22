from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.events import ChatMessage, ModelRequestedPayload
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage, ToolCall
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class ToolCallingModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=5, completion_tokens=2, cost_usd=0.01),
            tool_calls=(
                ToolCall(
                    tool_name="submit_transfer",
                    arguments_json='{"account_id":"acc_7","amount":"15"}',
                ),
            ),
        )


class PlainModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=5, completion_tokens=2, cost_usd=0.01),
        )


def register_flaky_submit_tool(kernel: ArtanaKernel, attempts: list[int]) -> None:
    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        attempts[0] += 1
        if attempts[0] == 1:
            raise RuntimeError("simulated tool failure")
        return f'{{"account_id":"{account_id}","amount":"{amount}","status":"submitted"}}'


@pytest.mark.asyncio
async def test_execute_tool_replays_completed_call_without_reexecution(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=PlainModelPort())
    tenant = TenantContext(
        tenant_id="org_exec",
        capabilities=frozenset({"finance:write"}),
        budget_usd_limit=1.0,
    )
    executions = 0

    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        nonlocal executions
        executions += 1
        return f'{{"account_id":"{account_id}","amount":"{amount}","status":"submitted"}}'

    try:
        first = await kernel.execute_tool(
            run_id="run_exec",
            tenant=tenant,
            tool_name="submit_transfer",
            arguments_json='{"account_id":"acc_1","amount":"10"}',
        )
        second = await kernel.execute_tool(
            run_id="run_exec",
            tenant=tenant,
            tool_name="submit_transfer",
            arguments_json='{"account_id":"acc_1","amount":"10"}',
        )
        assert first == second
        assert executions == 1
        events = await store.get_events_for_run("run_exec")
        assert [event.event_type for event in events] == [
            "tool_requested",
            "tool_completed",
        ]
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_execute_tool_resumes_pending_request_after_failure(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    tenant = TenantContext(
        tenant_id="org_exec_pending",
        capabilities=frozenset({"finance:write"}),
        budget_usd_limit=1.0,
    )
    attempts = [0]

    first_store = SQLiteStore(str(database_path))
    first_kernel = ArtanaKernel(store=first_store, model_port=PlainModelPort())
    register_flaky_submit_tool(first_kernel, attempts)

    with pytest.raises(RuntimeError, match="simulated tool failure"):
        await first_kernel.execute_tool(
            run_id="run_exec_pending",
            tenant=tenant,
            tool_name="submit_transfer",
            arguments_json='{"account_id":"acc_2","amount":"20"}',
        )
    await first_kernel.close()

    second_store = SQLiteStore(str(database_path))
    second_kernel = ArtanaKernel(store=second_store, model_port=PlainModelPort())
    register_flaky_submit_tool(second_kernel, attempts)

    try:
        result = await second_kernel.execute_tool(
            run_id="run_exec_pending",
            tenant=tenant,
            tool_name="submit_transfer",
            arguments_json='{"account_id":"acc_2","amount":"20"}',
        )
        assert '"status":"submitted"' in result
        assert attempts[0] == 2

        events = await second_store.get_events_for_run("run_exec_pending")
        assert [event.event_type for event in events] == [
            "tool_requested",
            "tool_completed",
        ]
    finally:
        await second_kernel.close()


@pytest.mark.asyncio
async def test_resume_reports_run_statuses(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    tenant = TenantContext(
        tenant_id="org_resume",
        capabilities=frozenset({"finance:write"}),
        budget_usd_limit=1.0,
    )

    kernel_complete = ArtanaKernel(store=store, model_port=PlainModelPort())
    try:
        await kernel_complete.chat(
            run_id="run_complete",
            prompt="check",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
        complete_state = await kernel_complete.resume(run_id="run_complete")
        assert complete_state.status == "complete"
    finally:
        await kernel_complete.close()

    store_paused = SQLiteStore(str(tmp_path / "state_paused.db"))
    kernel_paused = ArtanaKernel(store=store_paused, model_port=PlainModelPort())
    try:
        await kernel_paused.chat(
            run_id="run_paused",
            prompt="pause flow",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
        await kernel_paused.pause_for_human(run_id="run_paused", reason="manual review")
        paused_state = await kernel_paused.resume(run_id="run_paused")
        assert paused_state.status == "paused"
        assert paused_state.pause_reason == "manual review"
    finally:
        await kernel_paused.close()

    store_pending = SQLiteStore(str(tmp_path / "state_pending.db"))
    kernel_pending = ArtanaKernel(store=store_pending, model_port=ToolCallingModelPort())

    @kernel_pending.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError):
        await kernel_pending.chat(
            run_id="run_pending",
            prompt="needs tool",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
    pending_state = await kernel_pending.resume(run_id="run_pending")
    try:
        assert pending_state.status == "pending_tool"
        assert pending_state.pending_tool is not None
        assert pending_state.pending_tool.tool_name == "submit_transfer"
    finally:
        await kernel_pending.close()

    store_ready = SQLiteStore(str(tmp_path / "state_ready.db"))
    await store_ready.append_event(
        run_id="run_ready",
        tenant_id=tenant.tenant_id,
        event_type="model_requested",
        payload=ModelRequestedPayload(
            model="gpt-4o-mini",
            prompt="ready",
            messages=[ChatMessage(role="user", content="ready")],
            allowed_tools=[],
        ),
    )
    kernel_ready = ArtanaKernel(store=store_ready, model_port=PlainModelPort())
    try:
        ready_state = await kernel_ready.resume(run_id="run_ready")
        assert ready_state.status == "ready"
    finally:
        await kernel_ready.close()
