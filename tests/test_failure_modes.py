from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import KernelModelClient
from artana.events import EventPayload, EventType, KernelEvent
from artana.kernel import ArtanaKernel, CapabilityDeniedError
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage, ToolCall
from artana.store import SQLiteStore
from artana.store.base import EventStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class TransferArgs(BaseModel):
    account_id: str
    amount: str


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
                    arguments_json='{"account_id":"acc_1","amount":"10"}',
                ),
            ),
        )


class CountingModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.001),
        )


class FailingStore(EventStore):
    async def append_event(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
    ) -> KernelEvent:
        raise RuntimeError("simulated store write failure")

    async def get_events_for_run(self, run_id: str) -> list[KernelEvent]:
        return []

    async def verify_run_chain(self, run_id: str) -> bool:
        return True

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_chat_does_not_execute_unauthorized_tool_call_implicitly(
    tmp_path: Path,
) -> None:
    model_port = ToolCallingModelPort()
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=model_port)

    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        return '{"status":"submitted"}'

    tenant = TenantContext(
        tenant_id="org_denied",
        capabilities=frozenset({"finance:read"}),
        budget_usd_limit=1.0,
    )
    try:
        response = await KernelModelClient(kernel=kernel).chat(
            run_id="run_denied",
            prompt="Transfer money",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "submit_transfer"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_execute_tool_denies_missing_capability(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state_exec.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)

    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        return '{"status":"submitted"}'

    tenant = TenantContext(
        tenant_id="org_exec_denied",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )
    try:
        await kernel.start_run(tenant=tenant, run_id="run_exec_denied")
        with pytest.raises(CapabilityDeniedError):
            await kernel.step_tool(
                run_id="run_exec_denied",
                tenant=tenant,
                tool_name="submit_transfer",
                arguments=TransferArgs(account_id="acc_1", amount="10"),
            )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_store_failure_prevents_model_execution() -> None:
    store = FailingStore()
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    tenant = TenantContext(
        tenant_id="org_store_fail",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    try:
        with pytest.raises(RuntimeError, match="simulated store write failure"):
            await KernelModelClient(kernel=kernel).chat(
                run_id="run_store_fail",
                prompt="hello",
                model="gpt-4o-mini",
                tenant=tenant,
                output_schema=Decision,
            )
        assert model_port.calls == 0
    finally:
        await kernel.close()
