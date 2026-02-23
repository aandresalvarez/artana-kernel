from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import KernelModelClient
from artana.events import EventType
from artana.kernel import ArtanaKernel
from artana.middleware import CapabilityGuardMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage, ToolCall
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class FakeModelPort:
    def __init__(self) -> None:
        self.calls = 0
        self.allowed_tool_names: list[list[str]] = []

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        self.allowed_tool_names.append([tool.name for tool in request.allowed_tools])
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=12, completion_tokens=4, cost_usd=0.01),
        )


class FakeModelPortWithToolCall:
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


@pytest.mark.asyncio
async def test_chat_replays_completed_model_response(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = FakeModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    tenant = TenantContext(
        tenant_id="org_1",
        capabilities=frozenset({"finance:read"}),
        budget_usd_limit=1.0,
    )

    try:
        first = await KernelModelClient(kernel=kernel).step(
            run_id="run_1",
            prompt="Should we transfer?",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
        second = await KernelModelClient(kernel=kernel).step(
            run_id="run_1",
            prompt="Should we transfer?",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )

        assert first.replayed is False
        assert second.replayed is True
        assert model_port.calls == 1

        events = await store.get_events_for_run("run_1")
        assert [event.event_type for event in events] == [
            "run_started",
            "model_requested",
            "model_completed",
            "run_summary",
        ]
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_chat_filters_tools_by_capability(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = FakeModelPort()
    kernel = ArtanaKernel(
        store=store,
        model_port=model_port,
        middleware=[CapabilityGuardMiddleware()],
    )

    @kernel.tool(requires_capability="finance:read")
    async def get_balance(account_id: str) -> str:
        return '{"balance":"10000"}'

    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        return '{"status":"submitted"}'

    tenant = TenantContext(
        tenant_id="org_2",
        capabilities=frozenset({"finance:read"}),
        budget_usd_limit=1.0,
    )

    try:
        await KernelModelClient(kernel=kernel).step(
            run_id="run_2",
            prompt="Get account summary",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
        assert model_port.allowed_tool_names == [["get_balance"]]
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_pause_persists_pause_event(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = FakeModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    tenant = TenantContext(
        tenant_id="org_3",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    try:
        await KernelModelClient(kernel=kernel).step(
            run_id="run_3",
            prompt="Approve transfer?",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )

        ticket = await kernel.pause(
            run_id="run_3",
            tenant=tenant,
            reason="Need manager sign-off",
        )
        assert ticket.run_id == "run_3"
        events = await store.get_events_for_run("run_3")
        assert events[-1].event_type == EventType.PAUSE_REQUESTED
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_chat_replays_tools_without_reexecuting_completed_tool(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = FakeModelPortWithToolCall()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    tool_invocations = 0

    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        nonlocal tool_invocations
        tool_invocations += 1
        return f'{{"account_id":"{account_id}","amount":"{amount}","status":"submitted"}}'

    tenant = TenantContext(
        tenant_id="org_4",
        capabilities=frozenset({"finance:write"}),
        budget_usd_limit=1.0,
    )

    try:
        first = await KernelModelClient(kernel=kernel).step(
            run_id="run_4",
            prompt="Execute transfer",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
        second = await KernelModelClient(kernel=kernel).step(
            run_id="run_4",
            prompt="Execute transfer",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )

        assert first.replayed is False
        assert second.replayed is True
        assert model_port.calls == 1
        assert len(first.tool_calls) == 1
        assert first.tool_calls[0].tool_name == "submit_transfer"
        assert tool_invocations == 0

        events = await store.get_events_for_run("run_4")
        assert [event.event_type for event in events] == [
            "run_started",
            "model_requested",
            "model_completed",
            "run_summary",
        ]
    finally:
        await kernel.close()
