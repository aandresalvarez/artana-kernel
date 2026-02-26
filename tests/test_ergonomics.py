from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import (
    AgentRuntime,
    ArtanaKernel,
    AutonomousAgent,
    KernelModelClient,
    MockModelPort,
    TenantContext,
)
from artana.events import EventType, ModelRequestedPayload
from artana.middleware import CapabilityGuardMiddleware
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class CountingModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        output = request.output_schema.model_validate(
            {"approved": True, "reason": f"call-{self.calls}"}
        )
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=2, completion_tokens=1, cost_usd=0.001),
        )


def _tenant(*, capabilities: frozenset[str] = frozenset()) -> TenantContext:
    return TenantContext(
        tenant_id="org_ergonomics",
        capabilities=capabilities,
        budget_usd_limit=2.0,
    )


@pytest.mark.asyncio
async def test_client_runtime_and_agent_accept_positional_kernel(tmp_path: Path) -> None:
    kernel = ArtanaKernel(
        store=SQLiteStore(str(tmp_path / "state.db")),
        model_port=MockModelPort(output={"approved": True, "reason": "ok"}),
        middleware=[CapabilityGuardMiddleware()],
    )
    try:
        KernelModelClient(kernel)
        KernelModelClient(kernel=kernel)
        AgentRuntime(kernel)
        AgentRuntime(kernel=kernel)
        AutonomousAgent(kernel)
        AutonomousAgent(kernel=kernel)
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_kernel_model_client_auto_step_key_replays_same_prompt(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    client = KernelModelClient(kernel)

    try:
        first = await client.step(
            run_id="run_auto_step_replay",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="same prompt",
            output_schema=Decision,
        )
        second = await client.step(
            run_id="run_auto_step_replay",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="same prompt",
            output_schema=Decision,
        )
        assert first.replayed is False
        assert second.replayed is True
        assert model_port.calls == 1

        events = await store.get_events_for_run("run_auto_step_replay")
        model_requested = [
            event.payload
            for event in events
            if event.event_type == EventType.MODEL_REQUESTED
            and isinstance(event.payload, ModelRequestedPayload)
        ]
        assert len(model_requested) == 1
        step_key = model_requested[0].step_key
        assert step_key is not None
        assert step_key.startswith("kernelmodelclient_gpt_4o_mini_")
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_kernel_model_client_auto_step_key_changes_with_prompt(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    client = KernelModelClient(kernel)

    try:
        first = await client.step(
            run_id="run_auto_step_change",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="first prompt",
            output_schema=Decision,
        )
        second = await client.step(
            run_id="run_auto_step_change",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="second prompt",
            output_schema=Decision,
        )
        assert first.replayed is False
        assert second.replayed is False
        assert model_port.calls == 2

        events = await store.get_events_for_run("run_auto_step_change")
        model_requested = [
            event.payload
            for event in events
            if event.event_type == EventType.MODEL_REQUESTED
            and isinstance(event.payload, ModelRequestedPayload)
        ]
        assert len(model_requested) == 2
        assert model_requested[0].step_key != model_requested[1].step_key
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_mock_model_port_integrates_with_kernel_client(tmp_path: Path) -> None:
    kernel = ArtanaKernel(
        store=SQLiteStore(str(tmp_path / "state.db")),
        model_port=MockModelPort(output={"approved": True, "reason": "mocked"}),
    )
    try:
        result = await KernelModelClient(kernel).step(
            run_id="run_mock_model_port",
            tenant=_tenant(),
            model="demo-model",
            prompt="approve",
            output_schema=Decision,
            step_key="decision",
        )
        assert result.output.reason == "mocked"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_capability_helpers_expose_tenant_view(tmp_path: Path) -> None:
    kernel = ArtanaKernel(
        store=SQLiteStore(str(tmp_path / "state.db")),
        model_port=MockModelPort(output={"approved": True, "reason": "ok"}),
        middleware=[CapabilityGuardMiddleware()],
    )

    @kernel.tool()
    async def public_lookup(account_id: str) -> str:
        return json.dumps({"ok": True, "account_id": account_id})

    @kernel.tool(requires_capability="payments")
    async def transfer_funds(account_id: str) -> str:
        return json.dumps({"ok": True, "account_id": account_id})

    try:
        tenant = _tenant()
        described = await kernel.describe_capabilities(tenant=tenant)
        raw_decisions = described.get("decisions")
        assert isinstance(raw_decisions, list)
        decisions: dict[str, dict[str, object]] = {}
        for item in raw_decisions:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool_name")
            if isinstance(tool_name, str):
                decisions[tool_name] = item
        assert decisions["public_lookup"].get("decision") == "allowed"
        assert decisions["transfer_funds"].get("decision") == "filtered"

        visible = kernel.list_tools_for_tenant(tenant=tenant)
        assert [tool.name for tool in visible] == ["public_lookup"]
    finally:
        await kernel.close()
