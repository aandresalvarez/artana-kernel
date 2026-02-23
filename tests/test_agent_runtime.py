from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import AgentRuntime, AgentRuntimeState, ArtanaKernel
from artana.events import ChatMessage
from artana.models import TenantContext
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
            usage=ModelUsage(prompt_tokens=3, completion_tokens=2, cost_usd=0.01),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_agent_runtime",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


@pytest.mark.asyncio
async def test_agent_runtime_run_turn_appends_assistant_message(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    runtime = AgentRuntime(kernel=kernel)

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_agent_turn")
        result = await runtime.run_turn(
            run_id="run_agent_turn",
            tenant=_tenant(),
            model="gpt-4o-mini",
            output_schema=Decision,
            state=AgentRuntimeState(
                messages=(ChatMessage(role="user", content="approve transfer?"),)
            ),
        )
        assert model_port.calls == 1
        assert result.state.turn_index == 1
        assert result.state.messages[-1].role == "assistant"
        assert '"approved":true' in result.state.messages[-1].content
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_agent_runtime_run_until_stops_on_policy(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    runtime = AgentRuntime(kernel=kernel)

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_agent_loop")
        result = await runtime.run_until(
            run_id="run_agent_loop",
            tenant=_tenant(),
            model="gpt-4o-mini",
            output_schema=Decision,
            initial_messages=(ChatMessage(role="user", content="start"),),
            should_continue=lambda runtime_result: runtime_result.state.turn_index < 2,
            max_turns=4,
        )
        assert model_port.calls == 2
        assert result.state.turn_index == 2
    finally:
        await kernel.close()
