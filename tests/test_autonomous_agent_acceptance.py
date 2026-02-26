from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import ArtanaKernel
from artana.acceptance import AcceptanceSpec, ToolGate
from artana.agent import AutonomousAgent, DraftVerifyLoopConfig
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class AgentResult(BaseModel):
    done: bool


class LoopAwareModelPort:
    def __init__(self) -> None:
        self.models_seen: list[str] = []

    async def complete(self, request: ModelRequest[OutputModelT]) -> ModelResult[OutputModelT]:
        self.models_seen.append(request.model)
        if "accepted" in request.output_schema.model_fields:
            output = request.output_schema.model_validate(
                {"accepted": True, "reasoning": "verification accepted"}
            )
            return ModelResult(
                output=output,
                usage=ModelUsage(prompt_tokens=4, completion_tokens=3, cost_usd=0.001),
            )
        output = request.output_schema.model_validate({"done": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=6, completion_tokens=3, cost_usd=0.001),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_acceptance",
        capabilities=frozenset(),
        budget_usd_limit=2.0,
    )


@pytest.mark.asyncio
async def test_autonomous_agent_draft_verify_and_acceptance_gates(tmp_path: Path) -> None:
    model_port = LoopAwareModelPort()
    kernel = ArtanaKernel(store=SQLiteStore(str(tmp_path / "state.db")), model_port=model_port)
    gate_calls = 0

    @kernel.tool()
    async def run_tests() -> str:
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == 1:
            return json.dumps({"passed": False, "status": "failed"})
        return json.dumps({"passed": True, "status": "passed"})

    agent = AutonomousAgent(
        kernel=kernel,
        loop=DraftVerifyLoopConfig(
            draft_model="gpt-5.3-codex-spark",
            verify_model="gpt-5.3-codex",
        ),
    )
    try:
        result = await agent.run(
            run_id="run_acceptance_loop",
            tenant=_tenant(),
            model="ignored_when_loop_enabled",
            prompt="Fix flaky tests and finish when done.",
            output_schema=AgentResult,
            max_iterations=4,
            acceptance=AcceptanceSpec(gates=(ToolGate(tool="run_tests", must_pass=True),)),
        )
        assert result.done is True
        assert gate_calls == 2
        assert model_port.models_seen == [
            "gpt-5.3-codex-spark",
            "gpt-5.3-codex-spark",
            "gpt-5.3-codex",
        ]
    finally:
        await kernel.close()
