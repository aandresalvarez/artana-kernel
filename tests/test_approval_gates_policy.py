from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.events import EventType, PauseRequestedPayload
from artana.kernel import (
    ApprovalRequiredError,
    ArtanaKernel,
    PolicyViolationError,
)
from artana.middleware import SafetyPolicyMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.safety import ApprovalGatePolicy, SafetyPolicyConfig, ToolSafetyPolicy
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class TransferArgs(BaseModel):
    account_id: str
    amount: str


class CriticModelPort:
    def __init__(self, *, approve: bool) -> None:
        self.approve = approve
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        if "approval_key=" in request.prompt:
            output = request.output_schema.model_validate(
                {
                    "approved": self.approve,
                    "reason": "critic_ok" if self.approve else "critic_denied",
                }
            )
        else:
            output = request.output_schema.model_validate({"ok": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=2, completion_tokens=1, cost_usd=0.001),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_approval",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


@pytest.mark.asyncio
async def test_human_approval_gate_pauses_once_and_allows_after_approval(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=CriticModelPort(approve=True),
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "submit_transfer": ToolSafetyPolicy(
                            approval=ApprovalGatePolicy(mode="human")
                        )
                    }
                )
            )
        ],
    )
    tool_calls = 0

    @kernel.tool()
    async def submit_transfer(account_id: str, amount: str) -> str:
        nonlocal tool_calls
        tool_calls += 1
        return f'{{"ok":true,"account_id":"{account_id}","amount":"{amount}"}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_human_gate")
        with pytest.raises(ApprovalRequiredError) as first_exc:
            await kernel.step_tool(
                run_id="run_human_gate",
                tenant=_tenant(),
                tool_name="submit_transfer",
                arguments=TransferArgs(account_id="acc_1", amount="10"),
            )
        with pytest.raises(ApprovalRequiredError):
            await kernel.step_tool(
                run_id="run_human_gate",
                tenant=_tenant(),
                tool_name="submit_transfer",
                arguments=TransferArgs(account_id="acc_1", amount="10"),
            )

        events = await store.get_events_for_run("run_human_gate")
        pauses = [
            event
            for event in events
            if event.event_type == EventType.PAUSE_REQUESTED
            and isinstance(event.payload, PauseRequestedPayload)
        ]
        assert len(pauses) == 1

        await kernel.approve_tool_call(
            run_id="run_human_gate",
            tenant=_tenant(),
            approval_key=first_exc.value.approval_key,
            mode="human",
            reason="manager approved",
        )
        result = await kernel.step_tool(
            run_id="run_human_gate",
            tenant=_tenant(),
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
        )
        assert result.replayed is False
        assert tool_calls == 1
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_critic_approval_gate_approves_and_executes_tool(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CriticModelPort(approve=True)
    kernel = ArtanaKernel(
        store=store,
        model_port=model_port,
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "submit_transfer": ToolSafetyPolicy(
                            approval=ApprovalGatePolicy(
                                mode="critic",
                                critic_model="gpt-4o-mini",
                            )
                        )
                    }
                )
            )
        ],
    )
    tool_calls = 0

    @kernel.tool()
    async def submit_transfer(account_id: str, amount: str) -> str:
        nonlocal tool_calls
        tool_calls += 1
        return f'{{"ok":true,"account_id":"{account_id}","amount":"{amount}"}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_critic_allow")
        result = await kernel.step_tool(
            run_id="run_critic_allow",
            tenant=_tenant(),
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
        )
        assert result.replayed is False
        assert tool_calls == 1
        assert model_port.calls == 1

        replayed = await kernel.step_tool(
            run_id="run_critic_allow",
            tenant=_tenant(),
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
        )
        assert replayed.replayed is True
        assert model_port.calls == 1
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_critic_approval_gate_denial_blocks_tool_execution(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CriticModelPort(approve=False)
    kernel = ArtanaKernel(
        store=store,
        model_port=model_port,
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "submit_transfer": ToolSafetyPolicy(
                            approval=ApprovalGatePolicy(
                                mode="critic",
                                critic_model="gpt-4o-mini",
                            )
                        )
                    }
                )
            )
        ],
    )
    tool_calls = 0

    @kernel.tool()
    async def submit_transfer(account_id: str, amount: str) -> str:
        nonlocal tool_calls
        tool_calls += 1
        return f'{{"ok":true,"account_id":"{account_id}","amount":"{amount}"}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_critic_deny")
        with pytest.raises(PolicyViolationError, match="Critic gate denied"):
            await kernel.step_tool(
                run_id="run_critic_deny",
                tenant=_tenant(),
                tool_name="submit_transfer",
                arguments=TransferArgs(account_id="acc_1", amount="10"),
            )
        assert tool_calls == 0
        assert model_port.calls == 1
    finally:
        await kernel.close()

