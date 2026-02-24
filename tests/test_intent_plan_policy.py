from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.kernel import ArtanaKernel, PolicyViolationError
from artana.middleware import SafetyPolicyMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.safety import IntentPlanRecord, IntentRequirement, SafetyPolicyConfig, ToolSafetyPolicy
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
            usage=ModelUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.001),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_intent",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


@pytest.mark.asyncio
async def test_tool_with_required_intent_blocks_without_recorded_intent(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "transfer_funds": ToolSafetyPolicy(
                            intent=IntentRequirement(require_intent=True)
                        )
                    }
                )
            )
        ],
    )

    @kernel.tool()
    async def transfer_funds(account_id: str, amount: str) -> str:
        return f'{{"ok":true,"account_id":"{account_id}","amount":"{amount}"}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_intent_required")
        with pytest.raises(PolicyViolationError, match="requires an active intent plan"):
            await kernel.step_tool(
                run_id="run_intent_required",
                tenant=_tenant(),
                tool_name="transfer_funds",
                arguments=TransferArgs(account_id="acc_1", amount="10"),
            )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_recorded_intent_allows_tool_execution(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "transfer_funds": ToolSafetyPolicy(
                            intent=IntentRequirement(require_intent=True)
                        )
                    }
                )
            )
        ],
    )

    @kernel.tool()
    async def transfer_funds(account_id: str, amount: str) -> str:
        return f'{{"ok":true,"account_id":"{account_id}","amount":"{amount}"}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_intent_ok")
        await kernel.record_intent_plan(
            run_id="run_intent_ok",
            tenant=_tenant(),
            intent=IntentPlanRecord(
                intent_id="intent_1",
                goal="Send approved transfer",
                why="Customer requested payout",
                success_criteria="Transfer submitted once",
                assumed_state="Recipient is verified",
                applies_to_tools=("transfer_funds",),
            ),
        )
        result = await kernel.step_tool(
            run_id="run_intent_ok",
            tenant=_tenant(),
            tool_name="transfer_funds",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
        )
        assert result.replayed is False
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_stale_intent_is_rejected_when_max_age_configured(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "transfer_funds": ToolSafetyPolicy(
                            intent=IntentRequirement(require_intent=True, max_age_seconds=5)
                        )
                    }
                )
            )
        ],
    )

    @kernel.tool()
    async def transfer_funds(account_id: str, amount: str) -> str:
        return f'{{"ok":true,"account_id":"{account_id}","amount":"{amount}"}}'

    stale_created_at = datetime.now(timezone.utc) - timedelta(seconds=20)
    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_intent_stale")
        await kernel.record_intent_plan(
            run_id="run_intent_stale",
            tenant=_tenant(),
            intent=IntentPlanRecord(
                intent_id="intent_stale",
                goal="Old intent",
                why="Outdated context",
                success_criteria="N/A",
                assumed_state="N/A",
                applies_to_tools=("transfer_funds",),
                created_at=stale_created_at,
            ),
        )
        with pytest.raises(PolicyViolationError, match="requires an active intent plan"):
            await kernel.step_tool(
                run_id="run_intent_stale",
                tenant=_tenant(),
                tool_name="transfer_funds",
                arguments=TransferArgs(account_id="acc_1", amount="10"),
            )
    finally:
        await kernel.close()

