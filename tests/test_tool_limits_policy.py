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
from artana.safety import SafetyPolicyConfig, ToolLimitPolicy, ToolSafetyPolicy
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class PaymentArgs(BaseModel):
    payment_id: str
    amount: float


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
        tenant_id="org_limits",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


@pytest.mark.asyncio
async def test_max_calls_per_run_limit_blocks_after_threshold(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "submit_payment": ToolSafetyPolicy(
                            limits=ToolLimitPolicy(max_calls_per_run=1)
                        )
                    }
                )
            )
        ],
    )

    @kernel.tool()
    async def submit_payment(payment_id: str, amount: float) -> str:
        return f'{{"ok":true,"payment_id":"{payment_id}","amount":{amount}}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_limit_per_run")
        await kernel.step_tool(
            run_id="run_limit_per_run",
            tenant=_tenant(),
            tool_name="submit_payment",
            arguments=PaymentArgs(payment_id="p1", amount=10.0),
            step_key="payment_1",
        )
        with pytest.raises(PolicyViolationError, match="max_calls_per_run"):
            await kernel.step_tool(
                run_id="run_limit_per_run",
                tenant=_tenant(),
                tool_name="submit_payment",
                arguments=PaymentArgs(payment_id="p2", amount=20.0),
                step_key="payment_2",
            )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_max_calls_per_tenant_window_blocks_across_runs(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "submit_payment": ToolSafetyPolicy(
                            limits=ToolLimitPolicy(
                                max_calls_per_tenant_window=1,
                                tenant_window_seconds=300,
                            )
                        )
                    }
                )
            )
        ],
    )

    @kernel.tool()
    async def submit_payment(payment_id: str, amount: float) -> str:
        return f'{{"ok":true,"payment_id":"{payment_id}","amount":{amount}}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_window_1")
        await kernel.step_tool(
            run_id="run_window_1",
            tenant=_tenant(),
            tool_name="submit_payment",
            arguments=PaymentArgs(payment_id="p1", amount=10.0),
            step_key="payment_1",
        )

        await kernel.start_run(tenant=_tenant(), run_id="run_window_2")
        with pytest.raises(PolicyViolationError, match="tenant window limit"):
            await kernel.step_tool(
                run_id="run_window_2",
                tenant=_tenant(),
                tool_name="submit_payment",
                arguments=PaymentArgs(payment_id="p2", amount=20.0),
                step_key="payment_2",
            )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_max_amount_per_call_blocks_excessive_amount(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "submit_payment": ToolSafetyPolicy(
                            limits=ToolLimitPolicy(
                                max_amount_usd_per_call=500.0,
                                amount_arg_path="amount",
                            )
                        )
                    }
                )
            )
        ],
    )

    @kernel.tool()
    async def submit_payment(payment_id: str, amount: float) -> str:
        return f'{{"ok":true,"payment_id":"{payment_id}","amount":{amount}}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_limit_amount")
        with pytest.raises(PolicyViolationError, match="exceeds max_amount_usd_per_call"):
            await kernel.step_tool(
                run_id="run_limit_amount",
                tenant=_tenant(),
                tool_name="submit_payment",
                arguments=PaymentArgs(payment_id="p_big", amount=700.0),
            )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_tenant_window_query_uses_utc_timestamp_semantics(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=[],
    )

    @kernel.tool()
    async def submit_payment(payment_id: str, amount: float) -> str:
        return f'{{"ok":true,"payment_id":"{payment_id}","amount":{amount}}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_utc_check")
        await kernel.step_tool(
            run_id="run_utc_check",
            tenant=_tenant(),
            tool_name="submit_payment",
            arguments=PaymentArgs(payment_id="p1", amount=1.0),
        )
        count_recent = await store.get_tool_request_count_for_tenant_since(
            tenant_id=_tenant().tenant_id,
            tool_name="submit_payment",
            since=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        assert count_recent == 1
    finally:
        await kernel.close()

