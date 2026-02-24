from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.kernel import ArtanaKernel, PolicyViolationError, ToolExecutionFailedError
from artana.middleware import SafetyPolicyMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.safety import (
    SafetyPolicyConfig,
    SemanticIdempotencyRequirement,
    ToolSafetyPolicy,
)
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class InvoiceArgs(BaseModel):
    billing_period: str


class ChargeArgs(BaseModel):
    account_id: str


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
        tenant_id="org_semantic",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


@pytest.mark.asyncio
async def test_semantic_idempotency_blocks_duplicate_success_across_runs(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    safety = SafetyPolicyMiddleware(
        config=SafetyPolicyConfig(
            tools={
                "send_invoice": ToolSafetyPolicy(
                    semantic_idempotency=SemanticIdempotencyRequirement(
                        template="send_invoice:{tenant_id}:{billing_period}",
                        required_fields=("billing_period",),
                    )
                )
            }
        )
    )
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=[safety],
    )

    @kernel.tool()
    async def send_invoice(billing_period: str) -> str:
        return f'{{"ok":true,"period":"{billing_period}"}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_semantic_1")
        await kernel.step_tool(
            run_id="run_semantic_1",
            tenant=_tenant(),
            tool_name="send_invoice",
            arguments=InvoiceArgs(billing_period="2026-02"),
        )

        await kernel.start_run(tenant=_tenant(), run_id="run_semantic_2")
        with pytest.raises(PolicyViolationError, match="already executed"):
            await kernel.step_tool(
                run_id="run_semantic_2",
                tenant=_tenant(),
                tool_name="send_invoice",
                arguments=InvoiceArgs(billing_period="2026-02"),
            )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_semantic_idempotency_missing_required_field_blocks_request(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    safety = SafetyPolicyMiddleware(
        config=SafetyPolicyConfig(
            tools={
                "charge_card": ToolSafetyPolicy(
                    semantic_idempotency=SemanticIdempotencyRequirement(
                        template="charge:{tenant_id}:{billing_period}",
                        required_fields=("billing_period",),
                    )
                )
            }
        )
    )
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=[safety],
    )

    @kernel.tool()
    async def charge_card(account_id: str) -> str:
        return f'{{"ok":true,"account_id":"{account_id}"}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_missing_field")
        with pytest.raises(PolicyViolationError, match="is missing"):
            await kernel.step_tool(
                run_id="run_missing_field",
                tenant=_tenant(),
                tool_name="charge_card",
                arguments=ChargeArgs(account_id="acc_1"),
            )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_semantic_idempotency_blocks_when_previous_outcome_is_unknown(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"

    first_store = SQLiteStore(str(database_path))
    first_kernel = ArtanaKernel(
        store=first_store,
        model_port=PlainModelPort(),
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "send_invoice": ToolSafetyPolicy(
                            semantic_idempotency=SemanticIdempotencyRequirement(
                                template="send_invoice:{tenant_id}:{billing_period}",
                                required_fields=("billing_period",),
                            )
                        )
                    }
                )
            )
        ],
    )

    @first_kernel.tool()
    async def send_invoice(billing_period: str) -> str:
        raise RuntimeError("network dropped")

    await first_kernel.start_run(tenant=_tenant(), run_id="run_unknown_semantic_1")
    try:
        with pytest.raises(ToolExecutionFailedError, match="unknown outcome"):
            await first_kernel.step_tool(
                run_id="run_unknown_semantic_1",
                tenant=_tenant(),
                tool_name="send_invoice",
                arguments=InvoiceArgs(billing_period="2026-03"),
            )
    finally:
        await first_kernel.close()

    second_store = SQLiteStore(str(database_path))
    second_kernel = ArtanaKernel(
        store=second_store,
        model_port=PlainModelPort(),
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "send_invoice": ToolSafetyPolicy(
                            semantic_idempotency=SemanticIdempotencyRequirement(
                                template="send_invoice:{tenant_id}:{billing_period}",
                                required_fields=("billing_period",),
                            )
                        )
                    }
                )
            )
        ],
    )

    async def send_invoice_second(billing_period: str) -> str:
        return f'{{"ok":true,"period":"{billing_period}"}}'
    send_invoice_second.__name__ = "send_invoice"
    second_kernel.tool()(send_invoice_second)

    try:
        await second_kernel.start_run(tenant=_tenant(), run_id="run_unknown_semantic_2")
        with pytest.raises(PolicyViolationError, match="must be reconciled"):
            await second_kernel.step_tool(
                run_id="run_unknown_semantic_2",
                tenant=_tenant(),
                tool_name="send_invoice",
                arguments=InvoiceArgs(billing_period="2026-03"),
            )
    finally:
        await second_kernel.close()
