from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.kernel import ArtanaKernel, PolicyViolationError
from artana.middleware import SafetyPolicyMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.safety import InvariantRule, SafetyPolicyConfig, ToolSafetyPolicy
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class RequiredArgs(BaseModel):
    verified: bool


class EmailArgs(BaseModel):
    email: str


class RecipientArgs(BaseModel):
    recipient: str


class AmountArgs(BaseModel):
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
        tenant_id="org_invariants",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


@pytest.mark.asyncio
async def test_required_arg_true_invariant_blocks_false_and_allows_true(
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
                        "perform_verified_action": ToolSafetyPolicy(
                            invariants=(
                                InvariantRule(type="required_arg_true", field="verified"),
                            )
                        )
                    }
                )
            )
        ],
    )

    @kernel.tool()
    async def perform_verified_action(verified: bool) -> str:
        return f'{{"ok":true,"verified":{str(verified).lower()}}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_required_arg")
        with pytest.raises(PolicyViolationError, match="required_arg_true"):
            await kernel.step_tool(
                run_id="run_required_arg",
                tenant=_tenant(),
                tool_name="perform_verified_action",
                arguments=RequiredArgs(verified=False),
                step_key="verify_false",
            )
        allowed = await kernel.step_tool(
            run_id="run_required_arg",
            tenant=_tenant(),
            tool_name="perform_verified_action",
            arguments=RequiredArgs(verified=True),
            step_key="verify_true",
        )
        assert allowed.replayed is False
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_email_domain_allowlist_invariant(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "email_customer": ToolSafetyPolicy(
                            invariants=(
                                InvariantRule(
                                    type="email_domain_allowlist",
                                    field="email",
                                    allowed_domains=("trusted.com",),
                                ),
                            )
                        )
                    }
                )
            )
        ],
    )

    @kernel.tool()
    async def email_customer(email: str) -> str:
        return f'{{"ok":true,"email":"{email}"}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_email_domain")
        with pytest.raises(PolicyViolationError, match="allowlist"):
            await kernel.step_tool(
                run_id="run_email_domain",
                tenant=_tenant(),
                tool_name="email_customer",
                arguments=EmailArgs(email="user@evil.com"),
            )
        allowed = await kernel.step_tool(
            run_id="run_email_domain",
            tenant=_tenant(),
            tool_name="email_customer",
            arguments=EmailArgs(email="user@trusted.com"),
            step_key="trusted",
        )
        assert allowed.replayed is False
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_recipient_must_be_verified_invariant(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "transfer_funds": ToolSafetyPolicy(
                            invariants=(
                                InvariantRule(
                                    type="recipient_must_be_verified",
                                    recipient_arg_path="recipient",
                                ),
                            )
                        )
                    }
                )
            )
        ],
    )

    @kernel.tool()
    async def transfer_funds(recipient: str) -> str:
        return f'{{"ok":true,"recipient":"{recipient}"}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_recipient_verify")
        with pytest.raises(PolicyViolationError, match="not marked as verified"):
            await kernel.step_tool(
                run_id="run_recipient_verify",
                tenant=_tenant(),
                tool_name="transfer_funds",
                arguments=RecipientArgs(recipient="acct_new"),
            )

        await kernel.append_run_summary(
            run_id="run_recipient_verify",
            tenant=_tenant(),
            summary_type="policy::recipient_verification",
            summary_json='{"recipient":"acct_new","verified":true}',
            step_key="recipient_verified",
        )
        allowed = await kernel.step_tool(
            run_id="run_recipient_verify",
            tenant=_tenant(),
            tool_name="transfer_funds",
            arguments=RecipientArgs(recipient="acct_new"),
            step_key="transfer_verified",
        )
        assert allowed.replayed is False
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_custom_json_rule_invariant(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=[
            SafetyPolicyMiddleware(
                config=SafetyPolicyConfig(
                    tools={
                        "pay_invoice": ToolSafetyPolicy(
                            invariants=(
                                InvariantRule(
                                    type="custom_json_rule",
                                    json_path="amount",
                                    operator="lte",
                                    value_json="500",
                                ),
                            )
                        )
                    }
                )
            )
        ],
    )

    @kernel.tool()
    async def pay_invoice(amount: float) -> str:
        return f'{{"ok":true,"amount":{amount}}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_custom_rule")
        with pytest.raises(PolicyViolationError, match="custom_json_rule failed"):
            await kernel.step_tool(
                run_id="run_custom_rule",
                tenant=_tenant(),
                tool_name="pay_invoice",
                arguments=AmountArgs(amount=800.0),
                step_key="high_amount",
            )
        allowed = await kernel.step_tool(
            run_id="run_custom_rule",
            tenant=_tenant(),
            tool_name="pay_invoice",
            arguments=AmountArgs(amount=400.0),
            step_key="valid_amount",
        )
        assert allowed.replayed is False
    finally:
        await kernel.close()

