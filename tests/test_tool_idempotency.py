from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana._kernel.tool_execution import derive_idempotency_key
from artana.events import ToolRequestedPayload
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class TransferArgs(BaseModel):
    account_id: str
    amount: str


class PlainModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.001),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_idemp",
        capabilities=frozenset({"finance:write"}),
        budget_usd_limit=1.0,
    )


def test_derive_idempotency_key_is_stable_for_same_inputs() -> None:
    key_one = derive_idempotency_key(
        run_id="run_1",
        tool_name="submit_transfer",
        arguments_json='{"account_id":"acc_1","amount":"10"}',
        step_key="transfer",
    )
    key_two = derive_idempotency_key(
        run_id="run_1",
        tool_name="submit_transfer",
        arguments_json='{"account_id":"acc_1","amount":"10"}',
        step_key="transfer",
    )
    assert key_one == key_two


def test_derive_idempotency_key_changes_for_different_arguments() -> None:
    key_one = derive_idempotency_key(
        run_id="run_1",
        tool_name="submit_transfer",
        arguments_json='{"account_id":"acc_1","amount":"10"}',
        step_key="transfer",
    )
    key_two = derive_idempotency_key(
        run_id="run_1",
        tool_name="submit_transfer",
        arguments_json='{"account_id":"acc_1","amount":"20"}',
        step_key="transfer",
    )
    assert key_one != key_two


@pytest.mark.asyncio
async def test_step_tool_uses_deterministic_idempotency_key(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=PlainModelPort())

    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        return f'{{"account_id":"{account_id}","amount":"{amount}","status":"submitted"}}'

    try:
        await kernel.start_run(tenant=_tenant(), run_id="run_idemp")
        await kernel.step_tool(
            run_id="run_idemp",
            tenant=_tenant(),
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
            step_key="transfer",
        )

        events = await store.get_events_for_run("run_idemp")
        requested_payload = events[1].payload
        assert isinstance(requested_payload, ToolRequestedPayload)
        assert requested_payload.idempotency_key == derive_idempotency_key(
            run_id="run_idemp",
            tool_name="submit_transfer",
            arguments_json='{"account_id":"acc_1","amount":"10"}',
            step_key="transfer",
        )
    finally:
        await kernel.close()
