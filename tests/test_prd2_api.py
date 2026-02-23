from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import ArtanaKernel, KernelModelClient, ModelInput
from artana.events import ChatMessage, PauseRequestedPayload, ResumeRequestedPayload
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class TransferArgs(BaseModel):
    account_id: str
    amount: str


class PauseContext(BaseModel):
    ticket: str


class HumanInput(BaseModel):
    note: str


class CountingModelPort:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.message_batches: list[list[str]] = []

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        self.prompts.append(request.prompt)
        self.message_batches.append(
            [f"{message.role}:{message.content}" for message in request.messages]
        )
        output = request.output_schema.model_validate(
            {"approved": True, "reason": f"call-{self.calls}"}
        )
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=3, completion_tokens=2, cost_usd=0.01),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_prd2",
        capabilities=frozenset({"finance:write"}),
        budget_usd_limit=1.0,
    )


@pytest.mark.asyncio
async def test_start_load_and_resume_boundary_persist_lifecycle_events(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=CountingModelPort())

    try:
        run = await kernel.start_run(tenant=_tenant())
        loaded = await kernel.load_run(run_id=run.run_id)
        resumed = await kernel.resume(
            run_id=run.run_id,
            tenant=_tenant(),
            human_input=HumanInput(note="approved"),
        )

        assert loaded == run
        assert resumed.run_id == run.run_id
        assert resumed.tenant_id == "org_prd2"

        events = await store.get_events_for_run(run.run_id)
        assert [event.event_type for event in events] == [
            "run_started",
            "resume_requested",
        ]
        payload = events[-1].payload
        assert isinstance(payload, ResumeRequestedPayload)
        assert payload.human_input_json == '{"note":"approved"}'
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_step_model_replays_by_step_key(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)

    try:
        run = await kernel.start_run(tenant=_tenant())
        first = await kernel.step_model(
            run_id=run.run_id,
            tenant=_tenant(),
            model="gpt-4o-mini",
            input=ModelInput.from_prompt("Should we approve?"),
            output_schema=Decision,
            step_key="decision",
        )
        second = await kernel.step_model(
            run_id=run.run_id,
            tenant=_tenant(),
            model="gpt-4o-mini",
            input=ModelInput.from_prompt("Should we approve?"),
            output_schema=Decision,
            step_key="decision",
        )
        third = await kernel.step_model(
            run_id=run.run_id,
            tenant=_tenant(),
            model="gpt-4o-mini",
            input=ModelInput.from_prompt("Should we approve?"),
            output_schema=Decision,
            step_key="decision_2",
        )

        assert first.replayed is False
        assert second.replayed is True
        assert second.seq == first.seq
        assert third.replayed is False
        assert third.seq > second.seq
        assert model_port.calls == 2
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_step_model_supports_messages_input(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)

    try:
        run = await kernel.start_run(tenant=_tenant())
        result = await kernel.step_model(
            run_id=run.run_id,
            tenant=_tenant(),
            model="gpt-4o-mini",
            input=ModelInput.from_messages(
                [
                    ChatMessage(role="system", content="You are a strict approver."),
                    ChatMessage(role="user", content="approve transfer?"),
                ]
            ),
            output_schema=Decision,
            step_key="msg_step",
        )

        assert result.replayed is False
        assert model_port.calls == 1
        assert model_port.prompts == ["approve transfer?"]
        assert model_port.message_batches == [
            [
                "system:You are a strict approver.",
                "user:approve transfer?",
            ]
        ]
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_step_tool_replays_and_pause_persists_context(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=CountingModelPort())
    tool_calls = 0

    @kernel.tool(requires_capability="finance:write")
    async def submit_transfer(account_id: str, amount: str) -> str:
        nonlocal tool_calls
        tool_calls += 1
        return f'{{"status":"submitted","account_id":"{account_id}","amount":"{amount}"}}'

    try:
        run = await kernel.start_run(tenant=_tenant())
        first = await kernel.step_tool(
            run_id=run.run_id,
            tenant=_tenant(),
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
            step_key="transfer_step",
        )
        second = await kernel.step_tool(
            run_id=run.run_id,
            tenant=_tenant(),
            tool_name="submit_transfer",
            arguments=TransferArgs(account_id="acc_1", amount="10"),
            step_key="transfer_step",
        )
        ticket = await kernel.pause(
            run_id=run.run_id,
            tenant=_tenant(),
            reason="manual review",
            context=PauseContext(ticket="mgr-1"),
            step_key="review_pause",
        )

        assert first.replayed is False
        assert second.replayed is True
        assert second.seq == first.seq
        assert tool_calls == 1
        assert ticket.reason == "manual review"

        events = await store.get_events_for_run(run.run_id)
        pause_payload = events[-1].payload
        assert isinstance(pause_payload, PauseRequestedPayload)
        assert pause_payload.context_json == '{"ticket":"mgr-1"}'
        assert pause_payload.step_key == "review_pause"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_chat_client_wraps_step_model(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    client = KernelModelClient(kernel=kernel)

    try:
        run = await kernel.start_run(tenant=_tenant())
        first = await client.step(
            run_id=run.run_id,
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="approve?",
            output_schema=Decision,
            step_key="chat_step",
        )
        second = await client.step(
            run_id=run.run_id,
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="approve?",
            output_schema=Decision,
            step_key="chat_step",
        )

        assert first.replayed is False
        assert second.replayed is True
        assert model_port.calls == 1
    finally:
        await kernel.close()
