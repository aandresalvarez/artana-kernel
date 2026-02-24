from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana.events import EventType, HarnessSleepPayload
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    ok: bool


class PlainModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        output = request.output_schema.model_validate({"ok": True})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.001),
        )


def _tenant(*, capabilities: frozenset[str] = frozenset()) -> TenantContext:
    return TenantContext(
        tenant_id="org_kernel_os_api",
        capabilities=capabilities,
        budget_usd_limit=5.0,
    )


@pytest.mark.asyncio
async def test_kernel_run_status_resume_point_and_blocking_api(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=PlainModelPort())
    tenant = _tenant()

    try:
        await kernel.start_run(tenant=tenant, run_id="run_status")
        status = await kernel.get_run_status(run_id="run_status")
        assert status.status == "active"
        assert status.blocked_on is None

        await kernel.block_run(
            run_id="run_status",
            tenant=tenant,
            reason="Need manager approval",
            unblock_key="mgr_approval",
            metadata={"source": "qa"},
            step_key="block_1",
        )
        paused = await kernel.get_run_status(run_id="run_status")
        assert paused.status == "paused"
        assert paused.blocked_on == "unblock:mgr_approval"

        resume_point = await kernel.resume_point(run_id="run_status")
        assert resume_point.run_id == "run_status"
        assert resume_point.blocked_on == "unblock:mgr_approval"
        assert resume_point.last_event_seq >= 2

        await kernel.unblock_run(
            run_id="run_status",
            tenant=tenant,
            unblock_key="mgr_approval",
            metadata={"approved": True},
        )
        unblocked = await kernel.get_run_status(run_id="run_status")
        assert unblocked.status == "active"
        assert unblocked.blocked_on is None
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_checkpoint_and_artifact_syscalls(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=PlainModelPort())
    tenant = _tenant()
    try:
        await kernel.start_run(tenant=tenant, run_id="run_artifacts")

        await kernel.checkpoint(
            run_id="run_artifacts",
            tenant=tenant,
            name="phase_1",
            payload={"stage": "collect", "round": 1},
            step_key="checkpoint_1",
        )
        checkpoint = await kernel.get_latest_run_summary(
            run_id="run_artifacts",
            summary_type="checkpoint::phase_1",
        )
        assert checkpoint is not None
        assert checkpoint.step_key == "checkpoint_1"

        await kernel.set_artifact(
            run_id="run_artifacts",
            tenant=tenant,
            key="report",
            value={"version": 1},
        )
        await kernel.set_artifact(
            run_id="run_artifacts",
            tenant=tenant,
            key="report",
            value={"version": 2},
            schema_version="v2",
        )
        await kernel.set_artifact(
            run_id="run_artifacts",
            tenant=tenant,
            key="cursor",
            value={"offset": 10},
        )

        artifact = await kernel.get_artifact(run_id="run_artifacts", key="report")
        assert artifact == {"version": 2}

        artifacts = await kernel.list_artifacts(run_id="run_artifacts")
        assert artifacts["report"] == {"version": 2}
        assert artifacts["cursor"] == {"offset": 10}
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_explain_tool_allowlist_and_gateway_metadata(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(
        store=store,
        model_port=PlainModelPort(),
        middleware=ArtanaKernel.default_middleware_stack(),
    )

    @kernel.tool(
        requires_capability="finance:write",
        tool_version="2.1.0",
        schema_version="4",
        risk_level="high",
        sandbox_profile="payments",
    )
    async def transfer_funds(account_id: str, amount: float) -> str:
        return f'{{"ok":true,"account_id":"{account_id}","amount":{amount}}}'

    @kernel.tool()
    async def read_balance(account_id: str) -> str:
        return f'{{"ok":true,"account_id":"{account_id}"}}'

    try:
        decision = await kernel.explain_tool_allowlist(
            tenant=_tenant(capabilities=frozenset()),
            model="gpt-4o-mini",
            visible_tool_names={"transfer_funds", "read_balance"},
        )
        raw_decisions = decision.get("decisions")
        assert isinstance(raw_decisions, list)
        decisions = {
            str(item["tool_name"]): item
            for item in raw_decisions
            if isinstance(item, dict) and "tool_name" in item
        }
        assert decisions["read_balance"]["decision"] == "allowed"
        assert decisions["transfer_funds"]["decision"] == "filtered"
        assert decisions["transfer_funds"]["reason"] == "filtered_missing_capability"

        canonical_args, schema_hash = kernel.canonicalize_tool_args(
            tool_name="transfer_funds",
            arguments={"amount": 10.0, "account_id": "acc_1"},
        )
        assert canonical_args == '{"account_id":"acc_1","amount":10.0}'
        assert isinstance(schema_hash, str) and schema_hash != ""

        fingerprint = kernel.tool_fingerprint(tool_name="transfer_funds")
        assert fingerprint.tool_name == "transfer_funds"
        assert fingerprint.tool_version == "2.1.0"
        assert fingerprint.schema_version == "4"
        assert fingerprint.schema_hash == schema_hash
        assert fingerprint.risk_level == "high"
        assert fingerprint.sandbox_profile == "payments"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_stream_events_and_run_lease_syscalls(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=PlainModelPort())
    tenant = _tenant()
    try:
        await kernel.start_run(tenant=tenant, run_id="run_stream_lease")
        await kernel.append_harness_event(
            run_id="run_stream_lease",
            tenant=tenant,
            event_type=EventType.HARNESS_SLEEP,
            payload=HarnessSleepPayload(status="completed"),
        )

        streamed = [
            event
            async for event in kernel.stream_events(
                run_id="run_stream_lease",
                since_seq=0,
            )
        ]
        assert [event.seq for event in streamed] == [1, 2]

        streamed_since_one = [
            event
            async for event in kernel.stream_events(
                run_id="run_stream_lease",
                since_seq=1,
            )
        ]
        assert [event.seq for event in streamed_since_one] == [2]

        assert (
            await kernel.acquire_run_lease(
                run_id="run_stream_lease",
                worker_id="worker_a",
                ttl_seconds=30,
            )
            is True
        )
        assert (
            await kernel.acquire_run_lease(
                run_id="run_stream_lease",
                worker_id="worker_b",
                ttl_seconds=30,
            )
            is False
        )
        lease = await kernel.get_run_lease(run_id="run_stream_lease")
        assert lease is not None
        assert lease.worker_id == "worker_a"
        assert (
            await kernel.renew_run_lease(
                run_id="run_stream_lease",
                worker_id="worker_a",
                ttl_seconds=30,
            )
            is True
        )
        assert (
            await kernel.release_run_lease(
                run_id="run_stream_lease",
                worker_id="worker_b",
            )
            is False
        )
        assert (
            await kernel.release_run_lease(
                run_id="run_stream_lease",
                worker_id="worker_a",
            )
            is True
        )
        assert (
            await kernel.acquire_run_lease(
                run_id="run_stream_lease",
                worker_id="worker_b",
                ttl_seconds=30,
            )
            is True
        )
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_list_active_runs_filters_terminal_runs(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=PlainModelPort())
    tenant = _tenant()
    try:
        await kernel.start_run(tenant=tenant, run_id="run_active")

        await kernel.start_run(tenant=tenant, run_id="run_paused")
        await kernel.block_run(
            run_id="run_paused",
            tenant=tenant,
            reason="waiting on external webhook",
            unblock_key="webhook_1",
        )

        await kernel.start_run(tenant=tenant, run_id="run_completed")
        await kernel.append_harness_event(
            run_id="run_completed",
            tenant=tenant,
            event_type=EventType.HARNESS_SLEEP,
            payload=HarnessSleepPayload(status="completed"),
        )

        active_runs = await kernel.list_active_runs(tenant_id=tenant.tenant_id)
        active_ids = {run.run_id for run in active_runs}
        assert "run_active" in active_ids
        assert "run_paused" in active_ids
        assert "run_completed" not in active_ids

        paused_runs = await kernel.list_active_runs(
            tenant_id=tenant.tenant_id,
            status="paused",
        )
        assert {run.run_id for run in paused_runs} == {"run_paused"}
    finally:
        await kernel.close()
