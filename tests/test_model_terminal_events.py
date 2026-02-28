from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import KernelModelClient
from artana.events import ChatMessage, EventType, ModelRequestedPayload, ModelTerminalPayload
from artana.kernel import ArtanaKernel
from artana.models import TenantContext
from artana.ports.model import (
    ModelRequest,
    ModelResult,
    ModelTimeoutError,
    ModelTransientError,
    ModelUsage,
)
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class SuccessModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest[OutputModelT]) -> ModelResult[OutputModelT]:
        self.calls += 1
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=2, completion_tokens=1, cost_usd=0.01),
        )


class TimeoutModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest[OutputModelT]) -> ModelResult[OutputModelT]:
        self.calls += 1
        raise ModelTimeoutError("simulated timeout")


class FlakyModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest[OutputModelT]) -> ModelResult[OutputModelT]:
        self.calls += 1
        if self.calls == 1:
            raise ModelTransientError("simulated transient error")
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=3, completion_tokens=2, cost_usd=0.02),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_model_terminal",
        capabilities=frozenset(),
        budget_usd_limit=10.0,
    )


@pytest.mark.asyncio
async def test_model_requested_has_single_terminal_event_on_success(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = SuccessModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    try:
        await KernelModelClient(kernel=kernel).step(
            run_id="run_terminal_success",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="decide",
            output_schema=Decision,
        )
        events = await store.get_events_for_run("run_terminal_success")
        requested = [event for event in events if event.event_type == EventType.MODEL_REQUESTED]
        terminal = [event for event in events if event.event_type == EventType.MODEL_TERMINAL]
        assert len(requested) == 1
        assert len(terminal) == 1
        payload = terminal[0].payload
        assert isinstance(payload, ModelTerminalPayload)
        assert payload.outcome == "completed"
        assert payload.source_model_requested_event_id == requested[0].event_id
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_model_failure_emits_terminal_and_marks_run_failed(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = TimeoutModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    try:
        with pytest.raises(ModelTimeoutError):
            await KernelModelClient(kernel=kernel).step(
                run_id="run_terminal_timeout",
                tenant=_tenant(),
                model="gpt-4o-mini",
                prompt="decide",
                output_schema=Decision,
            )
        events = await store.get_events_for_run("run_terminal_timeout")
        requested = [event for event in events if event.event_type == EventType.MODEL_REQUESTED]
        terminal = [event for event in events if event.event_type == EventType.MODEL_TERMINAL]
        assert len(requested) == 1
        assert len(terminal) == 1
        payload = terminal[0].payload
        assert isinstance(payload, ModelTerminalPayload)
        assert payload.outcome == "timeout"
        status = await kernel.get_run_status(run_id="run_terminal_timeout")
        assert status.status == "failed"
        assert status.last_event_type == EventType.MODEL_TERMINAL.value
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_failed_model_cycle_replays_failure_without_provider_recall(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = TimeoutModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    try:
        with pytest.raises(ModelTimeoutError):
            await KernelModelClient(kernel=kernel).step(
                run_id="run_terminal_replay_fail",
                tenant=_tenant(),
                model="gpt-4o-mini",
                prompt="decide",
                output_schema=Decision,
            )
        with pytest.raises(ModelTimeoutError):
            await KernelModelClient(kernel=kernel).step(
                run_id="run_terminal_replay_fail",
                tenant=_tenant(),
                model="gpt-4o-mini",
                prompt="decide",
                output_schema=Decision,
            )
        assert model_port.calls == 1
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_retry_failed_step_creates_new_model_cycle(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = FlakyModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)
    try:
        with pytest.raises(ModelTransientError):
            await KernelModelClient(kernel=kernel).step(
                run_id="run_terminal_retry",
                tenant=_tenant(),
                model="gpt-4o-mini",
                prompt="decide",
                output_schema=Decision,
            )
        result = await KernelModelClient(kernel=kernel).step(
            run_id="run_terminal_retry",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="decide",
            output_schema=Decision,
            retry_failed_step=True,
        )
        assert result.output.approved is True
        assert model_port.calls == 2

        events = await store.get_events_for_run("run_terminal_retry")
        requested = [event for event in events if event.event_type == EventType.MODEL_REQUESTED]
        terminal = [event for event in events if event.event_type == EventType.MODEL_TERMINAL]
        assert len(requested) == 2
        assert len(terminal) == 2
        requested_payload_1 = requested[0].payload
        requested_payload_2 = requested[1].payload
        assert isinstance(requested_payload_1, ModelRequestedPayload)
        assert isinstance(requested_payload_2, ModelRequestedPayload)
        assert requested_payload_1.model_cycle_id is not None
        assert requested_payload_2.model_cycle_id is not None
        assert requested_payload_1.model_cycle_id != requested_payload_2.model_cycle_id
        first_terminal = terminal[0].payload
        second_terminal = terminal[1].payload
        assert isinstance(first_terminal, ModelTerminalPayload)
        assert isinstance(second_terminal, ModelTerminalPayload)
        assert first_terminal.source_model_requested_event_id == requested[0].event_id
        assert second_terminal.source_model_requested_event_id == requested[1].event_id
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_cleanup_stale_model_runs_emits_abandoned_terminal(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=SuccessModelPort())
    tenant = _tenant()
    run_id = "run_stale_cleanup"
    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.MODEL_REQUESTED,
            payload=ModelRequestedPayload(
                model="gpt-4o-mini",
                prompt="stuck",
                messages=[ChatMessage(role="user", content="stuck")],
                model_cycle_id="cycle_stale_1",
            ),
        )
        now_value = datetime.now(timezone.utc) + timedelta(seconds=601)
        closed = await kernel.cleanup_stale_model_runs(
            tenant_id=tenant.tenant_id,
            model_timeout_seconds=300.0,
            now=now_value,
        )
        assert closed == (run_id,)

        events = await store.get_events_for_run(run_id)
        terminal_event = events[-1]
        assert terminal_event.event_type == EventType.MODEL_TERMINAL
        payload = terminal_event.payload
        assert isinstance(payload, ModelTerminalPayload)
        assert payload.outcome == "abandoned"
        assert payload.failure_reason == "model_timeout_stale_cleanup"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_cleanup_stale_model_runs_skips_active_leased_runs(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=SuccessModelPort())
    tenant = _tenant()
    run_id = "run_stale_with_lease"
    try:
        await kernel.start_run(tenant=tenant, run_id=run_id)
        await store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.MODEL_REQUESTED,
            payload=ModelRequestedPayload(
                model="gpt-4o-mini",
                prompt="stuck",
                messages=[ChatMessage(role="user", content="stuck")],
                model_cycle_id="cycle_stale_2",
            ),
        )
        acquired = await kernel.acquire_run_lease(
            run_id=run_id,
            worker_id="worker_1",
            ttl_seconds=3600,
        )
        assert acquired is True
        now_value = datetime.now(timezone.utc) + timedelta(seconds=301)
        closed = await kernel.cleanup_stale_model_runs(
            tenant_id=tenant.tenant_id,
            model_timeout_seconds=1.0,
            now=now_value,
        )
        assert closed == ()

        events = await store.get_events_for_run(run_id)
        assert events[-1].event_type == EventType.MODEL_REQUESTED
    finally:
        await kernel.close()
