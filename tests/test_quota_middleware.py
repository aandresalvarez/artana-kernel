from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import KernelModelClient
from artana.events import EventPayload, EventType, KernelEvent, RunSummaryPayload
from artana.kernel import ArtanaKernel
from artana.middleware import BudgetExceededError, QuotaMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class CostModelPort:
    def __init__(self, *, cost_usd: float) -> None:
        self._cost_usd = cost_usd
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=10, completion_tokens=5, cost_usd=self._cost_usd),
        )


class AggregateOnlyStore:
    def __init__(self, spent_usd: float) -> None:
        self._spent_usd = spent_usd
        self.aggregate_calls = 0
        self.events_calls = 0

    async def append_event(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
        parent_step_key: str | None = None,
    ) -> KernelEvent:
        raise NotImplementedError

    async def get_events_for_run(self, run_id: str) -> list[KernelEvent]:
        self.events_calls += 1
        raise AssertionError(
            "QuotaMiddleware should not scan full event history when aggregate API exists."
        )

    async def verify_run_chain(self, run_id: str) -> bool:
        return True

    async def get_latest_run_summary(
        self,
        run_id: str,
        summary_type: str,
    ) -> RunSummaryPayload | None:
        return None

    async def close(self) -> None:
        return None

    async def get_model_cost_sum_for_run(self, run_id: str) -> float:
        self.aggregate_calls += 1
        return self._spent_usd


@pytest.mark.asyncio
async def test_quota_persists_from_event_log_across_kernel_restarts(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    tenant = TenantContext(
        tenant_id="org_budget",
        capabilities=frozenset(),
        budget_usd_limit=0.10,
    )

    first_model = CostModelPort(cost_usd=0.06)
    first_kernel = ArtanaKernel(
        store=store,
        model_port=first_model,
        middleware=[QuotaMiddleware()],
    )

    second_model = CostModelPort(cost_usd=0.06)
    second_kernel = ArtanaKernel(
        store=store,
        model_port=second_model,
        middleware=[QuotaMiddleware()],
    )

    try:
        await KernelModelClient(kernel=first_kernel).step(
            run_id="run_budget",
            prompt="first decision",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )

        with pytest.raises(BudgetExceededError):
            await KernelModelClient(kernel=second_kernel).step(
                run_id="run_budget",
                prompt="second decision",
                model="gpt-4o-mini",
                tenant=tenant,
                output_schema=Decision,
            )

        assert first_model.calls == 1
        assert second_model.calls == 1

        events = await store.get_events_for_run("run_budget")
        assert [event.event_type for event in events] == [
            "run_started",
            "model_requested",
            "model_terminal",
            "run_summary",
            "model_requested",
            "model_terminal",
        ]
    finally:
        await first_kernel.close()
        await second_kernel.close()


@pytest.mark.asyncio
async def test_quota_blocks_before_model_when_budget_already_exhausted(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    tenant = TenantContext(
        tenant_id="org_budget_2",
        capabilities=frozenset(),
        budget_usd_limit=0.10,
    )

    expensive_model = CostModelPort(cost_usd=0.11)
    first_kernel = ArtanaKernel(
        store=store,
        model_port=expensive_model,
        middleware=[QuotaMiddleware()],
    )

    try:
        with pytest.raises(BudgetExceededError):
            await KernelModelClient(kernel=first_kernel).step(
                run_id="run_budget_2",
                prompt="expensive call",
                model="gpt-4o-mini",
                tenant=tenant,
                output_schema=Decision,
            )
    finally:
        await first_kernel.close()

    retry_model = CostModelPort(cost_usd=0.01)
    second_kernel = ArtanaKernel(
        store=store,
        model_port=retry_model,
        middleware=[QuotaMiddleware()],
    )
    try:
        with pytest.raises(BudgetExceededError):
            await KernelModelClient(kernel=second_kernel).step(
                run_id="run_budget_2",
                prompt="retry call",
                model="gpt-4o-mini",
                tenant=tenant,
                output_schema=Decision,
            )
        assert retry_model.calls == 0
    finally:
        await second_kernel.close()


@pytest.mark.asyncio
async def test_quota_uses_aggregate_store_cost_api_when_available() -> None:
    store = AggregateOnlyStore(spent_usd=0.09)
    quota = QuotaMiddleware(store=store)
    allowed_tenant = TenantContext(
        tenant_id="org_agg_ok",
        capabilities=frozenset(),
        budget_usd_limit=0.10,
    )
    blocked_tenant = TenantContext(
        tenant_id="org_agg_block",
        capabilities=frozenset(),
        budget_usd_limit=0.09,
    )

    await quota.before_model(run_id="run_aggregate", tenant=allowed_tenant)
    assert store.aggregate_calls == 1
    assert store.events_calls == 0

    with pytest.raises(BudgetExceededError):
        await quota.before_model(run_id="run_aggregate", tenant=blocked_tenant)
    assert store.aggregate_calls == 2
    assert store.events_calls == 0
