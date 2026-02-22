from __future__ import annotations

from artana.events import EventType
from artana.middleware.base import BudgetExceededError, ModelInvocation
from artana.models import TenantContext
from artana.ports.model import ModelUsage
from artana.store.base import EventStore


class QuotaMiddleware:
    def __init__(self, store: EventStore | None = None) -> None:
        self._store = store
        self._spent_usd_by_run: dict[str, float] = {}

    def bind_store(self, store: EventStore) -> None:
        self._store = store

    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        return invocation

    async def before_model(self, *, run_id: str, tenant: TenantContext) -> None:
        spent = await self._load_spent_for_run(run_id=run_id)
        if spent >= tenant.budget_usd_limit:
            raise BudgetExceededError(
                "Run "
                f"{run_id!r} budget exhausted. "
                f"limit={tenant.budget_usd_limit:.6f}, spent={spent:.6f}"
            )

    async def after_model(
        self, *, run_id: str, tenant: TenantContext, usage: ModelUsage
    ) -> None:
        if self._store is None:
            spent_before = self._spent_usd_by_run.get(run_id, 0.0)
            spent_after = spent_before + usage.cost_usd
            self._spent_usd_by_run[run_id] = spent_after
        else:
            spent_after = await self._load_spent_from_store(run_id=run_id)
        if spent_after > tenant.budget_usd_limit:
            raise BudgetExceededError(
                "Run "
                f"{run_id!r} exceeded budget. "
                f"limit={tenant.budget_usd_limit:.6f}, spent={spent_after:.6f}"
            )

    async def _load_spent_for_run(self, *, run_id: str) -> float:
        if self._store is None:
            return self._spent_usd_by_run.get(run_id, 0.0)
        return await self._load_spent_from_store(run_id=run_id)

    async def _load_spent_from_store(self, *, run_id: str) -> float:
        if self._store is None:
            raise RuntimeError("QuotaMiddleware store is not configured.")

        events = await self._store.get_events_for_run(run_id)
        spent = 0.0
        for event in events:
            if event.event_type != EventType.MODEL_COMPLETED:
                continue
            payload = event.payload
            if payload.kind != "model_completed":
                raise RuntimeError(
                    f"Invalid event payload kind {payload.kind!r} for model_completed event."
                )
            spent += payload.cost_usd
        return spent
