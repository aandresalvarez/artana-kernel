from __future__ import annotations

from typing import Protocol, runtime_checkable

from artana.events import EventPayload, EventType, KernelEvent, RunSummaryPayload


class EventStore(Protocol):
    async def append_event(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
        parent_step_key: str | None = None,
    ) -> KernelEvent:
        ...

    async def get_events_for_run(self, run_id: str) -> list[KernelEvent]:
        ...

    async def get_latest_run_summary(
        self,
        run_id: str,
        summary_type: str,
    ) -> RunSummaryPayload | None:
        ...

    async def verify_run_chain(self, run_id: str) -> bool:
        ...

    async def close(self) -> None:
        ...


@runtime_checkable
class SupportsModelCostAggregation(Protocol):
    async def get_model_cost_sum_for_run(self, run_id: str) -> float:
        ...
