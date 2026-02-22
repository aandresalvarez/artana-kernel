from __future__ import annotations

from typing import Protocol

from artana.events import EventPayload, EventType, KernelEvent


class EventStore(Protocol):
    async def append_event(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
    ) -> KernelEvent:
        ...

    async def get_events_for_run(self, run_id: str) -> list[KernelEvent]:
        ...

    async def verify_run_chain(self, run_id: str) -> bool:
        ...

    async def close(self) -> None:
        ...
