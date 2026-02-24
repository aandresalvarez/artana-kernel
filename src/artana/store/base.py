from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
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


@dataclass(frozen=True, slots=True)
class ToolSemanticOutcomeRecord:
    run_id: str
    request_id: str
    outcome: str
    request_step_key: str | None
    request_arguments_json: str


@dataclass(frozen=True, slots=True)
class RunLeaseRecord:
    run_id: str
    worker_id: str
    lease_expires_at: datetime


@runtime_checkable
class SupportsToolPolicyAggregation(Protocol):
    async def get_tool_request_count_for_run(self, *, run_id: str, tool_name: str) -> int:
        ...

    async def get_tool_request_count_for_tenant_since(
        self,
        *,
        tenant_id: str,
        tool_name: str,
        since: datetime,
    ) -> int:
        ...

    async def get_latest_tool_semantic_outcome(
        self,
        *,
        tenant_id: str,
        tool_name: str,
        semantic_idempotency_key: str,
    ) -> ToolSemanticOutcomeRecord | None:
        ...


@runtime_checkable
class SupportsRunIndexing(Protocol):
    async def list_run_ids(
        self,
        *,
        tenant_id: str | None = None,
        since: datetime | None = None,
    ) -> list[str]:
        ...


@runtime_checkable
class SupportsEventStreaming(Protocol):
    async def stream_events(
        self,
        run_id: str,
        *,
        since_seq: int = 0,
        follow: bool = False,
        poll_interval_seconds: float = 0.5,
        idle_timeout_seconds: float | None = None,
    ) -> AsyncIterator[KernelEvent]:
        ...


@runtime_checkable
class SupportsRunLeasing(Protocol):
    async def acquire_run_lease(
        self,
        *,
        run_id: str,
        worker_id: str,
        ttl_seconds: int,
    ) -> bool:
        ...

    async def renew_run_lease(
        self,
        *,
        run_id: str,
        worker_id: str,
        ttl_seconds: int,
    ) -> bool:
        ...

    async def release_run_lease(
        self,
        *,
        run_id: str,
        worker_id: str,
    ) -> bool:
        ...

    async def get_run_lease(
        self,
        *,
        run_id: str,
    ) -> RunLeaseRecord | None:
        ...
