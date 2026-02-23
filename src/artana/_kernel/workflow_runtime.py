from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from artana._kernel.replay import validate_tenant_for_run
from artana._kernel.types import PauseTicket, ReplayConsistencyError
from artana.canonicalization import canonical_json_dumps
from artana.events import (
    EventType,
    KernelEvent,
    RunStartedPayload,
    WorkflowStepCompletedPayload,
    WorkflowStepRequestedPayload,
)
from artana.models import TenantContext
from artana.store.base import EventStore

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

StepT = TypeVar("StepT")
WorkflowOutputT = TypeVar("WorkflowOutputT")
WorkflowModelT = TypeVar("WorkflowModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class StepSerde(Generic[StepT]):
    dump: Callable[[StepT], str]
    load: Callable[[str], StepT]


def json_step_serde() -> StepSerde[JsonValue]:
    return StepSerde(
        dump=lambda value: canonical_json_dumps(value),
        load=lambda raw: _load_json_value(raw),
    )


def pydantic_step_serde(model: type[WorkflowModelT]) -> StepSerde[WorkflowModelT]:
    return StepSerde(
        dump=lambda value: value.model_dump_json(),
        load=lambda raw: model.model_validate_json(raw),
    )


def _load_json_value(raw: str) -> JsonValue:
    parsed = json.loads(raw)
    return _validate_json_value(parsed)


def _validate_json_value(value: object) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, list):
        return [_validate_json_value(item) for item in value]
    if isinstance(value, dict):
        validated: dict[str, JsonValue] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings.")
            validated[key] = _validate_json_value(nested)
        return validated
    raise TypeError(f"Unsupported JSON value type {type(value)!r}.")


@dataclass(frozen=True, slots=True)
class WorkflowRunResult(Generic[WorkflowOutputT]):
    run_id: str
    status: Literal["complete", "paused"]
    output: WorkflowOutputT | None
    pause_ticket: PauseTicket | None


class WorkflowPausedInterrupt(RuntimeError):
    def __init__(self, ticket: PauseTicket) -> None:
        super().__init__(f"Workflow paused for human review: ticket={ticket.ticket_id}")
        self.ticket = ticket


class _PauseAPI(Protocol):
    async def pause(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        reason: str,
        context: BaseModel | None = None,
        step_key: str | None = None,
    ) -> PauseTicket:
        ...


class WorkflowContext:
    def __init__(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        store: EventStore,
        pause_api: _PauseAPI,
        events: Sequence[KernelEvent],
    ) -> None:
        self.run_id = run_id
        self.tenant = tenant
        self.is_resuming = len(events) > 0
        self._store = store
        self._pause_api = pause_api
        self._cursor = 0
        self._requested_by_index: dict[int, WorkflowStepRequestedPayload] = {}
        self._completed_by_index: dict[int, WorkflowStepCompletedPayload] = {}
        self._load_step_cache(events)

    async def step(
        self,
        *,
        name: str,
        action: Callable[[], Awaitable[StepT]],
        serde: StepSerde[StepT],
    ) -> StepT:
        step_index = self._cursor
        self._cursor += 1

        cached_completed = self._completed_by_index.get(step_index)
        if cached_completed is not None:
            if cached_completed.step_name != name:
                raise ReplayConsistencyError(
                    f"Workflow step mismatch at index={step_index}. "
                    f"expected {cached_completed.step_name!r}, got {name!r}."
                )
            return serde.load(cached_completed.result_json)

        existing_requested = self._requested_by_index.get(step_index)
        if existing_requested is not None and existing_requested.step_name != name:
            raise ReplayConsistencyError(
                f"Workflow step mismatch at requested index={step_index}. "
                f"expected {existing_requested.step_name!r}, got {name!r}."
            )

        if existing_requested is None:
            await self._store.append_event(
                run_id=self.run_id,
                tenant_id=self.tenant.tenant_id,
                event_type=EventType.WORKFLOW_STEP_REQUESTED,
                payload=WorkflowStepRequestedPayload(
                    step_index=step_index,
                    step_name=name,
                ),
            )

        result = await action()
        serialized = serde.dump(result)
        await self._store.append_event(
            run_id=self.run_id,
            tenant_id=self.tenant.tenant_id,
            event_type=EventType.WORKFLOW_STEP_COMPLETED,
            payload=WorkflowStepCompletedPayload(
                step_index=step_index,
                step_name=name,
                result_json=serialized,
            ),
        )
        self._completed_by_index[step_index] = WorkflowStepCompletedPayload(
            step_index=step_index,
            step_name=name,
            result_json=serialized,
        )
        return result

    async def pause(
        self,
        reason: str,
        *,
        context: BaseModel | None = None,
        step_key: str | None = None,
    ) -> PauseTicket:
        ticket = await self._pause_api.pause(
            run_id=self.run_id,
            tenant=self.tenant,
            reason=reason,
            context=context,
            step_key=step_key,
        )
        raise WorkflowPausedInterrupt(ticket)

    def _load_step_cache(self, events: Sequence[KernelEvent]) -> None:
        for event in events:
            if event.event_type == EventType.WORKFLOW_STEP_REQUESTED:
                payload = event.payload
                if not isinstance(payload, WorkflowStepRequestedPayload):
                    raise ReplayConsistencyError(
                        f"Invalid workflow_step_requested payload at seq={event.seq}."
                    )
                self._requested_by_index[payload.step_index] = payload
            if event.event_type == EventType.WORKFLOW_STEP_COMPLETED:
                payload = event.payload
                if not isinstance(payload, WorkflowStepCompletedPayload):
                    raise ReplayConsistencyError(
                        f"Invalid workflow_step_completed payload at seq={event.seq}."
                    )
                self._completed_by_index[payload.step_index] = payload


async def run_workflow(
    *,
    store: EventStore,
    pause_api: _PauseAPI,
    run_id: str | None,
    tenant: TenantContext,
    workflow: Callable[[WorkflowContext], Awaitable[WorkflowOutputT]],
) -> WorkflowRunResult[WorkflowOutputT]:
    run_id_value = run_id if run_id is not None else uuid4().hex
    events = await store.get_events_for_run(run_id_value)
    if len(events) == 0:
        await store.append_event(
            run_id=run_id_value,
            tenant_id=tenant.tenant_id,
            event_type=EventType.RUN_STARTED,
            payload=RunStartedPayload(),
        )
        events = await store.get_events_for_run(run_id_value)
    validate_tenant_for_run(events=events, tenant=tenant)
    context = WorkflowContext(
        run_id=run_id_value,
        tenant=tenant,
        store=store,
        pause_api=pause_api,
        events=events,
    )
    try:
        output = await workflow(context)
    except WorkflowPausedInterrupt as paused:
        return WorkflowRunResult(
            run_id=run_id_value,
            status="paused",
            output=None,
            pause_ticket=paused.ticket,
        )
    return WorkflowRunResult(
        run_id=run_id_value,
        status="complete",
        output=output,
        pause_ticket=None,
    )
