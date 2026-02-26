from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Literal, cast

from pydantic import BaseModel, Field

from artana.events import EventType, RunSummaryPayload
from artana.harness.base import BaseHarness, HarnessContext, HarnessStateError
from artana.kernel import ArtanaKernel, ReplayPolicy
from artana.models import TenantContext


class TaskUnit(BaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    state: Literal["pending", "in_progress", "done"] = "pending"


class TaskProgressSnapshot(BaseModel):
    units: list[TaskUnit] = Field(default_factory=list)


class TaskProgressValidationError(HarnessStateError):
    pass


type SanityCheckHook = Callable[[HarnessContext], Awaitable[None]]


class IncrementalTaskHarness(BaseHarness[tuple[TaskUnit, ...]]):
    SUMMARY_TYPE = "task_progress"

    def __init__(
        self,
        kernel: ArtanaKernel,
        tenant: TenantContext | None = None,
        *,
        default_model: str = "gpt-4o-mini",
        draft_model: str = "gpt-5.3-codex-spark",
        verify_model: str = "gpt-5.3-codex",
        replay_policy: ReplayPolicy = "allow_prompt_drift",
        sanity_check_hook: SanityCheckHook | None = None,
    ) -> None:
        super().__init__(
            kernel=kernel,
            tenant=tenant,
            default_model=default_model,
            draft_model=draft_model,
            verify_model=verify_model,
            replay_policy=replay_policy,
        )
        self._sanity_check_hook = sanity_check_hook
        self._session_run_id: str | None = None
        self._done_transitions_in_session = 0

    async def define_tasks(self) -> list[TaskUnit]:
        raise NotImplementedError(
            "IncrementalTaskHarness.define_tasks() must be implemented when using "
            "the default incremental execution flow."
        )

    async def work_on(self, task: TaskUnit) -> None:
        raise NotImplementedError(
            "IncrementalTaskHarness.work_on() must be implemented when using "
            "the default incremental execution flow."
        )

    async def step(self, *, context: HarnessContext) -> tuple[TaskUnit, ...]:
        if not self._uses_structured_flow:
            existing = await self.get_task_progress(run_id=context.run_id)
            if existing is None:
                return ()
            return existing

        task_progress = await self._ensure_task_progress_initialized(context=context)
        pending_task = next((unit for unit in task_progress if unit.state == "pending"), None)
        if pending_task is None:
            return task_progress

        await self.transition_task_unit(
            run_id=context.run_id,
            tenant=context.tenant,
            unit_id=pending_task.id,
            new_state="in_progress",
            step_key=f"task_{pending_task.id}_in_progress",
        )
        try:
            await self._invoke_work_on(task=pending_task, context=context)
        except Exception:
            await self.transition_task_unit(
                run_id=context.run_id,
                tenant=context.tenant,
                unit_id=pending_task.id,
                new_state="pending",
                step_key=f"task_{pending_task.id}_reset_pending",
            )
            raise

        await self.transition_task_unit(
            run_id=context.run_id,
            tenant=context.tenant,
            unit_id=pending_task.id,
            new_state="done",
            step_key=f"task_{pending_task.id}_done",
            verification_passed=True,
        )
        existing = await self.get_task_progress(run_id=context.run_id)
        if existing is None:
            return ()
        return existing

    async def on_initialize(self, *, context: HarnessContext) -> None:
        if not self._uses_structured_flow:
            return
        await self._ensure_task_progress_initialized(context=context)

    async def on_wake(
        self,
        *,
        context: HarnessContext,
        reorientation: Mapping[str, object],
    ) -> None:
        self._session_run_id = context.run_id
        self._done_transitions_in_session = 0
        if self._sanity_check_hook is not None:
            await self._sanity_check_hook(context)
        if self._uses_structured_flow:
            await self._ensure_task_progress_initialized(context=context)

    async def on_sleep(
        self,
        *,
        context: HarnessContext,
        execution_error: Exception | None,
    ) -> None:
        await super().on_sleep(context=context, execution_error=execution_error)
        await self._validate_no_partial_task_state(run_id=context.run_id)

    async def set_task_progress(
        self,
        *,
        run_id: str | None = None,
        tenant: TenantContext | None = None,
        units: Sequence[TaskUnit],
        step_key: str | None = None,
        verified_done_unit_ids: frozenset[str] = frozenset(),
    ) -> int:
        resolved_run_id = self._resolve_run_id(run_id=run_id)
        resolved_tenant = self._resolve_tenant(tenant=tenant)
        required_step_key = (
            self.require_step_key(step_key)
            if step_key is not None
            else self._next_step_key(prefix=self.SUMMARY_TYPE)
        )
        previous = await self.get_task_progress(run_id=resolved_run_id)
        previous_units = previous if previous is not None else ()
        current_units = tuple(unit.model_copy() for unit in units)
        done_transitions = self._validate_task_progress_update(
            run_id=resolved_run_id,
            previous=previous_units,
            current=current_units,
            verified_done_unit_ids=verified_done_unit_ids,
        )
        self._done_transitions_in_session += done_transitions
        return await self.emit_summary(
            run_id=resolved_run_id,
            tenant=resolved_tenant,
            summary_type=self.SUMMARY_TYPE,
            payload=TaskProgressSnapshot(units=list(current_units)).model_dump(mode="json"),
            step_key=required_step_key,
        )

    async def get_task_progress(
        self,
        *,
        run_id: str | None = None,
    ) -> tuple[TaskUnit, ...] | None:
        payload = await self.summary_payload(
            run_id=self._resolve_run_id(run_id=run_id),
            summary_type=self.SUMMARY_TYPE,
        )
        if payload is None:
            return None
        if isinstance(payload, list):
            snapshot = TaskProgressSnapshot(
                units=[TaskUnit.model_validate(item) for item in payload]
            )
            return tuple(snapshot.units)
        if isinstance(payload, dict):
            snapshot = TaskProgressSnapshot.model_validate(payload)
            return tuple(snapshot.units)
        raise TaskProgressValidationError(
            "task_progress summary payload must be an object or list."
        )

    async def transition_task_unit(
        self,
        *,
        run_id: str | None = None,
        tenant: TenantContext | None = None,
        unit_id: str,
        new_state: Literal["pending", "in_progress", "done"],
        step_key: str | None = None,
        verification_passed: bool = False,
    ) -> int:
        resolved_run_id = self._resolve_run_id(run_id=run_id)
        resolved_tenant = self._resolve_tenant(tenant=tenant)
        units = await self.get_task_progress(run_id=resolved_run_id)
        if units is None:
            raise TaskProgressValidationError(
                "Cannot transition task state before task_progress has been initialized."
            )
        mutated: list[TaskUnit] = []
        matched = False
        for unit in units:
            if unit.id != unit_id:
                mutated.append(unit)
                continue
            matched = True
            mutated.append(unit.model_copy(update={"state": new_state}))
        if not matched:
            raise TaskProgressValidationError(f"Unknown TaskUnit id={unit_id!r}.")
        verified: frozenset[str] = frozenset({unit_id}) if verification_passed else frozenset()
        return await self.set_task_progress(
            run_id=resolved_run_id,
            tenant=resolved_tenant,
            units=mutated,
            step_key=(
                self.require_step_key(step_key)
                if step_key is not None
                else self._next_step_key(prefix=f"task_{unit_id}_{new_state}")
            ),
            verified_done_unit_ids=verified,
        )

    async def _build_wake_reorientation(
        self,
        *,
        context: HarnessContext,
    ) -> dict[str, object]:
        base_payload = await super()._build_wake_reorientation(context=context)
        task_progress = await self.get_task_progress(run_id=context.run_id)
        latest_run_summary = await self._latest_run_summary(run_id=context.run_id)
        base_payload.update(
            {
                "task_progress": (
                    [unit.model_dump(mode="json") for unit in task_progress]
                    if task_progress is not None
                    else None
                ),
                "latest_run_summary": latest_run_summary,
            }
        )
        return base_payload

    async def _latest_run_summary(self, *, run_id: str) -> dict[str, object] | None:
        events = await self._kernel.get_events(run_id=run_id)
        for event in reversed(events):
            if event.event_type != EventType.RUN_SUMMARY:
                continue
            payload = event.payload
            if not isinstance(payload, RunSummaryPayload):
                continue
            try:
                summary_payload: object = json.loads(payload.summary_json)
            except json.JSONDecodeError:
                summary_payload = payload.summary_json
            return {
                "summary_type": payload.summary_type,
                "step_key": payload.step_key,
                "summary": summary_payload,
            }
        return None

    async def _validate_no_partial_task_state(self, *, run_id: str) -> None:
        task_progress = await self.get_task_progress(run_id=run_id)
        if task_progress is None:
            return
        partial_units = [unit.id for unit in task_progress if unit.state == "in_progress"]
        if partial_units:
            unit_ids = ", ".join(partial_units)
            raise TaskProgressValidationError(
                "Cannot sleep with partial TaskUnit state; units still in progress: "
                f"{unit_ids}."
            )

    def _validate_task_progress_update(
        self,
        *,
        run_id: str,
        previous: Sequence[TaskUnit],
        current: Sequence[TaskUnit],
        verified_done_unit_ids: frozenset[str],
    ) -> int:
        self._ensure_session_tracking(run_id=run_id)
        previous_map = _task_unit_map(previous)
        current_map = _task_unit_map(current)

        removed_ids = set(previous_map).difference(current_map)
        if removed_ids:
            removed = ", ".join(sorted(removed_ids))
            raise TaskProgressValidationError(
                f"TaskUnit deletion is not allowed. Removed ids: {removed}."
            )

        done_transitions: list[str] = []
        for unit_id, current_unit in current_map.items():
            previous_unit = previous_map.get(unit_id)
            previous_state = previous_unit.state if previous_unit is not None else None
            if current_unit.state == "done" and previous_state != "done":
                done_transitions.append(unit_id)

        if self._done_transitions_in_session + len(done_transitions) > 1:
            raise TaskProgressValidationError(
                "Only one TaskUnit can transition to done in a single harness session."
            )

        unverified = [
            unit_id
            for unit_id in done_transitions
            if unit_id not in verified_done_unit_ids
        ]
        if unverified:
            missing = ", ".join(sorted(unverified))
            raise TaskProgressValidationError(
                "TaskUnit transition to done requires explicit verification for ids: "
                f"{missing}."
            )
        return len(done_transitions)

    def _ensure_session_tracking(self, *, run_id: str) -> None:
        if self._session_run_id == run_id:
            return
        self._session_run_id = run_id
        self._done_transitions_in_session = 0

    @property
    def _uses_structured_flow(self) -> bool:
        define_overridden = type(self).define_tasks is not IncrementalTaskHarness.define_tasks
        work_overridden = type(self).work_on is not IncrementalTaskHarness.work_on
        if define_overridden != work_overridden:
            raise TaskProgressValidationError(
                "Structured incremental flow requires overriding both "
                "define_tasks() and work_on()."
            )
        return define_overridden and work_overridden

    async def _ensure_task_progress_initialized(
        self,
        *,
        context: HarnessContext,
    ) -> tuple[TaskUnit, ...]:
        existing = await self.get_task_progress(run_id=context.run_id)
        if existing is not None:
            return existing

        defined_units = await self._invoke_define_tasks(context=context)
        snapshot = tuple(
            unit if unit.state == "pending" else unit.model_copy(update={"state": "pending"})
            for unit in defined_units
        )
        await self.set_task_progress(
            run_id=context.run_id,
            tenant=context.tenant,
            units=snapshot,
            step_key="task_progress_init",
        )
        stored = await self.get_task_progress(run_id=context.run_id)
        if stored is not None:
            return stored
        return snapshot

    async def _invoke_define_tasks(self, *, context: HarnessContext) -> list[TaskUnit]:
        call = cast(Callable[..., Awaitable[object]], self.define_tasks)
        params = tuple(inspect.signature(call).parameters.values())
        if len(params) == 0:
            units_obj = await call()
        elif len(params) == 1:
            name = params[0].name
            if name == "run_id":
                units_obj = await call(context.run_id)
            elif name == "tenant":
                units_obj = await call(context.tenant)
            else:
                units_obj = await call(context)
        else:
            raise TypeError(
                "define_tasks must accept 0 or 1 arguments (context/run_id/tenant)."
            )
        if not isinstance(units_obj, Sequence):
            raise TypeError("define_tasks must return a sequence of TaskUnit values.")
        units: Sequence[object] = units_obj
        return [TaskUnit.model_validate(unit) for unit in units]

    async def _invoke_work_on(
        self,
        *,
        task: TaskUnit,
        context: HarnessContext,
    ) -> None:
        call = cast(Callable[..., Awaitable[None]], self.work_on)
        params = tuple(inspect.signature(call).parameters.values())
        if len(params) == 1:
            name = params[0].name
            if name in {"context", "ctx", "run_context"}:
                await call(context)
                return
            await call(task)
            return
        if len(params) == 2:
            bound_args: list[object] = []
            for parameter in params:
                if parameter.name in {"task", "unit", "task_unit"}:
                    bound_args.append(task)
                elif parameter.name in {"context", "ctx", "run_context"}:
                    bound_args.append(context)
                elif parameter.name == "run_id":
                    bound_args.append(context.run_id)
                elif parameter.name == "tenant":
                    bound_args.append(context.tenant)
                else:
                    bound_args.append(task if not bound_args else context)
            await call(*bound_args)
            return
        raise TypeError(
            "work_on must accept 1 argument (task) or 2 arguments (task + context)."
        )


def _task_unit_map(units: Sequence[TaskUnit]) -> dict[str, TaskUnit]:
    unit_map: dict[str, TaskUnit] = {}
    for unit in units:
        if unit.id in unit_map:
            raise TaskProgressValidationError(
                f"Duplicate TaskUnit id detected: {unit.id!r}."
            )
        unit_map[unit.id] = unit
    return unit_map


__all__ = [
    "IncrementalTaskHarness",
    "SanityCheckHook",
    "TaskProgressSnapshot",
    "TaskProgressValidationError",
    "TaskUnit",
]
