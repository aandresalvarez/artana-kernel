from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

from artana.agent.autonomous import AutonomousAgent
from artana.agent.model_steps import execute_model_step
from artana.events import (
    ChatMessage,
    EventPayload,
    EventType,
    HarnessFailedPayload,
    HarnessInitializedPayload,
    HarnessSleepPayload,
    HarnessStagePayload,
    HarnessWakePayload,
)
from artana.kernel import (
    ArtanaKernel,
    ContextVersion,
    ModelInput,
    ReplayPolicy,
    StepModelResult,
    StepToolResult,
    TraceLevel,
    resolve_tool_resolutions,
)
from artana.models import TenantContext
from artana.ports.model import ModelCallOptions, ToolDefinition

OutputT = TypeVar("OutputT", bound=BaseModel)
HarnessResultT = TypeVar("HarnessResultT")


@dataclass(frozen=True, slots=True)
class HarnessContext:
    run_id: str
    tenant: TenantContext
    model: str
    run_created: bool


class HarnessStateError(RuntimeError):
    pass


class BaseHarness(ABC, Generic[HarnessResultT]):
    def __init__(
        self,
        kernel: ArtanaKernel,
        tenant: TenantContext | None = None,
        *,
        default_model: str = "gpt-4o-mini",
        replay_policy: ReplayPolicy = "allow_prompt_drift",
        trace_level: TraceLevel = "stage",
    ) -> None:
        self._kernel = kernel
        self._tenant = tenant
        self._default_model = default_model
        self._replay_policy = replay_policy
        self._trace_level = trace_level
        self._step_key_namespace = _normalize_step_key_prefix(type(self).__name__)
        self._active_context: HarnessContext | None = None
        self._step_key_counters: dict[str, int] = {}
        self._active_trace_level = trace_level
        self._active_trace_parent_step_key: str | None = None
        self._active_trace_step_key: str | None = None
        self._active_trace_last_step_key: str | None = None
        self._active_trace_stage_round: int | None = None
        self._active_trace_stage_cost: float = 0.0
        self._active_trace_stage_model_calls: int = 0
        self._active_trace_stage_tool_calls: int = 0
        self._active_trace_stage_started_at: float | None = None
        self._active_trace_stage_prev_step_key: str | None = None
        self._active_trace_stage_prev_parent_step_key: str | None = None
        self._active_trace_previous_stage_name: str | None = None

    @property
    def kernel(self) -> ArtanaKernel:
        return self._kernel

    @property
    def tenant(self) -> TenantContext | None:
        return self._tenant

    async def run(
        self,
        run_id: str,
        *,
        tenant: TenantContext | None = None,
        model: str | None = None,
        trace_level: TraceLevel | None = None,
    ) -> HarnessResultT:
        resolved_tenant = self._resolve_tenant(tenant)
        run_created = await self._ensure_run_exists(run_id=run_id, tenant=resolved_tenant)
        context = HarnessContext(
            run_id=run_id,
            tenant=resolved_tenant,
            model=model if model is not None else self._default_model,
            run_created=run_created,
        )
        result: HarnessResultT | None = None
        execution_error: Exception | None = None
        sleep_error: Exception | None = None
        failure_parent_step_key: str | None = None
        try:
            self._activate_context(context=context)
            self._set_active_trace_level(self._resolve_trace_level(trace_level))
            reorientation = await self._build_wake_reorientation(context=context)

            initialize_stage = await self._start_trace_stage(
                stage="initialize",
                round=1,
            )
            if context.run_created:
                if self._should_emit_stage_trace():
                    await self._append_harness_event(
                        event_type=EventType.HARNESS_INITIALIZED,
                        payload=HarnessInitializedPayload(
                            harness_name=type(self).__name__,
                            model=context.model,
                        ),
                    )
                try:
                    await self.on_initialize(context=context)
                except Exception as exc:
                    execution_error = exc
                    failure_parent_step_key = initialize_stage
            await self._finish_trace_stage(
                stage="initialize",
                round=1,
            )

            wake_stage = await self._start_trace_stage(
                stage="wake",
                round=2,
            )
            if self._should_emit_stage_trace():
                await self._append_harness_event(
                    event_type=EventType.HARNESS_WAKE,
                    payload=HarnessWakePayload(
                        run_created=context.run_created,
                        reason=(
                            "resuming"
                            if reorientation["run_created"] is False
                            else "initializing"
                        ),
                    ),
                )
            if execution_error is None:
                try:
                    await self.on_wake(context=context, reorientation=reorientation)
                except Exception as exc:
                    execution_error = exc
                    failure_parent_step_key = wake_stage
            await self._finish_trace_stage(
                stage="wake",
                round=2,
            )

            work_stage = await self._start_trace_stage(
                stage="work",
                round=3,
            )
            if execution_error is None:
                try:
                    result = await self.step(context=context)
                except Exception as exc:
                    execution_error = exc
                    failure_parent_step_key = work_stage
            await self._finish_trace_stage(
                stage="work",
                round=3,
            )

            sleep_stage = await self._start_trace_stage(
                stage="sleep",
                round=4,
            )
            try:
                await self.on_sleep(
                    context=context, execution_error=execution_error
                )
            except Exception as exc:
                sleep_error = exc
                if failure_parent_step_key is None:
                    failure_parent_step_key = sleep_stage
            if self._should_emit_stage_trace() or self._should_emit_verbose_trace():
                status: Literal["completed", "failed"] = (
                    "failed"
                    if execution_error is not None or sleep_error is not None
                    else "completed"
                )
                await self._append_harness_event(
                    event_type=EventType.HARNESS_SLEEP,
                    payload=HarnessSleepPayload(
                        status=status,
                        execution_error_type=self._exception_type(execution_error),
                        sleep_error_type=self._exception_type(sleep_error),
                    ),
                    parent_step_key=self._active_trace_step_key,
                )
                await self.emit_summary(
                    run_id=context.run_id,
                    tenant=context.tenant,
                    summary_type="harness_sleep",
                    payload={
                        "status": status,
                        "execution_error": _error_payload(execution_error),
                        "sleep_error": _error_payload(sleep_error),
                    },
                    step_key="harness_sleep",
                    parent_step_key=self._active_trace_step_key,
                )
            await self._finish_trace_stage(
                stage="sleep",
                round=4,
            )

            if self._should_emit_stage_trace() or self._should_emit_verbose_trace():
                if result is not None:
                    await self.emit_summary(
                        run_id=context.run_id,
                        tenant=context.tenant,
                        summary_type="wake_reorientation",
                        payload=reorientation,
                        step_key="wake_reorientation",
                        parent_step_key=sleep_stage,
                    )

            if execution_error is not None:
                await self._append_harness_failed(
                    error=execution_error,
                    parent_step_key=failure_parent_step_key,
                )
                raise execution_error
            if sleep_error is not None:
                await self._append_harness_failed(
                    error=sleep_error,
                    parent_step_key=failure_parent_step_key,
                )
                raise sleep_error
            if result is None:
                raise RuntimeError("Harness completed without returning a result.")
            return result
        finally:
            self._deactivate_context()

    def _resolve_trace_level(self, trace_level: TraceLevel | None) -> TraceLevel:
        if trace_level is None:
            return self._trace_level
        return trace_level

    def _set_active_trace_level(self, trace_level: TraceLevel) -> None:
        if trace_level not in ("minimal", "stage", "verbose"):
            raise ValueError(f"Unknown trace_level={trace_level!r}.")
        self._active_trace_level = trace_level

    def _should_emit_stage_trace(self) -> bool:
        return self._active_trace_level in ("stage", "verbose")

    def _should_emit_verbose_trace(self) -> bool:
        return self._active_trace_level == "verbose"

    def _set_active_trace_step(self, step_key: str | None) -> str | None:
        previous_step_key = self._active_trace_step_key
        if step_key is not None:
            self._active_trace_last_step_key = step_key
        self._active_trace_step_key = step_key
        return previous_step_key

    def _restore_active_trace_step(self, step_key: str | None) -> None:
        self._active_trace_step_key = step_key

    def _exception_type(self, exc: Exception | None) -> str | None:
        if exc is None:
            return None
        return type(exc).__name__

    async def _append_harness_event(
        self,
        *,
        event_type: EventType,
        payload: EventPayload,
        parent_step_key: str | None = None,
    ) -> None:
        if self._active_context is None:
            raise HarnessStateError(
                "Cannot append harness event without an active harness context."
            )
        await self._kernel.append_harness_event(
            run_id=self._active_context.run_id,
            tenant=self._active_context.tenant,
            event_type=event_type,
            payload=payload,
            parent_step_key=(
                self._active_trace_step_key
                if parent_step_key is None
                else parent_step_key
            ),
        )

    async def _append_harness_failed(
        self,
        *,
        error: Exception,
        parent_step_key: str | None = None,
    ) -> None:
        await self._append_harness_event(
            event_type=EventType.HARNESS_FAILED,
            payload=HarnessFailedPayload(
                error_type=type(error).__name__,
                message=str(error),
                last_step_key=self._active_trace_last_step_key,
            ),
            parent_step_key=parent_step_key,
        )

    async def _start_trace_stage(
        self,
        *,
        stage: str,
        round: int,
    ) -> str:
        stage_key = self._next_step_key(prefix=f"stage_{stage}")
        from_stage = self._active_trace_previous_stage_name
        self._active_trace_stage_round = round
        self._active_trace_stage_cost = 0.0
        self._active_trace_stage_model_calls = 0
        self._active_trace_stage_tool_calls = 0
        self._active_trace_stage_started_at = monotonic()
        self._active_trace_stage_prev_step_key = self._active_trace_step_key
        self._active_trace_stage_prev_parent_step_key = self._active_trace_parent_step_key
        self._active_trace_parent_step_key = self._active_trace_step_key
        self._set_active_trace_step(stage_key)

        if self._should_emit_stage_trace():
            await self.emit_summary(
                run_id=self._resolve_run_id(run_id=None),
                tenant=self._resolve_tenant(tenant=None),
                summary_type="trace::state_transition",
                payload={
                    "from_stage": from_stage,
                    "to_stage": stage,
                    "round": round,
                },
                step_key=f"trace_state_transition_{stage}_{round}",
                parent_step_key=self._active_trace_step_key,
            )
            await self._append_harness_event(
                event_type=EventType.HARNESS_STAGE,
                payload=HarnessStagePayload(
                    stage=stage,
                    round=round,
                ),
            )
        return stage_key

    async def _finish_trace_stage(
        self,
        *,
        stage: str,
        round: int,
    ) -> None:
        duration_ms: int | None = None
        if self._active_trace_stage_started_at is not None:
            duration_ms = int(
                (monotonic() - self._active_trace_stage_started_at) * 1000
            )
        if self._active_trace_stage_round is not None:
            claims_count = (
                self._active_trace_stage_model_calls
                + self._active_trace_stage_tool_calls
            )
            claims_payload = {
                "stage": stage,
                "round": self._active_trace_stage_round,
            }
            if duration_ms is not None:
                claims_payload["logical_duration_ms"] = duration_ms
            if claims_count:
                claims_payload["claims_count"] = claims_count
            if self._should_emit_stage_trace():
                await self.emit_summary(
                    run_id=self._resolve_run_id(run_id=None),
                    tenant=self._resolve_tenant(tenant=None),
                    summary_type="trace::round",
                    payload=claims_payload,
                    step_key=f"trace_round_{stage}_{round}_{claims_count}"
                    if claims_count
                    else f"trace_round_{stage}_{round}",
                    parent_step_key=self._active_trace_step_key,
                )
                cost_payload = {
                    "stage": stage,
                    "round": round,
                    "model_cost": self._active_trace_stage_cost,
                    "tool_cost": 0.0,
                    "total_cost": self._active_trace_stage_cost,
                    "logical_duration_ms": duration_ms or 0,
                }
                await self.emit_summary(
                    run_id=self._resolve_run_id(run_id=None),
                    tenant=self._resolve_tenant(tenant=None),
                    summary_type="trace::cost",
                    payload=cost_payload,
                    step_key=f"trace_cost_{stage}_{round}",
                    parent_step_key=self._active_trace_step_key,
                )
                await self.emit_summary(
                    run_id=self._resolve_run_id(run_id=None),
                    tenant=self._resolve_tenant(tenant=None),
                    summary_type="trace::cost_snapshot",
                    payload=cost_payload,
                    step_key=f"trace_cost_snapshot_{stage}_{round}",
                    parent_step_key=self._active_trace_step_key,
                )

        self._restore_active_trace_step(self._active_trace_stage_prev_step_key)
        self._active_trace_parent_step_key = self._active_trace_stage_prev_parent_step_key
        self._active_trace_previous_stage_name = stage
        self._active_trace_stage_round = None
        self._active_trace_stage_cost = 0.0
        self._active_trace_stage_model_calls = 0
        self._active_trace_stage_tool_calls = 0
        self._active_trace_stage_started_at = None
        self._active_trace_stage_prev_step_key = None
        self._active_trace_stage_prev_parent_step_key = None

    async def on_initialize(self, *, context: HarnessContext) -> None:
        return None

    async def on_wake(
        self,
        *,
        context: HarnessContext,
        reorientation: Mapping[str, object],
    ) -> None:
        return None

    async def on_sleep(
        self,
        *,
        context: HarnessContext,
        execution_error: Exception | None,
    ) -> None:
        await self.validate_clean_state(run_id=context.run_id)

    @abstractmethod
    async def step(self, *, context: HarnessContext) -> HarnessResultT:
        raise NotImplementedError

    async def read_summary(
        self,
        summary_type: str,
        *,
        run_id: str | None = None,
    ) -> object | None:
        return await self.summary_payload(
            run_id=self._resolve_run_id(run_id=run_id),
            summary_type=summary_type,
        )

    async def write_summary(
        self,
        summary_type: str,
        payload: object,
        *,
        step_key: str | None = None,
        run_id: str | None = None,
        tenant: TenantContext | None = None,
    ) -> int:
        return await self.emit_summary(
            run_id=self._resolve_run_id(run_id=run_id),
            tenant=self._resolve_tenant(tenant=tenant),
            summary_type=summary_type,
            payload=payload,
            step_key=(
                self.require_step_key(step_key)
                if step_key is not None
                else self._next_step_key(prefix=f"summary_{summary_type}")
            ),
        )

    def list_tools(
        self,
        *,
        visible_tool_names: set[str] | None = None,
        tenant: TenantContext | None = None,
    ) -> tuple[ToolDefinition, ...]:
        resolved_tenant = self._resolve_tenant(tenant=tenant)
        return self._kernel.list_tools(
            tenant_capabilities=resolved_tenant.capabilities,
            visible_tool_names=visible_tool_names,
        )

    async def run_model(
        self,
        *,
        output_schema: type[OutputT],
        prompt: str | None = None,
        messages: Sequence[ChatMessage] | None = None,
        input: ModelInput | None = None,
        model: str | None = None,
        step_key: str | None = None,
        run_id: str | None = None,
        tenant: TenantContext | None = None,
        visible_tool_names: set[str] | None = None,
        model_options: ModelCallOptions | None = None,
        replay_policy: ReplayPolicy | None = None,
        context_version: ContextVersion | None = None,
        parent_step_key: str | None = None,
    ) -> StepModelResult[OutputT]:
        resolved_input = _resolve_model_input(
            input=input,
            prompt=prompt,
            messages=messages,
        )
        return await self.model_step(
            run_id=self._resolve_run_id(run_id=run_id),
            tenant=self._resolve_tenant(tenant=tenant),
            model=self._resolve_model(model=model),
            input=resolved_input,
            output_schema=output_schema,
            step_key=(
                self.require_step_key(step_key)
                if step_key is not None
                else self._next_step_key(prefix="model")
            ),
            visible_tool_names=visible_tool_names,
            model_options=model_options,
            replay_policy=replay_policy,
            context_version=context_version,
            parent_step_key=parent_step_key,
        )

    async def run_tool(
        self,
        *,
        tool_name: str,
        arguments: BaseModel,
        step_key: str | None = None,
        run_id: str | None = None,
        tenant: TenantContext | None = None,
        parent_step_key: str | None = None,
    ) -> StepToolResult:
        return await self.tool_step(
            run_id=self._resolve_run_id(run_id=run_id),
            tenant=self._resolve_tenant(tenant=tenant),
            tool_name=tool_name,
            arguments=arguments,
            step_key=(
                self.require_step_key(step_key)
                if step_key is not None
                else self._next_step_key(prefix=f"tool_{tool_name}")
            ),
            parent_step_key=parent_step_key,
        )

    async def emit_summary(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        summary_type: str,
        payload: object,
        step_key: str,
        parent_step_key: str | None = None,
    ) -> int:
        required_step_key = self.require_step_key(step_key)
        resolved_parent_step_key = (
            self._active_trace_step_key if parent_step_key is None else parent_step_key
        )
        return await self._kernel.append_run_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type=summary_type,
            summary_json=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            step_key=required_step_key,
            parent_step_key=resolved_parent_step_key,
        )

    async def summary_payload(
        self,
        *,
        run_id: str,
        summary_type: str,
    ) -> object | None:
        summary = await self._kernel.get_latest_run_summary(
            run_id=run_id,
            summary_type=summary_type,
        )
        if summary is None:
            return None
        try:
            decoded: object = json.loads(summary.summary_json)
            return decoded
        except json.JSONDecodeError:
            return summary.summary_json

    async def set_artifact(
        self,
        *,
        key: str,
        value: object,
        step_key: str | None = None,
        run_id: str | None = None,
        tenant: TenantContext | None = None,
    ) -> int:
        return await self.emit_summary(
            run_id=self._resolve_run_id(run_id=run_id),
            tenant=self._resolve_tenant(tenant=tenant),
            summary_type=f"artifact::{key}",
            payload={"value": value},
            step_key=(
                self.require_step_key(step_key)
                if step_key is not None
                else self._next_step_key(prefix=f"artifact_{key}")
            ),
            parent_step_key=self._active_trace_step_key,
        )

    async def get_artifact(
        self,
        *,
        key: str,
        run_id: str | None = None,
    ) -> object | None:
        payload = await self.summary_payload(
            run_id=self._resolve_run_id(run_id=run_id),
            summary_type=f"artifact::{key}",
        )
        if payload is None:
            return None
        if isinstance(payload, dict) and "value" in payload:
            value = payload.get("value")
            return value
        return payload

    async def model_step(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        input: ModelInput,
        output_schema: type[OutputT],
        step_key: str,
        visible_tool_names: set[str] | None = None,
        model_options: ModelCallOptions | None = None,
        replay_policy: ReplayPolicy | None = None,
        context_version: ContextVersion | None = None,
        parent_step_key: str | None = None,
    ) -> StepModelResult[OutputT]:
        required_step_key = self.require_step_key(step_key)
        resolved_parent_step_key = (
            self._active_trace_step_key if parent_step_key is None else parent_step_key
        )
        previous_step_key = self._set_active_trace_step(required_step_key)
        try:
            model_result = await self._kernel.step_model(
                run_id=run_id,
                tenant=tenant,
                model=model,
                input=input,
                output_schema=output_schema,
                step_key=required_step_key,
                visible_tool_names=visible_tool_names,
                model_options=model_options,
                replay_policy=(
                    self._replay_policy if replay_policy is None else replay_policy
                ),
                context_version=context_version,
                parent_step_key=resolved_parent_step_key,
            )
        finally:
            self._restore_active_trace_step(previous_step_key)
        if self._active_trace_stage_round is not None:
            self._active_trace_stage_cost += model_result.usage.cost_usd
            self._active_trace_stage_model_calls += 1
        if self._should_emit_stage_trace() or self._should_emit_verbose_trace():
            if model_result.drift_fields:
                await self.emit_summary(
                    run_id=run_id,
                    tenant=tenant,
                    summary_type="trace::drift",
                    payload={
                        "step_key": required_step_key,
                        "drift_fields": list(model_result.drift_fields),
                        "forked": model_result.forked_from_run_id is not None,
                    },
                    step_key=f"trace_drift_{required_step_key}",
                    parent_step_key=required_step_key,
                )
            if self._should_emit_verbose_trace():
                await self.emit_summary(
                    run_id=run_id,
                    tenant=tenant,
                    summary_type="trace::tool_validation",
                    payload={
                        "step_key": required_step_key,
                        "model": model,
                        "replayed": model_result.replayed,
                        "replayed_with_drift": model_result.replayed_with_drift,
                        "forked_from_run_id": model_result.forked_from_run_id,
                        "tool_calls": [
                            tool_call.tool_name for tool_call in model_result.tool_calls
                        ],
                        "cost_usd": model_result.usage.cost_usd,
                    },
                    step_key=f"trace_model_{required_step_key}",
                    parent_step_key=required_step_key,
                )
        return model_result

    async def model_step_from_messages(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        messages: tuple[ChatMessage, ...],
        output_schema: type[OutputT],
        step_key: str,
        visible_tool_names: set[str] | None = None,
        model_options: ModelCallOptions | None = None,
        replay_policy: ReplayPolicy | None = None,
        context_version: ContextVersion | None = None,
        parent_step_key: str | None = None,
    ) -> StepModelResult[OutputT]:
        required_step_key = self.require_step_key(step_key)
        previous_step_key = self._set_active_trace_step(required_step_key)
        resolved_parent_step_key = (
            self._active_trace_step_key if parent_step_key is None else parent_step_key
        )
        try:
            result = await execute_model_step(
                kernel=self._kernel,
                run_id=run_id,
                tenant=tenant,
                model=model,
                messages=messages,
                output_schema=output_schema,
                step_key=required_step_key,
                visible_tool_names=visible_tool_names,
                model_options=model_options,
                replay_policy=(
                    self._replay_policy if replay_policy is None else replay_policy
                ),
                context_version=context_version,
                parent_step_key=resolved_parent_step_key,
            )
        finally:
            self._restore_active_trace_step(previous_step_key)
        if self._active_trace_stage_round is not None:
            self._active_trace_stage_cost += result.usage.cost_usd
            self._active_trace_stage_model_calls += 1
        if self._should_emit_stage_trace() or self._should_emit_verbose_trace():
            if result.drift_fields:
                await self.emit_summary(
                    run_id=run_id,
                    tenant=tenant,
                    summary_type="trace::drift",
                    payload={
                        "step_key": required_step_key,
                        "drift_fields": list(result.drift_fields),
                        "forked": result.forked_from_run_id is not None,
                    },
                    step_key=f"trace_drift_{required_step_key}",
                    parent_step_key=required_step_key,
                )
            if self._should_emit_verbose_trace():
                await self.emit_summary(
                    run_id=run_id,
                    tenant=tenant,
                    summary_type="trace::tool_validation",
                    payload={
                        "step_key": required_step_key,
                        "model": model,
                        "replayed": result.replayed,
                        "replayed_with_drift": result.replayed_with_drift,
                        "forked_from_run_id": result.forked_from_run_id,
                        "tool_calls": [tool_call.tool_name for tool_call in result.tool_calls],
                        "cost_usd": result.usage.cost_usd,
                    },
                    step_key=f"trace_model_{required_step_key}",
                    parent_step_key=required_step_key,
                )
        return result

    async def tool_step(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments: BaseModel,
        step_key: str,
        parent_step_key: str | None = None,
    ) -> StepToolResult:
        required_step_key = self.require_step_key(step_key)
        resolved_parent_step_key = (
            self._active_trace_step_key if parent_step_key is None else parent_step_key
        )
        previous_step_key = self._set_active_trace_step(required_step_key)
        try:
            result = await self._kernel.step_tool(
                run_id=run_id,
                tenant=tenant,
                tool_name=tool_name,
                arguments=arguments,
                step_key=required_step_key,
                parent_step_key=resolved_parent_step_key,
            )
        finally:
            self._restore_active_trace_step(previous_step_key)
        if self._active_trace_stage_round is not None:
            self._active_trace_stage_tool_calls += 1
        if self._should_emit_stage_trace() or self._should_emit_verbose_trace():
            if self._should_emit_verbose_trace():
                await self.emit_summary(
                    run_id=run_id,
                    tenant=tenant,
                    summary_type="trace::tool_validation",
                    payload={
                        "step_key": required_step_key,
                        "tool_name": tool_name,
                        "replayed": result.replayed,
                        "result_json": result.result_json,
                    },
                    step_key=f"trace_tool_{required_step_key}",
                    parent_step_key=required_step_key,
                )
        return result

    async def run_autonomous_agent(
        self,
        *,
        run_id: str | None = None,
        tenant: TenantContext | None = None,
        model: str | None = None,
        prompt: str,
        output_schema: type[OutputT],
        system_prompt: str = "You are a helpful autonomous agent.",
        max_iterations: int = 15,
        replay_policy: ReplayPolicy | None = None,
    ) -> OutputT:
        agent = AutonomousAgent(
            kernel=self._kernel,
            replay_policy=self._replay_policy if replay_policy is None else replay_policy,
        )
        return await agent.run(
            run_id=self._resolve_run_id(run_id=run_id),
            tenant=self._resolve_tenant(tenant=tenant),
            model=self._resolve_model(model=model),
            prompt=prompt,
            output_schema=output_schema,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
        )

    async def validate_clean_state(self, *, run_id: str) -> None:
        events = await self._kernel.get_events(run_id=run_id)
        resolutions = resolve_tool_resolutions(events)

        unresolved = [
            resolution.request.event_id
            for resolution in resolutions
            if resolution.completion is None
        ]
        if unresolved:
            unresolved_ids = ", ".join(unresolved)
            raise HarnessStateError(
                "Cannot sleep with unresolved tool requests: "
                f"{unresolved_ids}."
            )

        unknown_outcome = [
            resolution.request.event_id
            for resolution in resolutions
            if (
                resolution.completion is not None
                and resolution.completion.payload.outcome == "unknown_outcome"
            )
        ]
        if unknown_outcome:
            unresolved_ids = ", ".join(unknown_outcome)
            raise HarnessStateError(
                "Cannot sleep with unknown tool outcomes; reconciliation is required for "
                f"request ids: {unresolved_ids}."
            )

    def require_step_key(self, step_key: str | None) -> str:
        if step_key is None or step_key.strip() == "":
            raise ValueError("Harness model/tool/summary steps require a non-empty step_key.")
        return step_key

    async def _ensure_run_exists(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
    ) -> bool:
        try:
            await self._kernel.load_run(run_id=run_id)
        except ValueError:
            await self._kernel.start_run(tenant=tenant, run_id=run_id)
            return True
        return False

    async def _build_wake_reorientation(
        self,
        *,
        context: HarnessContext,
    ) -> dict[str, object]:
        previous_sleep = await self.summary_payload(
            run_id=context.run_id,
            summary_type="harness_sleep",
        )
        return {
            "run_created": context.run_created,
            "latest_harness_sleep": previous_sleep,
        }

    def _resolve_tenant(self, tenant: TenantContext | None) -> TenantContext:
        if tenant is not None:
            return tenant
        if self._active_context is not None:
            return self._active_context.tenant
        if self._tenant is not None:
            return self._tenant
        raise HarnessStateError(
            "TenantContext is required. Bind tenant in BaseHarness(..., tenant=...) "
            "or pass tenant=... to run()/helper calls."
        )

    def _resolve_run_id(self, *, run_id: str | None) -> str:
        if run_id is not None:
            return run_id
        if self._active_context is not None:
            return self._active_context.run_id
        raise HarnessStateError(
            "run_id is required outside an active harness session."
        )

    def _resolve_model(self, *, model: str | None) -> str:
        if model is not None:
            return model
        if self._active_context is not None:
            return self._active_context.model
        return self._default_model

    def _next_step_key(self, *, prefix: str) -> str:
        normalized = _normalize_step_key_prefix(prefix)
        next_index = self._step_key_counters.get(normalized, 0) + 1
        self._step_key_counters[normalized] = next_index
        return f"{self._step_key_namespace}_{normalized}_{next_index}"

    def _activate_context(self, *, context: HarnessContext) -> None:
        self._active_context = context
        self._step_key_counters = {}
        self._active_trace_previous_stage_name = None

    def _deactivate_context(self) -> None:
        self._active_context = None
        self._step_key_counters = {}
        self._active_trace_previous_stage_name = None


def _error_payload(exc: Exception | None) -> dict[str, str] | None:
    if exc is None:
        return None
    return {"type": type(exc).__name__, "message": str(exc)}


def _normalize_step_key_prefix(prefix: str) -> str:
    collapsed = re.sub(r"[^a-zA-Z0-9_]+", "_", prefix).strip("_")
    if collapsed == "":
        return "step"
    return collapsed.lower()


def _resolve_model_input(
    *,
    input: ModelInput | None,
    prompt: str | None,
    messages: Sequence[ChatMessage] | None,
) -> ModelInput:
    if input is not None:
        if prompt is not None or messages is not None:
            raise ValueError(
                "run_model accepts either input=... or prompt/messages, not both."
            )
        return input
    if prompt is not None and messages is not None:
        raise ValueError("Provide either prompt or messages, not both.")
    if prompt is not None:
        return ModelInput.from_prompt(prompt)
    if messages is not None:
        return ModelInput.from_messages(tuple(messages))
    raise ValueError("run_model requires one of: input, prompt, or messages.")


__all__ = ["BaseHarness", "HarnessContext", "HarnessStateError"]
