from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Literal, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel

from artana._kernel.model_cycle import (
    get_or_execute_model_step,
    tool_signatures_from_definitions,
)
from artana._kernel.policies import apply_prepare_model_middleware
from artana._kernel.replay import find_prompt_drift_candidate, validate_tenant_for_run
from artana._kernel.tool_cycle import (
    execute_tool_step_with_replay,
    reconcile_tool_with_replay,
)
from artana._kernel.types import (
    ApprovalRequiredError,
    ContextVersion,
    KernelPolicy,
    ModelInput,
    OutputT,
    PauseTicket,
    PolicyViolationError,
    ReplayPolicy,
    ResumePoint,
    RunHandle,
    RunLease,
    RunLifecycleStatus,
    RunRef,
    RunStatus,
    StepModelResult,
    StepToolResult,
    ToolCallable,
    ToolFingerprint,
)
from artana._kernel.workflow_runtime import (
    WorkflowContext,
    WorkflowRunResult,
    run_workflow,
)
from artana.events import (
    ChatMessage,
    EventPayload,
    EventType,
    HarnessFailedPayload,
    HarnessSleepPayload,
    HarnessStagePayload,
    KernelEvent,
    PauseRequestedPayload,
    ReplayedWithDriftPayload,
    ResumeRequestedPayload,
    RunStartedPayload,
    RunSummaryPayload,
    ToolCompletedPayload,
)
from artana.json_utils import canonical_json_dumps, canonicalize_json_object, sha256_hex
from artana.middleware import order_middleware
from artana.middleware.base import KernelMiddleware, ModelInvocation
from artana.middleware.capability_guard import CapabilityGuardMiddleware
from artana.middleware.pii_scrubber import PIIScrubberMiddleware
from artana.middleware.quota import QuotaMiddleware
from artana.middleware.safety_policy import SafetyPolicyMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelCallOptions, ModelPort, ToolDefinition
from artana.ports.model_adapter_helpers import serialize_messages_for_responses
from artana.ports.tool import LocalToolRegistry, ToolPort, ToolRiskLevel
from artana.safety import IntentPlanRecord
from artana.store.base import (
    EventStore,
    SupportsEventStreaming,
    SupportsModelCostAggregation,
    SupportsRunIndexing,
    SupportsRunLeasing,
)

WorkflowOutputT = TypeVar("WorkflowOutputT")


class _CriticDecision(BaseModel):
    approved: bool
    reason: str


@runtime_checkable
class _StoreBindableMiddleware(Protocol):
    def bind_store(self, store: EventStore) -> None:
        ...


class ArtanaKernel:
    def __init__(
        self,
        *,
        store: EventStore,
        model_port: ModelPort,
        tool_port: ToolPort | None = None,
        middleware: Sequence[KernelMiddleware] | None = None,
        policy: KernelPolicy | None = None,
    ) -> None:
        self._store = store
        self._model_port = model_port
        self._tool_port = tool_port if tool_port is not None else LocalToolRegistry()
        self._policy = policy if policy is not None else KernelPolicy()
        self._middleware = order_middleware(tuple(middleware or ()))
        self._validate_policy_requirements()
        for middleware_item in self._middleware:
            if isinstance(middleware_item, _StoreBindableMiddleware):
                middleware_item.bind_store(store)

    @staticmethod
    def default_middleware_stack(
        *,
        pii: bool = True,
        quota: bool = True,
        capabilities: bool = True,
        safety: SafetyPolicyMiddleware | None = None,
    ) -> tuple[KernelMiddleware, ...]:
        stack: list[KernelMiddleware] = []
        if pii:
            stack.append(PIIScrubberMiddleware())
        if quota:
            stack.append(QuotaMiddleware())
        if capabilities:
            stack.append(CapabilityGuardMiddleware())
        if safety is not None:
            stack.append(safety)
        return order_middleware(tuple(stack))

    async def _append_event(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
        parent_step_key: str | None = None,
    ) -> KernelEvent:
        if parent_step_key is None:
            return await self._store.append_event(
                run_id=run_id,
                tenant_id=tenant_id,
                event_type=event_type,
                payload=payload,
            )
        return await self._store.append_event(
            run_id=run_id,
            tenant_id=tenant_id,
            event_type=event_type,
            payload=payload,
            parent_step_key=parent_step_key,
        )

    async def start_run(
        self,
        *,
        tenant: TenantContext,
        run_id: str | None = None,
    ) -> RunRef:
        run_id_value = run_id
        if run_id_value is not None:
            existing = await self._store.get_events_for_run(run_id_value)
            if existing:
                raise ValueError(
                    f"run_id={run_id_value!r} already exists; provide a different run_id."
                )
        else:
            for _ in range(5):
                generated = uuid4().hex
                if not await self._store.get_events_for_run(generated):
                    run_id_value = generated
                    break
            if run_id_value is None:
                raise RuntimeError(
                    "Failed to allocate a unique run_id after multiple attempts."
                )

        event = await self._append_event(
            run_id=run_id_value,
            tenant_id=tenant.tenant_id,
            event_type=EventType.RUN_STARTED,
            payload=RunStartedPayload(),
        )
        return RunHandle(run_id=event.run_id, tenant_id=event.tenant_id)

    async def load_run(self, *, run_id: str) -> RunRef:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(f"No events found for run_id={run_id!r}.")
        return RunHandle(run_id=run_id, tenant_id=events[0].tenant_id)

    async def get_events(self, *, run_id: str) -> tuple[KernelEvent, ...]:
        return tuple(await self._store.get_events_for_run(run_id))

    async def get_latest_run_summary(
        self,
        *,
        run_id: str,
        summary_type: str,
    ) -> RunSummaryPayload | None:
        return await self._store.get_latest_run_summary(run_id, summary_type)

    async def get_latest_summary(
        self,
        *,
        run_id: str,
        summary_type: str,
    ) -> RunSummaryPayload | None:
        return await self.get_latest_run_summary(
            run_id=run_id,
            summary_type=summary_type,
        )

    async def explain_run(
        self,
        run_id: str,
    ) -> dict[str, object]:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(f"No events found for run_id={run_id!r}.")

        drift_count = 0
        last_stage: str | None = None
        last_tool: str | None = None
        failure_reason: str | None = None
        failure_step: str | None = None
        status = "completed"

        for event in events:
            if event.event_type == EventType.REPLAYED_WITH_DRIFT:
                drift_count += 1
            elif event.event_type == EventType.HARNESS_STAGE and isinstance(
                event.payload, HarnessStagePayload
            ):
                last_stage = event.payload.stage
            elif event.event_type == EventType.TOOL_COMPLETED and isinstance(
                event.payload, ToolCompletedPayload
            ):
                last_tool = event.payload.tool_name
            elif event.event_type == EventType.HARNESS_FAILED and isinstance(
                event.payload, HarnessFailedPayload
            ):
                failure_reason = event.payload.error_type
                failure_step = event.payload.last_step_key
                status = "failed"
            elif event.event_type == EventType.HARNESS_SLEEP and isinstance(
                event.payload, HarnessSleepPayload
            ):
                status = event.payload.status
            elif event.event_type == EventType.RUN_SUMMARY and isinstance(
                event.payload, RunSummaryPayload
            ):
                if last_stage is None and event.payload.summary_type == "trace::round":
                    try:
                        payload_json = json.loads(event.payload.summary_json)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload_json, dict):
                        stage_value = payload_json.get("stage")
                        if isinstance(stage_value, str):
                            last_stage = stage_value

        if status == "completed" and failure_reason is not None:
            status = "failed"

        if isinstance(self._store, SupportsModelCostAggregation):
            cost_total = await self._store.get_model_cost_sum_for_run(run_id)
        else:
            cost_total = 0.0
            for event in events:
                if event.event_type != EventType.MODEL_COMPLETED:
                    continue
                payload = event.payload
                if payload.kind != "model_completed":
                    raise RuntimeError(
                        "Invalid event payload kind "
                        f"{payload.kind!r} for model_completed event."
                    )
                cost_total += payload.cost_usd
        return {
            "status": status,
            "last_stage": last_stage,
            "last_tool": last_tool,
            "drift_count": drift_count,
            "drift_events": drift_count,
            "failure_reason": failure_reason,
            "failure_step": failure_step,
            "cost_total": cost_total,
        }

    async def get_run_status(self, *, run_id: str) -> RunStatus:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(f"No events found for run_id={run_id!r}.")
        status, blocked_on, failure_reason = _derive_run_status(events)
        last_event = events[-1]
        return RunStatus(
            run_id=run_id,
            tenant_id=events[0].tenant_id,
            status=status,
            last_event_seq=last_event.seq,
            last_event_type=last_event.event_type.value,
            updated_at=last_event.timestamp,
            blocked_on=blocked_on,
            failure_reason=failure_reason,
        )

    async def list_active_runs(
        self,
        *,
        tenant_id: str,
        status: RunLifecycleStatus | None = None,
        since: datetime | None = None,
    ) -> tuple[RunStatus, ...]:
        if not isinstance(self._store, SupportsRunIndexing):
            raise RuntimeError(
                "list_active_runs requires a store implementing SupportsRunIndexing."
            )
        run_ids = await self._store.list_run_ids(tenant_id=tenant_id, since=since)
        statuses: list[RunStatus] = []
        for run_id in run_ids:
            run_status = await self.get_run_status(run_id=run_id)
            if run_status.tenant_id != tenant_id:
                continue
            if since is not None and run_status.updated_at < since:
                continue
            if status is None:
                if run_status.status not in {"active", "paused"}:
                    continue
            elif run_status.status != status:
                continue
            statuses.append(run_status)
        return tuple(statuses)

    async def resume_point(self, *, run_id: str) -> ResumePoint:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(f"No events found for run_id={run_id!r}.")
        _, blocked_on, _ = _derive_run_status(events)
        return ResumePoint(
            run_id=run_id,
            last_event_seq=events[-1].seq,
            last_step_key=_latest_step_key(events),
            blocked_on=blocked_on,
        )

    async def checkpoint(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        name: str,
        payload: object,
        step_key: str | None = None,
        parent_step_key: str | None = None,
    ) -> int:
        if name.strip() == "":
            raise ValueError("Checkpoint name must be non-empty.")
        return await self.append_run_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type=f"checkpoint::{name}",
            summary_json=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            step_key=step_key,
            parent_step_key=parent_step_key,
        )

    async def set_artifact(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        key: str,
        value: object,
        schema_version: str | None = None,
        step_key: str | None = None,
        parent_step_key: str | None = None,
    ) -> int:
        if key.strip() == "":
            raise ValueError("Artifact key must be non-empty.")
        payload: dict[str, object] = {"value": value}
        if schema_version is not None:
            payload["schema_version"] = schema_version
        return await self.append_run_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type=f"artifact::{key}",
            summary_json=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            step_key=step_key,
            parent_step_key=parent_step_key,
        )

    async def get_artifact(
        self,
        *,
        run_id: str,
        key: str,
    ) -> object | None:
        summary = await self.get_latest_run_summary(
            run_id=run_id,
            summary_type=f"artifact::{key}",
        )
        if summary is None:
            return None
        try:
            payload: object = json.loads(summary.summary_json)
        except json.JSONDecodeError:
            return summary.summary_json
        if isinstance(payload, dict) and "value" in payload:
            return payload.get("value")
        return payload

    async def list_artifacts(self, *, run_id: str) -> dict[str, object]:
        events = await self._store.get_events_for_run(run_id)
        latest_by_key: dict[str, RunSummaryPayload] = {}
        for event in events:
            if event.event_type != EventType.RUN_SUMMARY:
                continue
            payload = event.payload
            if not isinstance(payload, RunSummaryPayload):
                continue
            if not payload.summary_type.startswith("artifact::"):
                continue
            artifact_key = payload.summary_type.split("artifact::", 1)[1]
            latest_by_key[artifact_key] = payload
        resolved: dict[str, object] = {}
        for key, payload in latest_by_key.items():
            try:
                parsed: object = json.loads(payload.summary_json)
            except json.JSONDecodeError:
                resolved[key] = payload.summary_json
                continue
            if isinstance(parsed, dict) and "value" in parsed:
                resolved[key] = parsed.get("value")
            else:
                resolved[key] = parsed
        return resolved

    async def block_run(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        reason: str,
        unblock_key: str | None = None,
        metadata: Mapping[str, object] | None = None,
        step_key: str | None = None,
        parent_step_key: str | None = None,
    ) -> PauseTicket:
        context = _BlockRunContext(
            unblock_key=unblock_key,
            metadata_json=(
                json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)
                if metadata is not None
                else None
            ),
        )
        return await self.pause(
            run_id=run_id,
            tenant=tenant,
            reason=reason,
            context=context,
            step_key=step_key,
            parent_step_key=parent_step_key,
        )

    async def unblock_run(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        unblock_key: str | None = None,
        metadata: Mapping[str, object] | None = None,
        parent_step_key: str | None = None,
    ) -> RunRef:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(f"No events found for run_id={run_id!r}.")
        validate_tenant_for_run(events=events, tenant=tenant)
        pending_pause = _pending_pause_event(events)
        if pending_pause is None:
            raise ValueError(f"Run {run_id!r} is not currently blocked or paused.")
        if unblock_key is not None and not _pause_matches_unblock_key(
            pending_pause=pending_pause,
            unblock_key=unblock_key,
        ):
            raise ValueError(
                f"Run {run_id!r} pause context does not match unblock_key={unblock_key!r}."
            )
        if unblock_key is not None or metadata is not None:
            await self.append_run_summary(
                run_id=run_id,
                tenant=tenant,
                summary_type=(
                    f"run_unblocked::{unblock_key}"
                    if unblock_key is not None
                    else "run_unblocked"
                ),
                summary_json=json.dumps(
                    {
                        "unblock_key": unblock_key,
                        "metadata": dict(metadata) if metadata is not None else None,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                step_key=(
                    f"unblock_{unblock_key}" if unblock_key is not None else "run_unblocked"
                ),
                parent_step_key=parent_step_key,
            )
        resume_payload = (
            _UnblockRunInput(
                unblock_key=unblock_key,
                metadata_json=(
                    json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)
                    if metadata is not None
                    else None
                ),
            )
            if unblock_key is not None or metadata is not None
            else None
        )
        return await self.resume(
            run_id=run_id,
            tenant=tenant,
            human_input=resume_payload,
            parent_step_key=parent_step_key,
        )

    async def explain_tool_allowlist(
        self,
        *,
        tenant: TenantContext,
        model: str = "tool_allowlist_explain",
        run_id: str = "__tool_allowlist_explain__",
        visible_tool_names: set[str] | None = None,
        prompt: str = "Explain tool allowlist for this tenant.",
        messages: Sequence[ChatMessage] | None = None,
    ) -> dict[str, object]:
        resolved_messages: tuple[ChatMessage, ...]
        if messages is None:
            resolved_messages = (ChatMessage(role="user", content=prompt),)
        else:
            resolved_messages = tuple(messages)
            if len(resolved_messages) == 0:
                resolved_messages = (ChatMessage(role="user", content=prompt),)
        resolved_prompt = (
            prompt if prompt.strip() != "" else _derive_prompt_from_messages(resolved_messages)
        )
        registered_tools = tuple(self._tool_port.to_all_tool_definitions())
        capability_map = self._tool_port.capability_map()
        visible_filter = set(visible_tool_names) if visible_tool_names is not None else None
        all_tools = registered_tools
        if visible_filter is not None:
            all_tools = tuple(tool for tool in all_tools if tool.name in visible_filter)
            capability_map = {
                name: capability
                for name, capability in capability_map.items()
                if name in visible_filter
            }
        initial_invocation = ModelInvocation(
            run_id=run_id,
            tenant=tenant,
            model=model,
            prompt=resolved_prompt,
            messages=resolved_messages,
            model_options=ModelCallOptions(),
            allowed_tools=all_tools,
            tool_capability_by_name=capability_map,
        )
        prepared_invocation = await apply_prepare_model_middleware(
            self._middleware,
            initial_invocation,
        )
        return _build_capability_decision_summary(
            tenant=tenant,
            model=prepared_invocation.model,
            step_key=None,
            registered_tools=registered_tools,
            capability_map=self._tool_port.capability_map(),
            visible_tool_names=visible_filter,
            middleware_allowed_tools=prepared_invocation.allowed_tools,
        )

    async def describe_capabilities(
        self,
        *,
        tenant: TenantContext,
        visible_tool_names: set[str] | None = None,
    ) -> dict[str, object]:
        return await self.explain_tool_allowlist(
            tenant=tenant,
            visible_tool_names=visible_tool_names,
        )

    def list_tools_for_tenant(
        self,
        *,
        tenant: TenantContext,
        visible_tool_names: set[str] | None = None,
    ) -> tuple[ToolDefinition, ...]:
        return self.list_tools(
            tenant_capabilities=tenant.capabilities,
            visible_tool_names=visible_tool_names,
        )

    async def stream_events(
        self,
        *,
        run_id: str,
        since_seq: int = 0,
        follow: bool = False,
        poll_interval_seconds: float = 0.5,
        idle_timeout_seconds: float | None = None,
    ) -> AsyncIterator[KernelEvent]:
        if isinstance(self._store, SupportsEventStreaming):
            async for event in self._store.stream_events(
                run_id,
                since_seq=since_seq,
                follow=follow,
                poll_interval_seconds=poll_interval_seconds,
                idle_timeout_seconds=idle_timeout_seconds,
            ):
                yield event
            return
        events = await self._store.get_events_for_run(run_id)
        for event in events:
            if event.seq <= since_seq:
                continue
            yield event

    async def acquire_run_lease(
        self,
        *,
        run_id: str,
        worker_id: str,
        ttl_seconds: int,
    ) -> bool:
        if not isinstance(self._store, SupportsRunLeasing):
            raise RuntimeError(
                "acquire_run_lease requires a store implementing SupportsRunLeasing."
            )
        return await self._store.acquire_run_lease(
            run_id=run_id,
            worker_id=worker_id,
            ttl_seconds=ttl_seconds,
        )

    async def renew_run_lease(
        self,
        *,
        run_id: str,
        worker_id: str,
        ttl_seconds: int,
    ) -> bool:
        if not isinstance(self._store, SupportsRunLeasing):
            raise RuntimeError(
                "renew_run_lease requires a store implementing SupportsRunLeasing."
            )
        return await self._store.renew_run_lease(
            run_id=run_id,
            worker_id=worker_id,
            ttl_seconds=ttl_seconds,
        )

    async def release_run_lease(
        self,
        *,
        run_id: str,
        worker_id: str,
    ) -> bool:
        if not isinstance(self._store, SupportsRunLeasing):
            raise RuntimeError(
                "release_run_lease requires a store implementing SupportsRunLeasing."
            )
        return await self._store.release_run_lease(run_id=run_id, worker_id=worker_id)

    async def get_run_lease(self, *, run_id: str) -> RunLease | None:
        if not isinstance(self._store, SupportsRunLeasing):
            raise RuntimeError(
                "get_run_lease requires a store implementing SupportsRunLeasing."
            )
        lease = await self._store.get_run_lease(run_id=run_id)
        if lease is None:
            return None
        return RunLease(
            run_id=lease.run_id,
            worker_id=lease.worker_id,
            lease_expires_at=lease.lease_expires_at,
        )

    def canonicalize_tool_args(
        self,
        *,
        tool_name: str,
        arguments: BaseModel | Mapping[str, object] | str,
    ) -> tuple[str, str]:
        tool_by_name = {tool.name: tool for tool in self.list_registered_tools()}
        tool = tool_by_name.get(tool_name)
        if tool is None:
            raise KeyError(f"Tool {tool_name!r} is not registered.")
        if isinstance(arguments, str):
            canonical_arguments_json = canonicalize_json_object(arguments)
        elif isinstance(arguments, BaseModel):
            canonical_arguments_json = canonical_json_dumps(arguments.model_dump(mode="json"))
        else:
            canonical_arguments_json = canonical_json_dumps(dict(arguments))
            canonical_arguments_json = canonicalize_json_object(canonical_arguments_json)
        return canonical_arguments_json, tool.schema_hash

    def tool_fingerprint(self, *, tool_name: str) -> ToolFingerprint:
        tool_by_name = {tool.name: tool for tool in self.list_registered_tools()}
        tool = tool_by_name.get(tool_name)
        if tool is None:
            raise KeyError(f"Tool {tool_name!r} is not registered.")
        return ToolFingerprint(
            tool_name=tool.name,
            tool_version=tool.tool_version,
            schema_version=tool.schema_version,
            schema_hash=tool.schema_hash,
            risk_level=_normalize_risk_level(tool.risk_level),
            sandbox_profile=tool.sandbox_profile,
        )

    def list_registered_tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tool_port.to_all_tool_definitions())

    def tool_capability_map(self) -> dict[str, str | None]:
        return self._tool_port.capability_map()

    def list_tools(
        self,
        *,
        tenant_capabilities: frozenset[str],
        visible_tool_names: set[str] | None = None,
    ) -> tuple[ToolDefinition, ...]:
        allowed = tuple(self._tool_port.to_tool_definitions(tenant_capabilities))
        if visible_tool_names is None:
            return allowed
        return tuple(tool for tool in allowed if tool.name in visible_tool_names)

    def tool(
        self,
        *,
        requires_capability: str | None = None,
        side_effect: bool = False,
        tool_version: str = "1.0.0",
        schema_version: str = "1",
        risk_level: ToolRiskLevel = "medium",
        sandbox_profile: str | None = None,
    ) -> Callable[[ToolCallable], ToolCallable]:
        def decorator(function: ToolCallable) -> ToolCallable:
            self._tool_port.register(
                function=function,
                requires_capability=requires_capability,
                side_effect=side_effect,
                tool_version=tool_version,
                schema_version=schema_version,
                risk_level=risk_level,
                sandbox_profile=sandbox_profile,
            )
            return function

        return decorator

    async def pause(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        reason: str,
        context: BaseModel | None = None,
        step_key: str | None = None,
        parent_step_key: str | None = None,
    ) -> PauseTicket:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(
                f"Cannot pause unknown run_id={run_id!r}; call start_run first."
            )
        validate_tenant_for_run(events=events, tenant=tenant)
        context_json = context.model_dump_json() if context is not None else None
        event = await self._append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.PAUSE_REQUESTED,
            parent_step_key=parent_step_key,
            payload=PauseRequestedPayload(
                reason=reason,
                context_json=context_json,
                step_key=step_key,
            ),
        )
        return PauseTicket(
            run_id=event.run_id,
            ticket_id=event.event_id,
            seq=event.seq,
            reason=reason,
        )

    async def step_model_with_visible_tools(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        input: ModelInput,
        output_schema: type[OutputT],
        visible_tool_names: set[str] | None,
        model_options: ModelCallOptions | None = None,
        step_key: str | None = None,
        replay_policy: ReplayPolicy = "strict",
        context_version: ContextVersion | None = None,
        parent_step_key: str | None = None,
    ) -> StepModelResult[OutputT]:
        return await self.step_model(
            run_id=run_id,
            tenant=tenant,
            model=model,
            input=input,
            output_schema=output_schema,
            step_key=step_key,
            visible_tool_names=visible_tool_names,
            model_options=model_options,
            replay_policy=replay_policy,
            context_version=context_version,
            parent_step_key=parent_step_key,
        )

    async def step_model(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        input: ModelInput,
        output_schema: type[OutputT],
        step_key: str | None = None,
        visible_tool_names: set[str] | None = None,
        model_options: ModelCallOptions | None = None,
        replay_policy: ReplayPolicy = "strict",
        context_version: ContextVersion | None = None,
        parent_step_key: str | None = None,
    ) -> StepModelResult[OutputT]:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(
                f"No events found for run_id={run_id!r}; call start_run first."
            )
        validate_tenant_for_run(events=events, tenant=tenant)

        prompt, messages = _normalize_model_input(input)
        resolved_model_options = model_options or ModelCallOptions()
        registered_tools = tuple(self._tool_port.to_all_tool_definitions())
        capability_map = self._tool_port.capability_map()
        all_tools = registered_tools
        visible_filter = set(visible_tool_names) if visible_tool_names is not None else None
        if visible_filter is not None:
            all_tools = tuple(tool for tool in all_tools if tool.name in visible_filter)
            capability_map = {
                name: capability
                for name, capability in capability_map.items()
                if name in visible_filter
            }
        initial_invocation = ModelInvocation(
            run_id=run_id,
            tenant=tenant,
            model=model,
            prompt=prompt,
            messages=messages,
            model_options=resolved_model_options,
            allowed_tools=all_tools,
            tool_capability_by_name=capability_map,
        )
        prepared_invocation = await apply_prepare_model_middleware(
            self._middleware,
            initial_invocation,
        )
        responses_input_items = (
            serialize_messages_for_responses(
                prepared_invocation.messages,
                fallback_prompt=prepared_invocation.prompt,
            )
            if prepared_invocation.model_options.api_mode != "chat"
            else None
        )
        capability_decision_summary = _build_capability_decision_summary(
            tenant=tenant,
            model=prepared_invocation.model,
            step_key=step_key,
            registered_tools=registered_tools,
            capability_map=self._tool_port.capability_map(),
            visible_tool_names=visible_filter,
            middleware_allowed_tools=prepared_invocation.allowed_tools,
        )
        target_run_id = run_id
        target_events = events
        forked_from_run_id: str | None = None
        normalized_replay_policy = replay_policy
        drift_fields: tuple[str, ...] = ()
        if replay_policy == "fork_on_drift":
            drift_candidate = find_prompt_drift_candidate(
                events=events,
                prompt=prepared_invocation.prompt,
                messages=prepared_invocation.messages,
                model=prepared_invocation.model,
                model_options=prepared_invocation.model_options,
                responses_input_items=responses_input_items,
                allowed_tool_signatures=tool_signatures_from_definitions(
                    prepared_invocation.allowed_tools
                ),
                step_key=step_key,
            )
            if drift_candidate is not None:
                target_run_id = _derive_fork_run_id(
                    run_id=run_id,
                    model=prepared_invocation.model,
                    step_key=step_key,
                    prompt=prepared_invocation.prompt,
                    messages=prepared_invocation.messages,
                    model_options=prepared_invocation.model_options,
                    responses_input_items=responses_input_items,
                )
                target_events = await self._ensure_fork_run_exists(
                    run_id=target_run_id,
                    tenant=tenant,
                )
                forked_from_run_id = run_id
                await self._append_event(
                    run_id=run_id,
                    tenant_id=tenant.tenant_id,
                    event_type=EventType.REPLAYED_WITH_DRIFT,
                    parent_step_key=parent_step_key,
                    payload=ReplayedWithDriftPayload(
                        step_key=step_key,
                        model=prepared_invocation.model,
                        drift_fields=list(drift_candidate.drift_fields),
                        source_model_requested_event_id=drift_candidate.request_event.event_id,
                        source_model_completed_seq=(
                            drift_candidate.completed_event.seq
                            if drift_candidate.completed_event is not None
                            else None
                        ),
                        replay_policy="fork_on_drift",
                        fork_run_id=target_run_id,
                    ),
                )
                drift_fields = drift_candidate.drift_fields
            normalized_replay_policy = "strict"

        model_result = await get_or_execute_model_step(
            store=self._store,
            model_port=self._model_port,
            middleware=self._middleware,
            run_id=target_run_id,
            prompt=prepared_invocation.prompt,
            messages=prepared_invocation.messages,
            model=prepared_invocation.model,
            tenant=tenant,
            output_schema=output_schema,
            tool_definitions=prepared_invocation.allowed_tools,
            events=target_events,
            model_options=prepared_invocation.model_options,
            responses_input_items=responses_input_items,
            step_key=step_key,
            parent_step_key=parent_step_key,
            replay_policy=normalized_replay_policy,
            context_version=context_version,
        )
        if model_result.replayed is False:
            await self._append_event(
                run_id=target_run_id,
                tenant_id=tenant.tenant_id,
                event_type=EventType.RUN_SUMMARY,
                parent_step_key=parent_step_key,
                payload=RunSummaryPayload(
                    summary_type="capability_decision",
                    summary_json=canonical_json_dumps(capability_decision_summary),
                    step_key=step_key,
                ),
            )
        return StepModelResult(
            run_id=target_run_id,
            seq=model_result.completed_seq,
            output=model_result.output,
            usage=model_result.usage,
            tool_calls=model_result.tool_calls,
            replayed=model_result.replayed,
            api_mode_used=model_result.api_mode_used,
            response_id=model_result.response_id,
            response_output_items=model_result.response_output_items,
            replayed_with_drift=model_result.replayed_with_drift,
            forked_from_run_id=forked_from_run_id,
            drift_fields=drift_fields or model_result.drift_fields,
        )

    async def step_tool(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments: BaseModel,
        step_key: str | None = None,
        parent_step_key: str | None = None,
    ) -> StepToolResult:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(
                f"No events found for run_id={run_id!r}; call start_run first."
            )
        validate_tenant_for_run(events=events, tenant=tenant)
        arguments_json = canonical_json_dumps(arguments.model_dump(mode="json"))
        critic_attempted = False
        while True:
            try:
                result = await execute_tool_step_with_replay(
                    store=self._store,
                    tool_port=self._tool_port,
                    middleware=self._middleware,
                    run_id=run_id,
                    tenant=tenant,
                    tool_name=tool_name,
                    arguments_json=arguments_json,
                    step_key=step_key,
                    parent_step_key=parent_step_key,
                )
                return StepToolResult(
                    run_id=run_id,
                    seq=result.seq,
                    tool_name=tool_name,
                    result_json=result.result_json,
                    replayed=result.replayed,
                )
            except ApprovalRequiredError as exc:
                if exc.mode == "human":
                    await self._pause_for_approval_if_needed(
                        run_id=run_id,
                        tenant=tenant,
                        tool_name=tool_name,
                        approval_key=exc.approval_key,
                        parent_step_key=parent_step_key,
                    )
                    raise
                if exc.mode != "critic":
                    raise
                if critic_attempted:
                    raise RuntimeError(
                        "Critic approval loop detected: middleware still requests critic "
                        "approval after a critic decision."
                    )
                critic_attempted = True
                decision = await self._run_critic_gate(
                    run_id=run_id,
                    tenant=tenant,
                    tool_name=tool_name,
                    arguments_json=(
                        exc.arguments_json
                        if exc.arguments_json is not None
                        else canonicalize_json_object(arguments_json)
                    ),
                    approval_key=exc.approval_key,
                    critic_model=exc.critic_model,
                    fingerprint=exc.fingerprint,
                    parent_step_key=parent_step_key,
                )
                if not decision.approved:
                    raise PolicyViolationError(
                        code="critic_denied",
                        message=(
                            f"Critic gate denied tool {tool_name!r}: "
                            f"{decision.reason}"
                        ),
                        tool_name=tool_name,
                        fingerprint=exc.fingerprint,
                    )
                await self.approve_tool_call(
                    run_id=run_id,
                    tenant=tenant,
                    approval_key=exc.approval_key,
                    mode="critic",
                    reason=decision.reason,
                    step_key=(
                        f"approval_critic_{tool_name}"
                        if step_key is None
                        else f"{step_key}_approval_critic"
                    ),
                    parent_step_key=parent_step_key,
                )

    async def reconcile_tool(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments: BaseModel,
        step_key: str | None = None,
        parent_step_key: str | None = None,
    ) -> str:
        arguments_json = canonical_json_dumps(arguments.model_dump(mode="json"))
        return await reconcile_tool_with_replay(
            store=self._store,
            tool_port=self._tool_port,
            middleware=self._middleware,
            run_id=run_id,
            tenant=tenant,
            tool_name=tool_name,
            arguments_json=arguments_json,
            step_key=step_key,
            parent_step_key=parent_step_key,
        )

    async def resume(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        human_input: BaseModel | None = None,
        parent_step_key: str | None = None,
    ) -> RunRef:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(f"No events found for run_id={run_id!r}.")
        validate_tenant_for_run(events=events, tenant=tenant)
        human_input_json = human_input.model_dump_json() if human_input is not None else None
        event = await self._append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.RESUME_REQUESTED,
            parent_step_key=parent_step_key,
            payload=ResumeRequestedPayload(human_input_json=human_input_json),
        )
        return RunHandle(run_id=event.run_id, tenant_id=event.tenant_id)

    async def append_run_summary(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        summary_type: str,
        summary_json: str,
        step_key: str | None = None,
        parent_step_key: str | None = None,
    ) -> int:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(
                f"No events found for run_id={run_id!r}; call start_run first."
            )
        validate_tenant_for_run(events=events, tenant=tenant)
        event = await self._append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.RUN_SUMMARY,
            parent_step_key=parent_step_key,
            payload=RunSummaryPayload(
                summary_type=summary_type,
                summary_json=summary_json,
                step_key=step_key,
            ),
        )
        return event.seq

    async def record_intent_plan(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        intent: IntentPlanRecord,
        step_key: str | None = None,
        parent_step_key: str | None = None,
    ) -> int:
        return await self.append_run_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type="policy::intent_plan",
            summary_json=intent.model_dump_json(),
            step_key=step_key,
            parent_step_key=parent_step_key,
        )

    async def approve_tool_call(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        approval_key: str,
        mode: str,
        reason: str,
        step_key: str | None = None,
        parent_step_key: str | None = None,
    ) -> int:
        return await self.append_run_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type=f"policy::approval::{approval_key}",
            summary_json=json.dumps(
                {
                    "approved": True,
                    "mode": mode,
                    "reason": reason,
                    "approved_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            step_key=step_key,
            parent_step_key=parent_step_key,
        )

    async def append_harness_event(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        event_type: EventType,
        payload: EventPayload,
        parent_step_key: str | None = None,
    ) -> KernelEvent:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(
                f"No events found for run_id={run_id!r}; call start_run first."
            )
        validate_tenant_for_run(events=events, tenant=tenant)
        return await self._append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=event_type,
            parent_step_key=parent_step_key,
            payload=payload,
        )

    async def close(self) -> None:
        await self._store.close()

    async def run_workflow(
        self,
        *,
        run_id: str | None,
        tenant: TenantContext,
        workflow: Callable[[WorkflowContext], Awaitable[WorkflowOutputT]],
    ) -> WorkflowRunResult[WorkflowOutputT]:
        return await run_workflow(
            store=self._store,
            pause_api=self,
            run_id=run_id,
            tenant=tenant,
            workflow=workflow,
        )

    async def _ensure_fork_run_exists(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
    ) -> list[KernelEvent]:
        events = await self._store.get_events_for_run(run_id)
        if events:
            validate_tenant_for_run(events=events, tenant=tenant)
            return events
        try:
            await self.start_run(tenant=tenant, run_id=run_id)
        except ValueError:
            events = await self._store.get_events_for_run(run_id)
            if not events:
                raise
            validate_tenant_for_run(events=events, tenant=tenant)
            return events
        return await self._store.get_events_for_run(run_id)

    def _validate_policy_requirements(self) -> None:
        if self._policy.mode not in {"enforced", "enforced_v2"}:
            return

        required: tuple[type[KernelMiddleware], ...] = (
            PIIScrubberMiddleware,
            QuotaMiddleware,
            CapabilityGuardMiddleware,
        )
        if self._policy.mode == "enforced_v2":
            required = required + (SafetyPolicyMiddleware,)
        for middleware_type in required:
            if not any(
                isinstance(middleware_item, middleware_type)
                for middleware_item in self._middleware
            ):
                raise ValueError(
                    f"KernelPolicy(mode={self._policy.mode!r}) requires middleware "
                    f"{middleware_type.__name__}."
                )
        if not any(
            hasattr(middleware_item, "prepare_tool_request")
            and hasattr(middleware_item, "prepare_tool_result")
            for middleware_item in self._middleware
        ):
            raise ValueError(
                f"KernelPolicy(mode={self._policy.mode!r}) requires tool IO policy "
                "middleware hooks."
            )

    async def _pause_for_approval_if_needed(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        approval_key: str,
        parent_step_key: str | None,
    ) -> None:
        events = await self._store.get_events_for_run(run_id)
        validate_tenant_for_run(events=events, tenant=tenant)
        for event in reversed(events):
            if event.event_type != EventType.PAUSE_REQUESTED:
                continue
            payload = event.payload
            if not isinstance(payload, PauseRequestedPayload):
                continue
            if payload.context_json is None:
                continue
            try:
                context_payload: object = json.loads(payload.context_json)
            except json.JSONDecodeError:
                continue
            if not isinstance(context_payload, dict):
                continue
            if context_payload.get("approval_key") != approval_key:
                continue
            return
        await self.pause(
            run_id=run_id,
            tenant=tenant,
            reason=f"Approval required for tool {tool_name!r}.",
            context=_ApprovalPauseContext(
                approval_key=approval_key,
                tool_name=tool_name,
            ),
            step_key=f"approval_human_{tool_name}",
            parent_step_key=parent_step_key,
        )

    async def _run_critic_gate(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments_json: str,
        approval_key: str,
        critic_model: str | None,
        fingerprint: str | None,
        parent_step_key: str | None,
    ) -> _CriticDecision:
        if critic_model is None:
            raise RuntimeError("Critic approval requires critic_model.")
        messages = (
            ChatMessage(
                role="system",
                content=(
                    "You are a strict safety critic. Approve only if the requested tool "
                    "execution is safe and policy-compliant."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"tool_name={tool_name}\n"
                    f"approval_key={approval_key}\n"
                    f"fingerprint={fingerprint}\n"
                    f"arguments_json={arguments_json}\n"
                    "Return JSON with keys approved (bool) and reason (string)."
                ),
            ),
        )
        result = await self.step_model(
            run_id=run_id,
            tenant=tenant,
            model=critic_model,
            input=ModelInput.from_messages(messages),
            output_schema=_CriticDecision,
            step_key=f"critic::{tool_name}::{approval_key[:12]}",
            visible_tool_names=set(),
            replay_policy="strict",
            parent_step_key=parent_step_key,
        )
        return result.output


class _ApprovalPauseContext(BaseModel):
    approval_key: str
    tool_name: str


def _normalize_model_input(model_input: ModelInput) -> tuple[str, tuple[ChatMessage, ...]]:
    if model_input.kind == "prompt":
        if model_input.prompt is None:
            raise ValueError("ModelInput(kind='prompt') requires prompt.")
        if model_input.messages is None:
            return model_input.prompt, (ChatMessage(role="user", content=model_input.prompt),)
        if len(model_input.messages) == 0:
            raise ValueError("ModelInput(kind='prompt') messages cannot be empty.")
        return model_input.prompt, model_input.messages

    if model_input.messages is None or len(model_input.messages) == 0:
        raise ValueError("ModelInput(kind='messages') requires non-empty messages.")

    prompt = model_input.prompt
    if prompt is None:
        prompt = _derive_prompt_from_messages(model_input.messages)
    return prompt, model_input.messages


def _derive_fork_run_id(
    *,
    run_id: str,
    model: str,
    step_key: str | None,
    prompt: str,
    messages: tuple[ChatMessage, ...],
    model_options: ModelCallOptions,
    responses_input_items: list[dict[str, object]] | None,
) -> str:
    messages_json = canonical_json_dumps(
        [message.model_dump(mode="json") for message in messages]
    )
    model_options_json = canonical_json_dumps(
        {
            "api_mode": model_options.api_mode,
            "reasoning_effort": model_options.reasoning_effort,
            "verbosity": model_options.verbosity,
            "previous_response_id": model_options.previous_response_id,
        }
    )
    responses_items_json = (
        canonical_json_dumps(responses_input_items)
        if responses_input_items is not None
        else "null"
    )
    token = (
        f"{model}|{step_key}|{prompt}|{messages_json}|"
        f"{model_options_json}|{responses_items_json}"
    )
    return f"{run_id}::fork::{sha256_hex(token)[:12]}"


def _derive_prompt_from_messages(messages: tuple[ChatMessage, ...]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def _build_capability_decision_summary(
    *,
    tenant: TenantContext,
    model: str,
    step_key: str | None,
    registered_tools: tuple[ToolDefinition, ...],
    capability_map: dict[str, str | None],
    visible_tool_names: set[str] | None,
    middleware_allowed_tools: tuple[ToolDefinition, ...],
) -> dict[str, object]:
    final_allowed_names = {tool.name for tool in middleware_allowed_tools}
    decisions: list[dict[str, object]] = []
    for tool in sorted(registered_tools, key=lambda item: item.name):
        required_capability = capability_map.get(tool.name)
        if visible_tool_names is not None and tool.name not in visible_tool_names:
            decision = "filtered"
            reason = "filtered_by_visible_tool_names"
        elif required_capability is None:
            if tool.name in capability_map:
                decision = "allowed"
                reason = "allowed_no_capability_required"
            else:
                decision = "filtered"
                reason = "filtered_not_registered"
        elif required_capability in tenant.capabilities:
            decision = "allowed"
            reason = "allowed_tenant_has_capability"
        else:
            decision = "filtered"
            reason = "filtered_missing_capability"

        if decision == "allowed" and tool.name not in final_allowed_names:
            decision = "filtered"
            reason = "filtered_by_model_middleware"
        elif decision == "filtered" and tool.name in final_allowed_names:
            decision = "allowed"
            reason = "allowed_by_model_middleware"

        decisions.append(
            {
                "tool_name": tool.name,
                "required_capability": required_capability,
                "decision": decision,
                "reason": reason,
            }
        )
    return {
        "model": model,
        "step_key": step_key,
        "tenant_capabilities": sorted(tenant.capabilities),
        "visible_tool_names_applied": visible_tool_names is not None,
        "final_allowed_tools": sorted(final_allowed_names),
        "decisions": decisions,
    }


class _BlockRunContext(BaseModel):
    unblock_key: str | None = None
    metadata_json: str | None = None


class _UnblockRunInput(BaseModel):
    unblock_key: str | None = None
    metadata_json: str | None = None


def _pending_pause_event(events: Sequence[KernelEvent]) -> KernelEvent | None:
    pending_pause: KernelEvent | None = None
    for event in events:
        if event.event_type == EventType.PAUSE_REQUESTED:
            pending_pause = event
            continue
        if event.event_type == EventType.RESUME_REQUESTED and pending_pause is not None:
            pending_pause = None
    return pending_pause


def _pause_matches_unblock_key(*, pending_pause: KernelEvent, unblock_key: str) -> bool:
    payload = pending_pause.payload
    if not isinstance(payload, PauseRequestedPayload):
        return False
    if payload.context_json is None:
        return False
    try:
        context_payload: object = json.loads(payload.context_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(context_payload, dict):
        return False
    context_unblock_key = context_payload.get("unblock_key")
    return isinstance(context_unblock_key, str) and context_unblock_key == unblock_key


def _pause_blocked_on(payload: PauseRequestedPayload) -> str | None:
    if payload.context_json is None:
        return None
    try:
        context_payload: object = json.loads(payload.context_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(context_payload, dict):
        return None
    approval_key = context_payload.get("approval_key")
    if isinstance(approval_key, str) and approval_key != "":
        return f"approval:{approval_key}"
    unblock_key = context_payload.get("unblock_key")
    if isinstance(unblock_key, str) and unblock_key != "":
        return f"unblock:{unblock_key}"
    return None


def _derive_run_status(
    events: Sequence[KernelEvent],
) -> tuple[RunLifecycleStatus, str | None, str | None]:
    pending_pause = _pending_pause_event(events)
    if pending_pause is not None:
        payload = pending_pause.payload
        if isinstance(payload, PauseRequestedPayload):
            return "paused", _pause_blocked_on(payload), None
        return "paused", None, None

    for event in reversed(events):
        if event.event_type != EventType.HARNESS_SLEEP:
            continue
        payload = event.payload
        if not isinstance(payload, HarnessSleepPayload):
            continue
        if payload.status == "failed":
            return "failed", None, payload.execution_error_type or payload.sleep_error_type
        return "completed", None, None

    for event in reversed(events):
        if event.event_type != EventType.HARNESS_FAILED:
            continue
        payload = event.payload
        if not isinstance(payload, HarnessFailedPayload):
            continue
        return "failed", None, payload.error_type

    return "active", None, None


def _latest_step_key(events: Sequence[KernelEvent]) -> str | None:
    for event in reversed(events):
        payload_step_key = getattr(event.payload, "step_key", None)
        if isinstance(payload_step_key, str):
            return payload_step_key
        if isinstance(event.parent_step_key, str):
            return event.parent_step_key
    return None


def _normalize_risk_level(value: str) -> Literal["low", "medium", "high", "critical"]:
    if value == "low":
        return "low"
    if value == "high":
        return "high"
    if value == "critical":
        return "critical"
    return "medium"
