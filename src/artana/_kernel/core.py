from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeVar, runtime_checkable
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
    ContextVersion,
    KernelPolicy,
    ModelInput,
    OutputT,
    PauseTicket,
    ReplayPolicy,
    RunHandle,
    RunRef,
    StepModelResult,
    StepToolResult,
    ToolCallable,
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
from artana.json_utils import canonical_json_dumps, sha256_hex
from artana.middleware import order_middleware
from artana.middleware.base import KernelMiddleware, ModelInvocation
from artana.middleware.capability_guard import CapabilityGuardMiddleware
from artana.middleware.pii_scrubber import PIIScrubberMiddleware
from artana.middleware.quota import QuotaMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelPort, ToolDefinition
from artana.ports.tool import LocalToolRegistry, ToolPort
from artana.store.base import EventStore, SupportsModelCostAggregation

WorkflowOutputT = TypeVar("WorkflowOutputT")


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
    ) -> tuple[KernelMiddleware, ...]:
        stack: list[KernelMiddleware] = []
        if pii:
            stack.append(PIIScrubberMiddleware())
        if quota:
            stack.append(QuotaMiddleware())
        if capabilities:
            stack.append(CapabilityGuardMiddleware())
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
        tool_version: str = "1.0.0",
        schema_version: str = "1",
    ) -> Callable[[ToolCallable], ToolCallable]:
        def decorator(function: ToolCallable) -> ToolCallable:
            self._tool_port.register(
                function=function,
                requires_capability=requires_capability,
                tool_version=tool_version,
                schema_version=schema_version,
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
            allowed_tools=all_tools,
            tool_capability_by_name=capability_map,
        )
        prepared_invocation = await apply_prepare_model_middleware(
            self._middleware,
            initial_invocation,
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
        if self._policy.mode != "enforced":
            return

        required: tuple[type[KernelMiddleware], ...] = (
            PIIScrubberMiddleware,
            QuotaMiddleware,
            CapabilityGuardMiddleware,
        )
        for middleware_type in required:
            if not any(
                isinstance(middleware_item, middleware_type)
                for middleware_item in self._middleware
            ):
                raise ValueError(
                    "KernelPolicy(mode='enforced') requires middleware "
                    f"{middleware_type.__name__}."
                )
        if not any(
            hasattr(middleware_item, "prepare_tool_request")
            and hasattr(middleware_item, "prepare_tool_result")
            for middleware_item in self._middleware
        ):
            raise ValueError(
                "KernelPolicy(mode='enforced') requires tool IO policy middleware hooks."
            )


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
) -> str:
    messages_json = canonical_json_dumps(
        [message.model_dump(mode="json") for message in messages]
    )
    token = f"{model}|{step_key}|{prompt}|{messages_json}"
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
