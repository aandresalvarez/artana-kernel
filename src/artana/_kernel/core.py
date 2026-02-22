from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeVar, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel

from artana._kernel.model_cycle import get_or_execute_model_step
from artana._kernel.policies import apply_prepare_model_middleware, enforce_capability_scope
from artana._kernel.replay import validate_tenant_for_run
from artana._kernel.tool_cycle import (
    execute_tool_step_with_replay,
    reconcile_tool_with_replay,
)
from artana._kernel.types import (
    KernelPolicy,
    ModelInput,
    OutputT,
    PauseTicket,
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
    EventType,
    PauseRequestedPayload,
    ResumeRequestedPayload,
    RunStartedPayload,
)
from artana.middleware import order_middleware
from artana.middleware.base import KernelMiddleware, ModelInvocation
from artana.middleware.capability_guard import CapabilityGuardMiddleware
from artana.middleware.pii_scrubber import PIIScrubberMiddleware
from artana.middleware.quota import QuotaMiddleware
from artana.models import TenantContext
from artana.ports.model import ModelPort
from artana.ports.tool import LocalToolRegistry, ToolPort
from artana.store.base import EventStore

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

        event = await self._store.append_event(
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

    def tool(
        self, *, requires_capability: str | None = None
    ) -> Callable[[ToolCallable], ToolCallable]:
        def decorator(function: ToolCallable) -> ToolCallable:
            self._tool_port.register(
                function=function,
                requires_capability=requires_capability,
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
    ) -> PauseTicket:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(
                f"Cannot pause unknown run_id={run_id!r}; call start_run first."
            )
        validate_tenant_for_run(events=events, tenant=tenant)
        context_json = context.model_dump_json() if context is not None else None
        event = await self._store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.PAUSE_REQUESTED,
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

    async def step_model(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        input: ModelInput,
        output_schema: type[OutputT],
        step_key: str | None = None,
    ) -> StepModelResult[OutputT]:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(
                f"No events found for run_id={run_id!r}; call start_run first."
            )
        validate_tenant_for_run(events=events, tenant=tenant)

        prompt, messages = _normalize_model_input(input)
        initial_invocation = ModelInvocation(
            run_id=run_id,
            tenant=tenant,
            model=model,
            prompt=prompt,
            messages=messages,
            allowed_tools=tuple(self._tool_port.to_all_tool_definitions()),
            tool_capability_by_name=self._tool_port.capability_map(),
        )
        prepared_invocation = await apply_prepare_model_middleware(
            self._middleware,
            initial_invocation,
        )
        scoped_invocation = enforce_capability_scope(prepared_invocation)

        model_result = await get_or_execute_model_step(
            store=self._store,
            model_port=self._model_port,
            middleware=self._middleware,
            run_id=run_id,
            prompt=scoped_invocation.prompt,
            messages=scoped_invocation.messages,
            model=scoped_invocation.model,
            tenant=tenant,
            output_schema=output_schema,
            tool_definitions=scoped_invocation.allowed_tools,
            allowed_tool_names=[tool.name for tool in scoped_invocation.allowed_tools],
            events=events,
            step_key=step_key,
        )
        return StepModelResult(
            run_id=run_id,
            seq=model_result.completed_seq,
            output=model_result.output,
            usage=model_result.usage,
            tool_calls=model_result.tool_calls,
            replayed=model_result.replayed,
        )

    async def step_tool(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments: BaseModel,
        step_key: str | None = None,
    ) -> StepToolResult:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(
                f"No events found for run_id={run_id!r}; call start_run first."
            )
        validate_tenant_for_run(events=events, tenant=tenant)
        arguments_json = json.dumps(
            arguments.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        result = await execute_tool_step_with_replay(
            store=self._store,
            tool_port=self._tool_port,
            run_id=run_id,
            tenant=tenant,
            tool_name=tool_name,
            arguments_json=arguments_json,
            step_key=step_key,
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
    ) -> str:
        arguments_json = json.dumps(
            arguments.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return await reconcile_tool_with_replay(
            store=self._store,
            tool_port=self._tool_port,
            run_id=run_id,
            tenant=tenant,
            tool_name=tool_name,
            arguments_json=arguments_json,
            step_key=step_key,
        )

    async def resume(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        human_input: BaseModel | None = None,
    ) -> RunRef:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(f"No events found for run_id={run_id!r}.")
        validate_tenant_for_run(events=events, tenant=tenant)
        human_input_json = human_input.model_dump_json() if human_input is not None else None
        event = await self._store.append_event(
            run_id=run_id,
            tenant_id=tenant.tenant_id,
            event_type=EventType.RESUME_REQUESTED,
            payload=ResumeRequestedPayload(human_input_json=human_input_json),
        )
        return RunHandle(run_id=event.run_id, tenant_id=event.tenant_id)

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


def _derive_prompt_from_messages(messages: tuple[ChatMessage, ...]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return "\n".join(f"{message.role}: {message.content}" for message in messages)
