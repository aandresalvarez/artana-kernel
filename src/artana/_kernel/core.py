from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeVar, runtime_checkable
from uuid import uuid4

from artana._kernel.model_cycle import get_or_execute_model_step
from artana._kernel.policies import apply_prepare_model_middleware, enforce_capability_scope
from artana._kernel.replay import derive_run_resume_state, validate_tenant_for_run
from artana._kernel.tool_cycle import (
    execute_or_replay_tools_for_model,
    execute_tool_with_replay,
    reconcile_tool_with_replay,
)
from artana._kernel.types import (
    ChatResponse,
    KernelPolicy,
    OutputT,
    PauseTicket,
    RunHandle,
    RunResumeState,
    ToolCallable,
)
from artana._kernel.workflow_runtime import (
    WorkflowContext,
    WorkflowRunResult,
    run_workflow,
)
from artana.events import ChatMessage, PauseRequestedPayload
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
    ) -> RunHandle:
        if run_id is not None:
            existing = await self._store.get_events_for_run(run_id)
            if existing:
                raise ValueError(
                    f"run_id={run_id!r} already exists; provide a different run_id."
                )
            return RunHandle(run_id=run_id, tenant_id=tenant.tenant_id)

        for _ in range(5):
            generated = uuid4().hex
            if not await self._store.get_events_for_run(generated):
                return RunHandle(run_id=generated, tenant_id=tenant.tenant_id)
        raise RuntimeError("Failed to allocate a unique run_id after multiple attempts.")

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

    async def chat(
        self,
        *,
        run_id: str | None,
        prompt: str,
        model: str,
        tenant: TenantContext,
        output_schema: type[OutputT],
    ) -> ChatResponse[OutputT]:
        run_id_value = run_id if run_id is not None else uuid4().hex
        initial_invocation = ModelInvocation(
            run_id=run_id_value,
            tenant=tenant,
            model=model,
            prompt=prompt,
            messages=(ChatMessage(role="user", content=prompt),),
            allowed_tools=tuple(self._tool_port.to_all_tool_definitions()),
            tool_capability_by_name=self._tool_port.capability_map(),
        )
        prepared_invocation = await apply_prepare_model_middleware(
            self._middleware,
            initial_invocation,
        )
        scoped_invocation = enforce_capability_scope(prepared_invocation)
        tool_definitions = scoped_invocation.allowed_tools
        allowed_tool_names = [tool.name for tool in tool_definitions]

        events = await self._store.get_events_for_run(run_id_value)
        validate_tenant_for_run(events=events, tenant=tenant)
        model_result = await get_or_execute_model_step(
            store=self._store,
            model_port=self._model_port,
            middleware=self._middleware,
            run_id=run_id_value,
            prompt=scoped_invocation.prompt,
            messages=scoped_invocation.messages,
            model=scoped_invocation.model,
            tenant=tenant,
            output_schema=output_schema,
            tool_definitions=tool_definitions,
            allowed_tool_names=allowed_tool_names,
            events=events,
        )
        await execute_or_replay_tools_for_model(
            store=self._store,
            tool_port=self._tool_port,
            run_id=run_id_value,
            tenant=tenant,
            model_completed_seq=model_result.completed_seq,
            expected_tool_calls=model_result.tool_calls,
            allowed_tool_names=frozenset(allowed_tool_names),
        )
        return ChatResponse(
            run_id=run_id_value,
            output=model_result.output,
            usage=model_result.usage,
            replayed=model_result.replayed,
        )

    async def pause_for_human(self, *, run_id: str, reason: str) -> PauseTicket:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(
                f"Cannot pause unknown run_id={run_id!r}; run must exist before pausing."
            )
        tenant_id = events[-1].tenant_id
        event = await self._store.append_event(
            run_id=run_id,
            tenant_id=tenant_id,
            event_type="pause_requested",
            payload=PauseRequestedPayload(reason=reason),
        )
        return PauseTicket(
            run_id=event.run_id,
            ticket_id=event.event_id,
            seq=event.seq,
            reason=reason,
        )

    async def execute_tool(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments_json: str,
    ) -> str:
        return await execute_tool_with_replay(
            store=self._store,
            tool_port=self._tool_port,
            run_id=run_id,
            tenant=tenant,
            tool_name=tool_name,
            arguments_json=arguments_json,
        )

    async def reconcile_tool(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments_json: str,
    ) -> str:
        return await reconcile_tool_with_replay(
            store=self._store,
            tool_port=self._tool_port,
            run_id=run_id,
            tenant=tenant,
            tool_name=tool_name,
            arguments_json=arguments_json,
        )

    async def resume(self, *, run_id: str) -> RunResumeState:
        events = await self._store.get_events_for_run(run_id)
        if not events:
            raise ValueError(f"No events found for run_id={run_id!r}.")
        return derive_run_resume_state(events)

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
