from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

from artana._kernel.tool_state import resolve_tool_resolutions
from artana.agent.autonomous import AutonomousAgent
from artana.agent.model_steps import execute_model_step
from artana.events import ChatMessage
from artana.kernel import (
    ArtanaKernel,
    ContextVersion,
    ModelInput,
    ReplayPolicy,
    StepModelResult,
    StepToolResult,
)
from artana.models import TenantContext
from artana.ports.model import ToolDefinition

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
    ) -> None:
        self._kernel = kernel
        self._tenant = tenant
        self._default_model = default_model
        self._replay_policy = replay_policy
        self._step_key_namespace = _normalize_step_key_prefix(type(self).__name__)
        self._active_context: HarnessContext | None = None
        self._step_key_counters: dict[str, int] = {}

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
    ) -> HarnessResultT:
        resolved_tenant = self._resolve_tenant(tenant)
        run_created = await self._ensure_run_exists(run_id=run_id, tenant=resolved_tenant)
        context = HarnessContext(
            run_id=run_id,
            tenant=resolved_tenant,
            model=model if model is not None else self._default_model,
            run_created=run_created,
        )
        try:
            self._activate_context(context=context)
            if context.run_created:
                await self.on_initialize(context=context)

            reorientation = await self._build_wake_reorientation(context=context)
            await self.emit_summary(
                run_id=context.run_id,
                tenant=context.tenant,
                summary_type="wake_reorientation",
                payload=reorientation,
                step_key="wake_reorientation",
            )

            result: HarnessResultT | None = None
            execution_error: Exception | None = None
            sleep_error: Exception | None = None
            try:
                await self.on_wake(context=context, reorientation=reorientation)
                result = await self.step(context=context)
            except Exception as exc:
                execution_error = exc

            try:
                await self.on_sleep(context=context, execution_error=execution_error)
            except Exception as exc:
                sleep_error = exc

            await self.emit_summary(
                run_id=context.run_id,
                tenant=context.tenant,
                summary_type="harness_sleep",
                payload={
                    "status": (
                        "failed"
                        if execution_error is not None or sleep_error is not None
                        else "completed"
                    ),
                    "execution_error": _error_payload(execution_error),
                    "sleep_error": _error_payload(sleep_error),
                },
                step_key="harness_sleep",
            )

            if execution_error is not None:
                raise execution_error
            if sleep_error is not None:
                raise sleep_error
            if result is None:
                raise RuntimeError("Harness completed without returning a result.")
            return result
        finally:
            self._deactivate_context()

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
        replay_policy: ReplayPolicy | None = None,
        context_version: ContextVersion | None = None,
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
            replay_policy=replay_policy,
            context_version=context_version,
        )

    async def run_tool(
        self,
        *,
        tool_name: str,
        arguments: BaseModel,
        step_key: str | None = None,
        run_id: str | None = None,
        tenant: TenantContext | None = None,
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
        )

    async def emit_summary(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        summary_type: str,
        payload: object,
        step_key: str,
    ) -> int:
        required_step_key = self.require_step_key(step_key)
        return await self._kernel.append_run_summary(
            run_id=run_id,
            tenant=tenant,
            summary_type=summary_type,
            summary_json=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            step_key=required_step_key,
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
        replay_policy: ReplayPolicy | None = None,
        context_version: ContextVersion | None = None,
    ) -> StepModelResult[OutputT]:
        required_step_key = self.require_step_key(step_key)
        return await self._kernel.step_model(
            run_id=run_id,
            tenant=tenant,
            model=model,
            input=input,
            output_schema=output_schema,
            step_key=required_step_key,
            visible_tool_names=visible_tool_names,
            replay_policy=(
                self._replay_policy if replay_policy is None else replay_policy
            ),
            context_version=context_version,
        )

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
        replay_policy: ReplayPolicy | None = None,
        context_version: ContextVersion | None = None,
    ) -> StepModelResult[OutputT]:
        required_step_key = self.require_step_key(step_key)
        return await execute_model_step(
            kernel=self._kernel,
            run_id=run_id,
            tenant=tenant,
            model=model,
            messages=messages,
            output_schema=output_schema,
            step_key=required_step_key,
            visible_tool_names=visible_tool_names,
            replay_policy=(
                self._replay_policy if replay_policy is None else replay_policy
            ),
            context_version=context_version,
        )

    async def tool_step(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments: BaseModel,
        step_key: str,
    ) -> StepToolResult:
        required_step_key = self.require_step_key(step_key)
        return await self._kernel.step_tool(
            run_id=run_id,
            tenant=tenant,
            tool_name=tool_name,
            arguments=arguments,
            step_key=required_step_key,
        )

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

    def _deactivate_context(self) -> None:
        self._active_context = None
        self._step_key_counters = {}


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
