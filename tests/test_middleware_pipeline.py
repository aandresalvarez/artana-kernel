from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import KernelModelClient
from artana.events import ModelRequestedPayload
from artana.kernel import ArtanaKernel
from artana.middleware import (
    CapabilityGuardMiddleware,
    PIIScrubberMiddleware,
    QuotaMiddleware,
)
from artana.middleware.base import ModelInvocation
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class CaptureModelPort:
    def __init__(self) -> None:
        self.last_prompt: str | None = None
        self.last_tools: list[str] = []

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.last_prompt = request.prompt
        self.last_tools = [tool.name for tool in request.allowed_tools]
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=10, completion_tokens=5, cost_usd=0.01),
        )


class RecordingPIIScrubber(PIIScrubberMiddleware):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self._log = log

    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        self._log.append("pii.prepare")
        return await super().prepare_model(invocation)


class RecordingQuota(QuotaMiddleware):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self._log = log

    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        self._log.append("quota.prepare")
        return await super().prepare_model(invocation)


class RecordingCapabilityGuard(CapabilityGuardMiddleware):
    def __init__(self, log: list[str]) -> None:
        self._log = log

    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        self._log.append("cap.prepare")
        return await super().prepare_model(invocation)


@pytest.mark.asyncio
async def test_pii_scrubber_redacts_prompt_before_model_request(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CaptureModelPort()
    kernel = ArtanaKernel(
        store=store,
        model_port=model_port,
        middleware=[PIIScrubberMiddleware()],
    )
    tenant = TenantContext(
        tenant_id="org_pii",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )
    raw_prompt = "Contact me at user@example.com or 415-555-9999."

    try:
        await KernelModelClient(kernel=kernel).chat(
            run_id="run_pii",
            prompt=raw_prompt,
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )

        assert model_port.last_prompt is not None
        assert "user@example.com" not in model_port.last_prompt
        assert "415-555-9999" not in model_port.last_prompt
        assert "[REDACTED_EMAIL]" in model_port.last_prompt
        assert "[REDACTED_PHONE]" in model_port.last_prompt

        events = await store.get_events_for_run("run_pii")
        requested_payload = events[1].payload
        assert isinstance(requested_payload, ModelRequestedPayload)
        assert "[REDACTED_EMAIL]" in requested_payload.prompt
        assert "[REDACTED_PHONE]" in requested_payload.prompt
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_kernel_enforces_middleware_order_for_known_middleware(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state_order.db"))
    model_port = CaptureModelPort()
    log: list[str] = []
    kernel = ArtanaKernel(
        store=store,
        model_port=model_port,
        middleware=[
            RecordingCapabilityGuard(log),
            RecordingPIIScrubber(log),
            RecordingQuota(log),
        ],
    )
    tenant = TenantContext(
        tenant_id="org_order",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )

    try:
        await KernelModelClient(kernel=kernel).chat(
            run_id="run_order",
            prompt="order check",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
        assert log == ["pii.prepare", "quota.prepare", "cap.prepare"]
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_capability_guard_filters_unauthorized_tools(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CaptureModelPort()
    kernel = ArtanaKernel(
        store=store,
        model_port=model_port,
        middleware=[CapabilityGuardMiddleware()],
    )

    @kernel.tool(requires_capability="finance:read")
    async def get_balance(account_id: str) -> str:
        return '{"balance":"100"}'

    @kernel.tool(requires_capability="finance:write")
    async def execute_transfer(account_id: str, amount: str) -> str:
        return '{"status":"ok"}'

    tenant = TenantContext(
        tenant_id="org_guard",
        capabilities=frozenset({"finance:read"}),
        budget_usd_limit=1.0,
    )

    try:
        await KernelModelClient(kernel=kernel).chat(
            run_id="run_guard",
            prompt="Show tools",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )

        assert model_port.last_tools == ["get_balance"]
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_capability_guard_keeps_authorized_tools_visible(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state_cap_ok.db"))
    model_port = CaptureModelPort()
    kernel = ArtanaKernel(
        store=store,
        model_port=model_port,
        middleware=[CapabilityGuardMiddleware()],
    )

    @kernel.tool(requires_capability="finance:read")
    async def get_balance(account_id: str) -> str:
        return '{"balance":"100"}'

    @kernel.tool(requires_capability="finance:write")
    async def execute_transfer(account_id: str, amount: str) -> str:
        return '{"status":"ok"}'

    tenant = TenantContext(
        tenant_id="org_guard_ok",
        capabilities=frozenset({"finance:read", "finance:write"}),
        budget_usd_limit=1.0,
    )

    try:
        await KernelModelClient(kernel=kernel).chat(
            run_id="run_guard_ok",
            prompt="Show tools",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )

        assert model_port.last_tools == ["get_balance", "execute_transfer"]
    finally:
        await kernel.close()
