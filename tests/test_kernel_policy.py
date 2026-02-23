from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import KernelModelClient
from artana.kernel import ArtanaKernel, KernelPolicy
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class CountingModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=3, completion_tokens=2, cost_usd=0.01),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_policy",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


@pytest.mark.asyncio
async def test_start_run_issues_kernel_run_id_and_replay_works(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)

    try:
        run = await kernel.start_run(tenant=_tenant())
        assert run.run_id
        assert run.tenant_id == "org_policy"

        first = await KernelModelClient(kernel=kernel).step(
            run_id=run.run_id,
            prompt="approve?",
            model="gpt-4o-mini",
            tenant=_tenant(),
            output_schema=Decision,
        )
        second = await KernelModelClient(kernel=kernel).step(
            run_id=run.run_id,
            prompt="approve?",
            model="gpt-4o-mini",
            tenant=_tenant(),
            output_schema=Decision,
        )

        assert first.replayed is False
        assert second.replayed is True
        assert model_port.calls == 1
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_start_run_rejects_existing_run_id(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(store=store, model_port=model_port)

    try:
        await KernelModelClient(kernel=kernel).step(
            run_id="run_existing",
            prompt="approve?",
            model="gpt-4o-mini",
            tenant=_tenant(),
            output_schema=Decision,
        )
        with pytest.raises(ValueError, match="already exists"):
            await kernel.start_run(tenant=_tenant(), run_id="run_existing")
    finally:
        await kernel.close()


def test_enforced_policy_requires_guard_middleware(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()

    with pytest.raises(ValueError, match="CapabilityGuardMiddleware"):
        ArtanaKernel(
            store=store,
            model_port=model_port,
            middleware=ArtanaKernel.default_middleware_stack(capabilities=False),
            policy=KernelPolicy.enforced(),
        )


@pytest.mark.asyncio
async def test_default_middleware_stack_satisfies_enforced_policy(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    model_port = CountingModelPort()
    kernel = ArtanaKernel(
        store=store,
        model_port=model_port,
        middleware=ArtanaKernel.default_middleware_stack(),
        policy=KernelPolicy.enforced(),
    )

    try:
        result = await KernelModelClient(kernel=kernel).step(
            run_id="run_policy_ok",
            prompt="approve?",
            model="gpt-4o-mini",
            tenant=_tenant(),
            output_schema=Decision,
        )
        assert result.replayed is False
        assert model_port.calls == 1
    finally:
        await kernel.close()
