from __future__ import annotations

from typing import TypeVar, cast

import pytest
from pydantic import BaseModel

from artana import ContextVersion, KernelModelClient, ModelClientCapabilities
from artana.kernel import ArtanaKernel, ModelInput, ReplayPolicy, StepModelResult
from artana.models import TenantContext
from artana.ports.model import ModelCallOptions, ModelUsage

OutputT = TypeVar("OutputT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_client_compat",
        capabilities=frozenset(),
        budget_usd_limit=10.0,
    )


def _result(
    *,
    run_id: str,
    output_schema: type[OutputT],
) -> StepModelResult[OutputT]:
    return StepModelResult(
        run_id=run_id,
        seq=1,
        output=output_schema.model_validate({"approved": True}),
        usage=ModelUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.0),
        tool_calls=(),
        replayed=False,
    )


class CurrentKernelStub:
    def __init__(self) -> None:
        self.replay_policy_seen: ReplayPolicy | None = None
        self.context_version_seen: ContextVersion | None = None

    async def load_run(self, *, run_id: str) -> None:
        raise ValueError("missing run")

    async def start_run(self, *, tenant: TenantContext, run_id: str) -> None:
        return None

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
        retry_failed_step: bool = False,
        parent_step_key: str | None = None,
    ) -> StepModelResult[OutputT]:
        self.replay_policy_seen = replay_policy
        self.context_version_seen = context_version
        return _result(run_id=run_id, output_schema=output_schema)


class LegacyKernelStub:
    def __init__(self) -> None:
        self.step_calls = 0

    async def load_run(self, *, run_id: str) -> None:
        raise ValueError("missing run")

    async def start_run(self, *, tenant: TenantContext, run_id: str) -> None:
        return None

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
        parent_step_key: str | None = None,
    ) -> StepModelResult[OutputT]:
        self.step_calls += 1
        return _result(run_id=run_id, output_schema=output_schema)


@pytest.mark.asyncio
async def test_step_forwards_replay_policy_on_current_kernel() -> None:
    kernel = CurrentKernelStub()
    client = KernelModelClient(kernel=cast(ArtanaKernel, kernel))
    capabilities = client.capabilities()

    assert capabilities == ModelClientCapabilities(
        supports_replay_policy=True,
        supports_context_version=True,
        supports_retry_failed_step=True,
    )

    context_version = ContextVersion(system_prompt_hash="abc123")
    await client.step(
        run_id="run_current",
        tenant=_tenant(),
        model="gpt-4o-mini",
        prompt="hello",
        output_schema=Decision,
        replay_policy="allow_prompt_drift",
        context_version=context_version,
    )

    assert kernel.replay_policy_seen == "allow_prompt_drift"
    assert kernel.context_version_seen == context_version


@pytest.mark.asyncio
async def test_step_fallback_retries_without_unsupported_kwargs() -> None:
    kernel = LegacyKernelStub()
    client = KernelModelClient(kernel=cast(ArtanaKernel, kernel))
    capabilities = client.capabilities()

    assert capabilities == ModelClientCapabilities(
        supports_replay_policy=False,
        supports_context_version=False,
        supports_retry_failed_step=False,
    )

    with pytest.warns(UserWarning, match="unsupported_kwargs"):
        result = await client.step(
            run_id="run_legacy",
            tenant=_tenant(),
            model="gpt-4o-mini",
            prompt="hello",
            output_schema=Decision,
            replay_policy="allow_prompt_drift",
            context_version=ContextVersion(system_prompt_hash="abc123"),
        )

    assert result.output.approved is True
    assert kernel.step_calls == 1
