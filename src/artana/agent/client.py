from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from artana.kernel import (
    ArtanaKernel,
    ContextVersion,
    ModelInput,
    ReplayPolicy,
    StepModelResult,
)
from artana.models import TenantContext

OutputT = TypeVar("OutputT", bound=BaseModel)


class KernelModelClient:
    def __init__(self, *, kernel: ArtanaKernel) -> None:
        self._kernel = kernel

    async def step(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        model: str,
        prompt: str,
        output_schema: type[OutputT],
        step_key: str | None = None,
        replay_policy: ReplayPolicy = "strict",
        context_version: ContextVersion | None = None,
    ) -> StepModelResult[OutputT]:
        try:
            await self._kernel.load_run(run_id=run_id)
        except ValueError:
            await self._kernel.start_run(tenant=tenant, run_id=run_id)
        return await self._kernel.step_model(
            run_id=run_id,
            tenant=tenant,
            model=model,
            input=ModelInput.from_prompt(prompt),
            output_schema=output_schema,
            step_key=step_key,
            replay_policy=replay_policy,
            context_version=context_version,
        )


SingleStepModelClient = KernelModelClient


__all__ = ["KernelModelClient", "SingleStepModelClient"]
