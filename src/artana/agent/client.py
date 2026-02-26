from __future__ import annotations

import hashlib
import re
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
from artana.ports.model import ModelCallOptions

OutputT = TypeVar("OutputT", bound=BaseModel)


class KernelModelClient:
    def __init__(self, kernel: ArtanaKernel) -> None:
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
        model_options: ModelCallOptions | None = None,
        replay_policy: ReplayPolicy = "strict",
        context_version: ContextVersion | None = None,
    ) -> StepModelResult[OutputT]:
        resolved_step_key = (
            step_key
            if step_key is not None
            else _auto_step_key(
                model=model,
                prompt=prompt,
                output_schema=output_schema,
            )
        )
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
            step_key=resolved_step_key,
            model_options=model_options,
            replay_policy=replay_policy,
            context_version=context_version,
        )


def _auto_step_key(
    *,
    model: str,
    prompt: str,
    output_schema: type[BaseModel],
) -> str:
    model_token = re.sub(r"[^a-zA-Z0-9_]+", "_", model).strip("_").lower()
    if model_token == "":
        model_token = "model"
    schema_id = f"{output_schema.__module__}.{output_schema.__qualname__}"
    digest = hashlib.sha256(f"{model}\n{schema_id}\n{prompt}".encode("utf-8")).hexdigest()
    return f"kernelmodelclient_{model_token}_{digest[:12]}"


SingleStepModelClient = KernelModelClient


__all__ = ["KernelModelClient", "SingleStepModelClient"]
