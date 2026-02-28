from __future__ import annotations

import hashlib
import inspect
import re
import warnings
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

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


@dataclass(frozen=True, slots=True)
class ModelClientCapabilities:
    supports_replay_policy: bool
    supports_context_version: bool
    supports_retry_failed_step: bool


class _StepModelCompatCallable(Protocol):
    def __call__(self, **kwargs: object) -> Awaitable[object]:
        ...


class KernelModelClient:
    def __init__(self, kernel: ArtanaKernel) -> None:
        self._kernel = kernel
        self._capabilities: ModelClientCapabilities | None = None

    def capabilities(self) -> ModelClientCapabilities:
        if self._capabilities is None:
            self._capabilities = _infer_step_model_capabilities(self._kernel)
        return self._capabilities

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
        retry_failed_step: bool = False,
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
        try:
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
                retry_failed_step=retry_failed_step,
            )
        except TypeError as exc:
            unsupported_kwargs = _unsupported_kwargs_from_type_error(exc)
            capabilities = self.capabilities()
            if not capabilities.supports_replay_policy:
                unsupported_kwargs.add("replay_policy")
            if not capabilities.supports_context_version:
                unsupported_kwargs.add("context_version")
            if not capabilities.supports_retry_failed_step:
                unsupported_kwargs.add("retry_failed_step")
            if not unsupported_kwargs:
                raise

            self._capabilities = ModelClientCapabilities(
                supports_replay_policy=(
                    capabilities.supports_replay_policy
                    and "replay_policy" not in unsupported_kwargs
                ),
                supports_context_version=(
                    capabilities.supports_context_version
                    and "context_version" not in unsupported_kwargs
                ),
                supports_retry_failed_step=(
                    capabilities.supports_retry_failed_step
                    and "retry_failed_step" not in unsupported_kwargs
                ),
            )
            warnings.warn(
                "KernelModelClient compatibility fallback activated. "
                f"kernel={type(self._kernel).__name__}, "
                f"unsupported_kwargs={sorted(unsupported_kwargs)}",
                UserWarning,
                stacklevel=2,
            )
            fallback_kwargs: dict[str, object] = {
                "run_id": run_id,
                "tenant": tenant,
                "model": model,
                "input": ModelInput.from_prompt(prompt),
                "output_schema": output_schema,
                "step_key": resolved_step_key,
                "model_options": model_options,
            }
            if "replay_policy" not in unsupported_kwargs:
                fallback_kwargs["replay_policy"] = replay_policy
            if "context_version" not in unsupported_kwargs:
                fallback_kwargs["context_version"] = context_version
            if "retry_failed_step" not in unsupported_kwargs:
                fallback_kwargs["retry_failed_step"] = retry_failed_step
            step_model = cast(_StepModelCompatCallable, self._kernel.step_model)
            fallback_result = await step_model(**fallback_kwargs)
            return cast(StepModelResult[OutputT], fallback_result)


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


def _infer_step_model_capabilities(kernel: ArtanaKernel) -> ModelClientCapabilities:
    try:
        signature = inspect.signature(kernel.step_model)
    except (TypeError, ValueError):
        return ModelClientCapabilities(
            supports_replay_policy=True,
            supports_context_version=True,
            supports_retry_failed_step=True,
        )
    has_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    return ModelClientCapabilities(
        supports_replay_policy=has_var_keyword or "replay_policy" in signature.parameters,
        supports_context_version=has_var_keyword or "context_version" in signature.parameters,
        supports_retry_failed_step=(
            has_var_keyword or "retry_failed_step" in signature.parameters
        ),
    )


def _unsupported_kwargs_from_type_error(error: TypeError) -> set[str]:
    message = str(error)
    unsupported: set[str] = set()
    for key in ("replay_policy", "context_version", "retry_failed_step"):
        if (
            f"unexpected keyword argument '{key}'" in message
            or f'unexpected keyword argument "{key}"' in message
        ):
            unsupported.add(key)
    return unsupported


__all__ = ["KernelModelClient", "ModelClientCapabilities", "SingleStepModelClient"]
