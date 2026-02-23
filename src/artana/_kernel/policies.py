from __future__ import annotations

from collections.abc import Sequence

from artana._kernel.types import CapabilityDeniedError
from artana.middleware.base import KernelMiddleware, ModelInvocation
from artana.models import TenantContext


async def apply_prepare_model_middleware(
    middleware: Sequence[KernelMiddleware],
    invocation: ModelInvocation,
) -> ModelInvocation:
    current = invocation
    for middleware_item in middleware:
        current = await middleware_item.prepare_model(current)
    return current


async def apply_prepare_tool_request_middleware(
    middleware: Sequence[KernelMiddleware],
    *,
    run_id: str,
    tenant: TenantContext,
    tool_name: str,
    arguments_json: str,
) -> str:
    current = arguments_json
    for middleware_item in middleware:
        prepare_tool_request = getattr(middleware_item, "prepare_tool_request", None)
        if not callable(prepare_tool_request):
            continue
        current = await prepare_tool_request(
            run_id=run_id,
            tenant=tenant,
            tool_name=tool_name,
            arguments_json=current,
        )
    return current


async def apply_prepare_tool_result_middleware(
    middleware: Sequence[KernelMiddleware],
    *,
    run_id: str,
    tenant: TenantContext,
    tool_name: str,
    result_json: str,
) -> str:
    current = result_json
    for middleware_item in middleware:
        prepare_tool_result = getattr(middleware_item, "prepare_tool_result", None)
        if not callable(prepare_tool_result):
            continue
        current = await prepare_tool_result(
            run_id=run_id,
            tenant=tenant,
            tool_name=tool_name,
            result_json=current,
        )
    return current


def assert_tool_allowed_for_tenant(
    *,
    tool_name: str,
    tenant: TenantContext,
    capability_map: dict[str, str | None],
) -> None:
    required_capability = capability_map.get(tool_name)
    if required_capability is None:
        if tool_name in capability_map:
            return
        raise KeyError(f"Tool {tool_name!r} is not registered.")
    if required_capability not in tenant.capabilities:
        raise CapabilityDeniedError(
            f"Tool {tool_name!r} requires capability {required_capability!r}."
        )
