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


def enforce_capability_scope(invocation: ModelInvocation) -> ModelInvocation:
    filtered_tools = tuple(
        tool
        for tool in invocation.allowed_tools
        if is_tool_allowed_for_tenant(
            tool_name=tool.name,
            tenant=invocation.tenant,
            tool_capability_by_name=invocation.tool_capability_by_name,
        )
    )
    return invocation.with_updates(allowed_tools=filtered_tools)


def is_tool_allowed_for_tenant(
    *,
    tool_name: str,
    tenant: TenantContext,
    tool_capability_by_name: dict[str, str | None],
) -> bool:
    required_capability = tool_capability_by_name.get(tool_name)
    if required_capability is None:
        return True
    return required_capability in tenant.capabilities


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

