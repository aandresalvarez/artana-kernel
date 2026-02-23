from __future__ import annotations

from artana.middleware.base import ModelInvocation
from artana.models import TenantContext
from artana.ports.model import ModelUsage


class CapabilityGuardMiddleware:
    async def prepare_model(self, invocation: ModelInvocation) -> ModelInvocation:
        filtered_tools = tuple(
            tool
            for tool in invocation.allowed_tools
            if self._is_allowed(
                capability=invocation.tool_capability_by_name.get(tool.name),
                tenant=invocation.tenant,
            )
        )
        return invocation.with_updates(allowed_tools=filtered_tools)

    async def before_model(self, *, run_id: str, tenant: TenantContext) -> None:
        return None

    async def after_model(
        self, *, run_id: str, tenant: TenantContext, usage: ModelUsage
    ) -> None:
        return None

    async def prepare_tool_request(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        arguments_json: str,
    ) -> str:
        return arguments_json

    async def prepare_tool_result(
        self,
        *,
        run_id: str,
        tenant: TenantContext,
        tool_name: str,
        result_json: str,
    ) -> str:
        return result_json

    def _is_allowed(self, *, capability: str | None, tenant: TenantContext) -> bool:
        if capability is None:
            return True
        return capability in tenant.capabilities
