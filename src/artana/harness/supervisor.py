from __future__ import annotations

from typing import TypeVar

from artana.harness.base import BaseHarness
from artana.models import TenantContext

ChildResultT = TypeVar("ChildResultT")


class SupervisorHarness(BaseHarness[object]):
    async def run_child(
        self,
        harness: BaseHarness[ChildResultT],
        *,
        run_id: str,
        tenant: TenantContext | None = None,
        model: str | None = None,
    ) -> ChildResultT:
        return await harness.run(
            run_id=run_id,
            tenant=self._resolve_tenant(tenant=tenant),
            model=model,
        )


__all__ = ["SupervisorHarness"]
