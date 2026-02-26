from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from artana.harness.base import BaseHarness, HarnessContext
from artana.harness.supervisor import SupervisorHarness
from artana.kernel import ArtanaKernel
from artana.models import TenantContext


@dataclass(frozen=True, slots=True)
class DraftReviewVerifyResult:
    draft: object
    review: object
    verify: object
    approved: bool


class DraftReviewVerifySupervisor(SupervisorHarness):
    def __init__(
        self,
        *,
        kernel: ArtanaKernel,
        tenant: TenantContext | None = None,
        drafter: BaseHarness[object],
        reviewer: BaseHarness[object],
        verifier: BaseHarness[object],
        verify_passes: Callable[[object], bool] | None = None,
        default_model: str = "gpt-4o-mini",
    ) -> None:
        super().__init__(kernel=kernel, tenant=tenant, default_model=default_model)
        self._drafter = drafter
        self._reviewer = reviewer
        self._verifier = verifier
        self._verify_passes = verify_passes or _default_verify_passes

    async def step(self, *, context: HarnessContext) -> DraftReviewVerifyResult:
        draft_result = await self.run_child(
            harness=self._drafter,
            run_id=f"{context.run_id}::draft",
            tenant=context.tenant,
            model=context.model,
        )
        review_result = await self.run_child(
            harness=self._reviewer,
            run_id=f"{context.run_id}::review",
            tenant=context.tenant,
            model=context.model,
        )
        verify_result = await self.run_child(
            harness=self._verifier,
            run_id=f"{context.run_id}::verify",
            tenant=context.tenant,
            model=context.model,
        )
        return DraftReviewVerifyResult(
            draft=draft_result,
            review=review_result,
            verify=verify_result,
            approved=self._verify_passes(verify_result),
        )


def _default_verify_passes(result: object) -> bool:
    if isinstance(result, bool):
        return result
    if hasattr(result, "passed"):
        passed = getattr(result, "passed")
        if isinstance(passed, bool):
            return passed
    if isinstance(result, dict):
        flag = result.get("passed")
        if isinstance(flag, bool):
            return flag
        status = result.get("status")
        if isinstance(status, str):
            return status.lower() in {"passed", "success", "ok"}
    return bool(result)


__all__ = ["DraftReviewVerifyResult", "DraftReviewVerifySupervisor"]
