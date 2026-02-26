from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import ArtanaKernel
from artana.harness import (
    BaseHarness,
    DraftReviewVerifyResult,
    DraftReviewVerifySupervisor,
    HarnessContext,
)
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputT = TypeVar("OutputT", bound=BaseModel)


class DummyModelPort:
    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        output = request.output_schema.model_validate({})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0),
        )


class StaticHarness(BaseHarness[object]):
    def __init__(self, *, kernel: ArtanaKernel, result: object) -> None:
        super().__init__(kernel=kernel)
        self._result = result

    async def step(self, *, context: HarnessContext) -> object:
        return self._result


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_supervisor_template",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


@pytest.mark.asyncio
async def test_draft_review_verify_supervisor_template(tmp_path: Path) -> None:
    kernel = ArtanaKernel(
        store=SQLiteStore(str(tmp_path / "state.db")),
        model_port=DummyModelPort(),
    )
    try:
        supervisor = DraftReviewVerifySupervisor(
            kernel=kernel,
            tenant=_tenant(),
            drafter=StaticHarness(kernel=kernel, result={"draft": "patch"}),
            reviewer=StaticHarness(kernel=kernel, result={"review": "looks good"}),
            verifier=StaticHarness(kernel=kernel, result={"passed": True}),
        )
        result = await supervisor.run(run_id="run_supervisor_template")
        assert isinstance(result, DraftReviewVerifyResult)
        assert result.approved is True
        assert result.draft == {"draft": "patch"}
        assert result.review == {"review": "looks good"}
        assert result.verify == {"passed": True}
    finally:
        await kernel.close()
