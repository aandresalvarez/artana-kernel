from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import ArtanaKernel
from artana.events import EventType, ModelRequestedPayload
from artana.harness import TaskUnit, TestDrivenHarness
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class AdjudicationModelPort:
    async def complete(self, request: ModelRequest[OutputModelT]) -> ModelResult[OutputModelT]:
        passed = '"status":"passed"' in request.prompt or '"status": "passed"' in request.prompt
        output = request.output_schema.model_validate(
            {
                "passed": passed,
                "reasoning": "tests passed" if passed else "tests failed",
            }
        )
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=3, completion_tokens=2, cost_usd=0.001),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_tdd",
        capabilities=frozenset(),
        budget_usd_limit=2.0,
    )


class VerificationHarness(TestDrivenHarness):
    def __init__(self, *, kernel: ArtanaKernel, command: str) -> None:
        super().__init__(
            kernel=kernel,
            tenant=_tenant(),
            verify_model="gpt-5.3-codex",
        )
        self._command = command

    async def define_tasks(self) -> list[TaskUnit]:
        return [TaskUnit(id="fix_tests", description="Fix and verify tests")]

    async def work_on(self, task: TaskUnit) -> None:
        await self.verify_and_commit(task_id=task.id, test_command=self._command)


@pytest.mark.asyncio
async def test_test_driven_harness_marks_done_only_when_verification_passes(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=AdjudicationModelPort())

    @kernel.tool()
    async def execute_local_test(command: str) -> str:
        return json.dumps({"command": command, "status": "passed"})

    harness = VerificationHarness(kernel=kernel, command="pytest -q")
    try:
        result = await harness.run(run_id="run_tdd_pass")
        assert [unit.state for unit in result] == ["done"]

        events = await store.get_events_for_run("run_tdd_pass")
        requested = [
            event.payload
            for event in events
            if event.event_type == EventType.MODEL_REQUESTED
            and isinstance(event.payload, ModelRequestedPayload)
        ]
        assert requested
        assert requested[0].model == "gpt-5.3-codex"
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_test_driven_harness_resets_task_when_verification_fails(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    kernel = ArtanaKernel(store=store, model_port=AdjudicationModelPort())

    @kernel.tool()
    async def execute_local_test(command: str) -> str:
        return json.dumps({"command": command, "status": "failed"})

    harness = VerificationHarness(kernel=kernel, command="pytest -q")
    try:
        result = await harness.run(run_id="run_tdd_fail")
        assert [unit.state for unit in result] == ["pending"]
    finally:
        await kernel.close()
