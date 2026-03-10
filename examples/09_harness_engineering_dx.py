from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from artana import (
    AcceptanceSpec,
    ArtanaKernel,
    AutonomousAgent,
    DraftVerifyLoopConfig,
    ModelCallOptions,
    StepKey,
    TenantContext,
    ToolGate,
)
from artana.harness import BaseHarness, HarnessContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.ports.tool import ToolExecutionContext
from artana.store import SQLiteStore
from artana.tools import CodingHarnessTools

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class DraftPlan(BaseModel):
    patch: str


class VerifyPlan(BaseModel):
    approved: bool
    notes: str


class AgentDecision(BaseModel):
    done: bool
    summary: str


class MutateArgs(BaseModel):
    path: str
    content: str


class DemoModelPort:
    def __init__(self) -> None:
        self.models_seen: list[str] = []

    async def complete(
        self,
        request: ModelRequest[OutputModelT],
    ) -> ModelResult[OutputModelT]:
        self.models_seen.append(request.model)
        fields = request.output_schema.model_fields
        payload: dict[str, object]
        if "patch" in fields:
            payload = {"patch": "Add retry logic and tighten assertions."}
        elif "approved" in fields and "notes" in fields:
            payload = {
                "approved": True,
                "notes": "Plan is deterministic and testable.",
            }
        elif "done" in fields and "summary" in fields:
            payload = {"done": True, "summary": "Completed after passing required gates."}
        elif "accepted" in fields and "reasoning" in fields:
            payload = {"accepted": True, "reasoning": "All required gates passed."}
        else:
            payload = {}
        return ModelResult(
            output=request.output_schema.model_validate(payload),
            usage=ModelUsage(prompt_tokens=8, completion_tokens=5, cost_usd=0.0),
        )


class DXHarness(BaseHarness[dict[str, object]]):
    async def step(self, *, context: HarnessContext) -> dict[str, object]:
        step = StepKey(namespace=f"{context.run_id}_dx")

        draft = await self.run_draft_model(
            prompt=(
                "Plan a minimal patch for flaky tests.\n"
                "<tool_boundary>\n"
                "- This is a planning step only.\n"
                "- Do not assume shell execution or file mutation has already happened.\n"
                "</tool_boundary>\n"
                "<structured_output_contract>\n"
                "- Return only content suitable for the DraftPlan.patch field.\n"
                "- Keep it concise and implementation-oriented.\n"
                "</structured_output_contract>"
            ),
            output_schema=DraftPlan,
            step_key=step.next("draft"),
            model_options=ModelCallOptions(api_mode="auto", reasoning_effort="none"),
        )

        verify = await self.run_verify_model(
            prompt=(
                "Review this draft patch plan and decide whether it is ready to execute:\n"
                f"{draft.output.patch}\n\n"
                "<verification_loop>\n"
                "- Check correctness, determinism, and missing constraints.\n"
                "- Check whether the plan includes enough verification to justify execution.\n"
                "</verification_loop>\n"
                "<structured_output_contract>\n"
                "- Approve only if the plan is testable and low-ambiguity.\n"
                "- Return only VerifyPlan fields.\n"
                "</structured_output_contract>"
            ),
            output_schema=VerifyPlan,
            step_key=step.next("verify"),
            model_options=ModelCallOptions(api_mode="auto", reasoning_effort="high"),
        )

        mutation = await self.run_tool(
            tool_name="write_virtual_patch",
            arguments=MutateArgs(path="src/demo.py", content=draft.output.patch),
            step_key=step.next("mutation"),
        )
        mutation_payload = json.loads(mutation.result_json)

        return {
            "draft_patch": draft.output.patch,
            "verified": verify.output.approved,
            "verify_notes": verify.output.notes,
            "idempotency_key": mutation_payload["idempotency_key"],
        }


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_dx",
        capabilities=frozenset({"coding:write"}),
        budget_usd_limit=2.0,
    )


def _bundle_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        run_id="bundle_run",
        tenant_id="org_dx",
        idempotency_key="bundle_idemp_1",
        request_event_id=None,
        tool_version="1.0.0",
        schema_version="1",
    )


async def main() -> None:
    database_path = Path("examples/.state_harness_engineering_dx.db")
    bundle_root = Path("examples/.tmp_dx_bundle")

    if database_path.exists():
        database_path.unlink()
    if bundle_root.exists():
        shutil.rmtree(bundle_root)

    model_port = DemoModelPort()
    kernel = ArtanaKernel(
        store=SQLiteStore(str(database_path)),
        model_port=model_port,
        middleware=ArtanaKernel.default_middleware_stack(),
    )

    gate_state = {"calls": 0}

    @kernel.tool(requires_capability="coding:write", side_effect=True)
    async def write_virtual_patch(
        path: str,
        content: str,
        artana_context: ToolExecutionContext,
    ) -> str:
        return json.dumps(
            {
                "path": path,
                "content": content,
                "idempotency_key": artana_context.idempotency_key,
            }
        )

    @kernel.tool()
    async def run_tests() -> str:
        gate_state["calls"] += 1
        passed = gate_state["calls"] >= 2
        return json.dumps(
            {
                "passed": passed,
                "status": "passed" if passed else "failed",
            }
        )

    try:
        harness = DXHarness(
            kernel,
            tenant=_tenant(),
            draft_model="gpt-5-mini",
            verify_model="gpt-5.4",
        )
        harness_result = await harness.run(run_id="dx_harness_run")

        agent = AutonomousAgent(
            kernel,
            loop=DraftVerifyLoopConfig(
                draft_model="gpt-5-mini",
                verify_model="gpt-5.4",
            ),
        )
        agent_result = await agent.run(
            run_id="dx_agent_run",
            tenant=_tenant(),
            model="openai/gpt-5.4",
            system_prompt=(
                "You are a coding agent.\n"
                "<tool_persistence_rules>\n"
                "- Keep using available tools until the task is complete and verification passes.\n"
                "- Do not stop early while a required gate is still failing.\n"
                "</tool_persistence_rules>\n"
                "<verification_loop>\n"
                "- Before finalizing, confirm that every required check has passed.\n"
                "- If a gate fails, continue the loop instead of declaring success.\n"
                "</verification_loop>\n"
                "<structured_output_contract>\n"
                "- Return only AgentDecision fields.\n"
                "</structured_output_contract>"
            ),
            prompt=(
                "Complete the task only when all required checks pass.\n"
                "- Use the available workflow until run_tests passes.\n"
                "- Do not report done while any required gate is failing.\n"
                "- If you cannot complete the task, return a concise blocker summary."
            ),
            output_schema=AgentDecision,
            max_iterations=4,
            acceptance=AcceptanceSpec(gates=(ToolGate(tool="run_tests", must_pass=True),)),
        )

        coding_bundle = CodingHarnessTools(sandbox_root=str(bundle_root))
        registry = coding_bundle.registry()
        await registry.call("create_worktree", '{"name":"dx-worktree"}', context=_bundle_context())
        await registry.call(
            "apply_patch",
            '{"path":"notes.txt","content":"bundle-ready"}',
            context=_bundle_context(),
        )
        bundle_read = await registry.call(
            "read_file",
            '{"path":"notes.txt"}',
            context=_bundle_context(),
        )
        bundle_payload = json.loads(bundle_read.result_json)

        print("Harness result:", harness_result)
        print("Agent result:", agent_result.model_dump())
        print("Acceptance gate calls:", gate_state["calls"])
        print("Bundle read ok:", bundle_payload["ok"])
        print("Bundle content:", bundle_payload["content"])
        print("Models seen:", model_port.models_seen)
    finally:
        await kernel.close()
        if database_path.exists():
            database_path.unlink()
        if bundle_root.exists():
            shutil.rmtree(bundle_root)


if __name__ == "__main__":
    asyncio.run(main())
