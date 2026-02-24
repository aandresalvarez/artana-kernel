from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from artana.agent import AutonomousAgent, ContextBuilder
from artana.agent.experience import RuleType, SQLiteExperienceStore
from artana.events import ChatMessage
from artana.kernel import ArtanaKernel, KernelPolicy
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage, ToolCall
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class FinancialExtractionResult(BaseModel):
    status: str
    attempted_date: str
    note: str


class AdaptiveFinanceModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        usage = ModelUsage(prompt_tokens=24, completion_tokens=16, cost_usd=0.0)
        if "extracted_rules" in request.output_schema.model_fields:
            output = request.output_schema.model_validate(
                {
                    "extracted_rules": [
                        {
                            "rule_id": "rule_iso_date",
                            "tenant_id": "placeholder",
                            "task_category": "placeholder",
                            "rule_type": RuleType.WIN_PATTERN.value,
                            "content": "Always format dates as YYYY-MM-DD.",
                            "success_count": 1,
                            "fail_count": 0,
                        }
                    ]
                }
            )
            return ModelResult(output=output, usage=usage)

        has_learning = _has_iso_learning(request.messages)
        latest_tool_payload = _extract_tool_payload(request.messages)

        if has_learning:
            output = request.output_schema.model_validate(
                {
                    "status": "success",
                    "attempted_date": "2026-02-23",
                    "note": "Used injected learning and skipped retries.",
                }
            )
            return ModelResult(output=output, usage=usage)

        if latest_tool_payload is None:
            output = request.output_schema.model_validate(
                {
                    "status": "retrying",
                    "attempted_date": "02/23/2026",
                    "note": "First attempt uses legacy formatting.",
                }
            )
            return ModelResult(
                output=output,
                usage=usage,
                tool_calls=(
                    ToolCall(
                        tool_name="submit_financial_extract",
                        arguments_json='{"date":"02/23/2026"}',
                        tool_call_id="submit_extract_legacy_date",
                    ),
                ),
            )

        if latest_tool_payload.get("ok") is False:
            output = request.output_schema.model_validate(
                {
                    "status": "retrying",
                    "attempted_date": "2026-02-23",
                    "note": "Retrying with ISO date format.",
                }
            )
            return ModelResult(
                output=output,
                usage=usage,
                tool_calls=(
                    ToolCall(
                        tool_name="submit_financial_extract",
                        arguments_json='{"date":"2026-02-23"}',
                        tool_call_id="submit_extract_iso_date",
                    ),
                ),
            )

        output = request.output_schema.model_validate(
            {
                "status": "success",
                "attempted_date": str(latest_tool_payload.get("accepted_date", "unknown")),
                "note": "Extraction submitted after observing tool feedback.",
            }
        )
        return ModelResult(output=output, usage=usage)


def _has_iso_learning(messages: list[ChatMessage]) -> bool:
    marker = "Always format dates as YYYY-MM-DD."
    return any(marker in message.content for message in messages)


def _extract_tool_payload(messages: list[ChatMessage]) -> dict[str, object] | None:
    for message in reversed(messages):
        if message.role != "tool":
            continue
        payload_text = message.content
        if payload_text.startswith("submit_financial_extract: "):
            payload_text = payload_text.removeprefix("submit_financial_extract: ")
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


async def main() -> None:
    state_path = Path("examples/.state_adaptive_agent_learning.db")
    experience_path = Path("examples/.experience_adaptive_agent_learning.db")
    if state_path.exists():
        state_path.unlink()
    if experience_path.exists():
        experience_path.unlink()

    store = SQLiteStore(str(state_path))
    experience_store = SQLiteExperienceStore(str(experience_path))
    kernel = ArtanaKernel(
        store=store,
        model_port=AdaptiveFinanceModelPort(),
        middleware=ArtanaKernel.default_middleware_stack(),
        policy=KernelPolicy.enforced(),
    )
    submit_calls = [0]

    @kernel.tool()
    async def submit_financial_extract(date: str) -> str:
        submit_calls[0] += 1
        if "/" in date:
            return json.dumps({"ok": False, "error": "expected_yyyy_mm_dd", "received": date})
        return json.dumps({"ok": True, "accepted_date": date})

    tenant = TenantContext(
        tenant_id="org_adaptive_finance",
        capabilities=frozenset(),
        budget_usd_limit=3.0,
    )
    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=ContextBuilder(
            identity="You are a financial extraction agent.",
            experience_store=experience_store,
            task_category="Financial_Reporting",
            progressive_skills=False,
        ),
        auto_reflect=True,
        reflection_model="gpt-4o-mini",
    )

    try:
        first = await agent.run(
            run_id="financial_week_1",
            tenant=tenant,
            model="adaptive-finance-demo",
            prompt="Submit this week's extraction using the accepted date format.",
            output_schema=FinancialExtractionResult,
            max_iterations=6,
        )
        first_run_tool_calls = submit_calls[0]
        learned_rules = await experience_store.get_rules(
            tenant_id=tenant.tenant_id,
            task_category="Financial_Reporting",
        )

        second = await agent.run(
            run_id="financial_week_2",
            tenant=tenant,
            model="adaptive-finance-demo",
            prompt="Submit this week's extraction using the accepted date format.",
            output_schema=FinancialExtractionResult,
            max_iterations=6,
        )
        second_run_tool_calls = submit_calls[0] - first_run_tool_calls

        print("Run 1:")
        print(first.model_dump_json(indent=2))
        print("Run 1 tool calls:", first_run_tool_calls)
        print("\nLearned rules:")
        print([rule.model_dump(mode="json") for rule in learned_rules])
        print("\nRun 2:")
        print(second.model_dump_json(indent=2))
        print("Run 2 tool calls:", second_run_tool_calls)
    finally:
        await experience_store.close()
        await kernel.close()
        if state_path.exists():
            state_path.unlink()
        if experience_path.exists():
            experience_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
