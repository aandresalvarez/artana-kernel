from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from artana.agent import AutonomousAgent, ContextBuilder
from artana.events import ChatMessage
from artana.kernel import ArtanaKernel, KernelPolicy
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage, ToolCall
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class CompanyReport(BaseModel):
    company_name: str
    current_ceo: str
    latest_news_summary: str


class ResearchModelPort:
    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        usage = ModelUsage(prompt_tokens=72, completion_tokens=96, cost_usd=0.0)
        tool_result = _extract_tool_result(request.messages)

        if tool_result is None:
            output = request.output_schema.model_validate(
                {
                    "company_name": "Acme Corp",
                    "current_ceo": "Acme leadership team",
                    "latest_news_summary": "In-progress.",
                }
            )
            tool_calls = (
                ToolCall(
                    tool_name="search_web",
                    arguments_json=json.dumps({"query": "Acme Corp CEO and latest news."}),
                    tool_call_id="search_web_call_1",
                ),
            )
            return ModelResult(
                output=output,
                usage=usage,
                tool_calls=tool_calls,
            )

        output = request.output_schema.model_validate(
            {
                "company_name": tool_result.get("company_name", "Acme Corp"),
                "current_ceo": tool_result.get("current_ceo", "Unknown"),
                "latest_news_summary": tool_result.get("latest_news_summary", "No update found."),
            }
        )
        return ModelResult(
            output=output,
            usage=usage,
        )


def _extract_tool_result(messages: list[ChatMessage]) -> dict[str, object] | None:
    for message in reversed(messages):
        if message.role != "tool":
            continue
        raw_payload = message.content
        if raw_payload.startswith("Result from search_web: "):
            raw_payload = raw_payload.removeprefix("Result from search_web: ")
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


async def main() -> None:
    db_path = Path("examples/.state_autonomous_agent.db")
    if db_path.exists():
        db_path.unlink()

    store = SQLiteStore(str(db_path))
    kernel = ArtanaKernel(
        store=store,
        model_port=ResearchModelPort(),
        middleware=ArtanaKernel.default_middleware_stack(),
        policy=KernelPolicy.enforced(),
    )

    @kernel.tool()
    async def search_web(query: str) -> str:
        query_lower = query.lower()
        if "acme" in query_lower and "news" in query_lower:
            return json.dumps(
                {
                    "company_name": "Acme Corp",
                    "current_ceo": "Jane Doe",
                    "latest_news_summary": "Acme announced a new AI partnership in Q1.",
                }
            )
        return json.dumps(
            {
                "company_name": "Acme Corp",
                "current_ceo": "Unavailable",
                "latest_news_summary": "No relevant facts found.",
            }
        )

    tenant = TenantContext(
        tenant_id="org_research",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )
    agent = AutonomousAgent(
        kernel=kernel,
        context_builder=ContextBuilder(progressive_skills=False),
    )

    try:
        report = await agent.run(
            run_id="research_run_01",
            tenant=tenant,
            model="research-agent-demo",
            system_prompt=(
                "You are a research agent. Use the search_web tool when needed and "
                "then return the final structured report."
            ),
            prompt=(
                "Find the current CEO and latest news for Acme Corp using available tools "
                "before you answer."
            ),
            output_schema=CompanyReport,
            max_iterations=5,
        )
        print("Research report:")
        print(report.model_dump_json(indent=2))
    finally:
        await kernel.close()
        if db_path.exists():
            db_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
