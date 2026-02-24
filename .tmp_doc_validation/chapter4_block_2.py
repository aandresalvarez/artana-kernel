import asyncio
import json

from pydantic import BaseModel

from artana.events import ChatMessage
from artana.kernel import ArtanaKernel, ModelInput
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage, ToolCall
from artana.store import SQLiteStore


class DebateResponse(BaseModel):
    text: str


class StoreArgumentArgs(BaseModel):
    value: str


class DebateModelPort:
    async def complete(self, request: ModelRequest[DebateResponse]) -> ModelResult[DebateResponse]:
        last = request.messages[-1].content
        output = request.output_schema.model_validate({"text": f"Reply to: {last}"})

        tool_calls = ()
        if "store this" in last.lower():
            tool_calls = (
                ToolCall(
                    tool_name="store_argument",
                    arguments_json='{"value":"important"}',
                    tool_call_id="call_1",
                ),
            )

        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=10, completion_tokens=5, cost_usd=0.0),
            tool_calls=tool_calls,
        )


async def main():
    kernel = ArtanaKernel(
        store=SQLiteStore("chapter4_step2.db"),
        model_port=DebateModelPort(),
    )

    @kernel.tool()
    async def store_argument(value: str) -> str:
        return json.dumps({"stored": value})

    tenant = TenantContext(
        tenant_id="research",
        capabilities=frozenset(),
        budget_usd_limit=5.0,
    )

    run_id = "debate_run"
    await kernel.start_run(tenant=tenant, run_id=run_id)

    transcript = [ChatMessage(role="system", content="You are debating.")]

    result = await kernel.step_model(
        run_id=run_id,
        tenant=tenant,
        model="demo-model",
        input=ModelInput.from_messages(
            transcript + [ChatMessage(role="user", content="Store this idea")]
        ),
        output_schema=DebateResponse,
        step_key="turn_1",
    )

    for tool in result.tool_calls:
        tool_result = await kernel.step_tool(
            run_id=run_id,
            tenant=tenant,
            tool_name=tool.tool_name,
            arguments=StoreArgumentArgs(value="important"),
            step_key="tool_1",
        )
        print(tool_result.result_json)

    await kernel.close()


asyncio.run(main())