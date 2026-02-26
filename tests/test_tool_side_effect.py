from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import ArtanaKernel
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.ports.tool import ToolExecutionContext
from artana.store import SQLiteStore

OutputT = TypeVar("OutputT", bound=BaseModel)


class DummyModelPort:
    async def complete(self, request: ModelRequest[OutputT]) -> ModelResult[OutputT]:
        output = request.output_schema.model_validate({})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0),
        )


@pytest.mark.asyncio
async def test_kernel_tool_side_effect_requires_artana_context(tmp_path: Path) -> None:
    kernel = ArtanaKernel(
        store=SQLiteStore(str(tmp_path / "state.db")),
        model_port=DummyModelPort(),
    )
    try:
        with pytest.raises(ValueError, match="side_effect=True"):

            @kernel.tool(side_effect=True)
            async def unsafe_mutation(amount: int) -> str:
                return f'{{"amount":{amount}}}'
    finally:
        await kernel.close()


@pytest.mark.asyncio
async def test_kernel_tool_side_effect_accepts_valid_signature(tmp_path: Path) -> None:
    kernel = ArtanaKernel(
        store=SQLiteStore(str(tmp_path / "state.db")),
        model_port=DummyModelPort(),
    )
    try:

        @kernel.tool(side_effect=True)
        async def safe_mutation(amount: int, artana_context: ToolExecutionContext) -> str:
            return (
                '{"amount":'
                f"{amount}"
                ',"idempotency_key":"'
                f"{artana_context.idempotency_key}"
                '"}'
            )

        tools = kernel.list_registered_tools()
        assert any(tool.name == "safe_mutation" for tool in tools)
    finally:
        await kernel.close()
