from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

from artana import KernelModelClient
from artana.events import ModelRequestedPayload, compute_allowed_tools_hash
from artana.kernel import ArtanaKernel, ReplayConsistencyError
from artana.models import TenantContext
from artana.ports.model import ModelRequest, ModelResult, ModelUsage
from artana.store import SQLiteStore

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Decision(BaseModel):
    approved: bool
    reason: str


class CountingModelPort:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, request: ModelRequest[OutputModelT]
    ) -> ModelResult[OutputModelT]:
        self.calls += 1
        output = request.output_schema.model_validate({"approved": True, "reason": "ok"})
        return ModelResult(
            output=output,
            usage=ModelUsage(prompt_tokens=2, completion_tokens=1, cost_usd=0.01),
        )


def _tenant() -> TenantContext:
    return TenantContext(
        tenant_id="org_replay",
        capabilities=frozenset(),
        budget_usd_limit=1.0,
    )


@pytest.mark.asyncio
async def test_model_replay_ignores_tool_registration_order(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"

    first_store = SQLiteStore(str(database_path))
    first_model = CountingModelPort()
    first_kernel = ArtanaKernel(store=first_store, model_port=first_model)

    def register_first_tools() -> None:
        @first_kernel.tool()
        async def z_tool() -> str:
            return '{"ok":true}'

        @first_kernel.tool()
        async def a_tool() -> str:
            return '{"ok":true}'

    register_first_tools()

    try:
        first_result = await KernelModelClient(kernel=first_kernel).step(
            run_id="run_tool_order",
            prompt="check tools",
            model="gpt-4o-mini",
            tenant=_tenant(),
            output_schema=Decision,
        )
        assert first_result.replayed is False
        assert first_model.calls == 1

        events = await first_store.get_events_for_run("run_tool_order")
        payload = events[1].payload
        assert isinstance(payload, ModelRequestedPayload)
        assert payload.allowed_tools == ["a_tool", "z_tool"]
        signature_tokens = [
            (
                f"{signature.name}|{signature.tool_version}|"
                f"{signature.schema_version}|{signature.schema_hash}"
            )
            for signature in payload.allowed_tool_signatures
        ]
        assert payload.allowed_tools_hash == compute_allowed_tools_hash(signature_tokens)
    finally:
        await first_kernel.close()

    second_store = SQLiteStore(str(database_path))
    second_model = CountingModelPort()
    second_kernel = ArtanaKernel(store=second_store, model_port=second_model)

    def register_second_tools() -> None:
        @second_kernel.tool()
        async def a_tool() -> str:
            return '{"ok":true}'

        @second_kernel.tool()
        async def z_tool() -> str:
            return '{"ok":true}'

    register_second_tools()

    try:
        replayed_result = await KernelModelClient(kernel=second_kernel).step(
            run_id="run_tool_order",
            prompt="check tools",
            model="gpt-4o-mini",
            tenant=_tenant(),
            output_schema=Decision,
        )
        assert replayed_result.replayed is True
        assert second_model.calls == 0
    finally:
        await second_kernel.close()


@pytest.mark.asyncio
async def test_model_replay_rejects_tool_set_changes(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    tenant = _tenant()

    first_store = SQLiteStore(str(database_path))
    first_model = CountingModelPort()
    first_kernel = ArtanaKernel(store=first_store, model_port=first_model)

    def register_first_tools() -> None:
        @first_kernel.tool()
        async def only_tool() -> str:
            return '{"ok":true}'

    register_first_tools()

    try:
        await KernelModelClient(kernel=first_kernel).step(
            run_id="run_tool_set_change",
            prompt="check tools",
            model="gpt-4o-mini",
            tenant=tenant,
            output_schema=Decision,
        )
        assert first_model.calls == 1
    finally:
        await first_kernel.close()

    second_store = SQLiteStore(str(database_path))
    second_model = CountingModelPort()
    second_kernel = ArtanaKernel(store=second_store, model_port=second_model)

    def register_second_tools() -> None:
        @second_kernel.tool()
        async def only_tool() -> str:
            return '{"ok":true}'

        @second_kernel.tool()
        async def new_tool() -> str:
            return '{"ok":true}'

    register_second_tools()

    try:
        with pytest.raises(ReplayConsistencyError, match="changed allowed tool"):
            await KernelModelClient(kernel=second_kernel).step(
                run_id="run_tool_set_change",
                prompt="check tools",
                model="gpt-4o-mini",
                tenant=tenant,
                output_schema=Decision,
            )
        assert second_model.calls == 0
    finally:
        await second_kernel.close()
