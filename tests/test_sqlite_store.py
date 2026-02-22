from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from artana.events import ChatMessage, ModelRequestedPayload, ToolRequestedPayload
from artana.store import SQLiteStore


@pytest.mark.asyncio
async def test_append_and_get_events_for_run(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    try:
        first = await store.append_event(
            run_id="run_a",
            tenant_id="tenant_a",
            event_type="model_requested",
            payload=ModelRequestedPayload(
                model="gpt-4o-mini",
                prompt="hello",
                messages=[ChatMessage(role="user", content="hello")],
            ),
        )
        second = await store.append_event(
            run_id="run_a",
            tenant_id="tenant_a",
            event_type="tool_requested",
            payload=ToolRequestedPayload(
                tool_name="lookup_balance",
                arguments_json='{"account_id":"abc"}',
                idempotency_key="idemp-run_a-2",
            ),
        )

        events = await store.get_events_for_run("run_a")
        assert [event.seq for event in events] == [1, 2]
        assert events[0].event_id == first.event_id
        assert events[1].event_id == second.event_id
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_append_is_sequential_under_concurrency(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))

    async def append_tool_event(index: int) -> None:
        await store.append_event(
            run_id="run_concurrent",
            tenant_id="tenant_concurrent",
            event_type="tool_requested",
            payload=ToolRequestedPayload(
                tool_name="noop",
                arguments_json=f'{{"index":"{index}"}}',
                idempotency_key=f"idemp-run_concurrent-{index}",
            ),
        )

    try:
        await asyncio.gather(*(append_tool_event(index) for index in range(30)))
        events = await store.get_events_for_run("run_concurrent")
        assert [event.seq for event in events] == list(range(1, 31))
    finally:
        await store.close()
