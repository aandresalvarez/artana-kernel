from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from artana.events import (
    ChatMessage,
    EventType,
    ModelCompletedPayload,
    ModelRequestedPayload,
    ToolRequestedPayload,
)
from artana.store import SQLiteStore


@pytest.mark.asyncio
async def test_append_and_get_events_for_run(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    try:
        first = await store.append_event(
            run_id="run_a",
            tenant_id="tenant_a",
            event_type=EventType.MODEL_REQUESTED,
            payload=ModelRequestedPayload(
                model="gpt-4o-mini",
                prompt="hello",
                messages=[ChatMessage(role="user", content="hello")],
            ),
        )
        second = await store.append_event(
            run_id="run_a",
            tenant_id="tenant_a",
            event_type=EventType.TOOL_REQUESTED,
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
            event_type=EventType.TOOL_REQUESTED,
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


@pytest.mark.asyncio
async def test_append_is_sequential_across_store_instances(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    first_store = SQLiteStore(str(database_path))
    second_store = SQLiteStore(str(database_path))

    async def append_batch(store: SQLiteStore, prefix: str) -> None:
        for index in range(20):
            await store.append_event(
                run_id="run_multi_instance",
                tenant_id="tenant_multi",
                event_type=EventType.TOOL_REQUESTED,
                payload=ToolRequestedPayload(
                    tool_name="noop",
                    arguments_json=f'{{"index":"{index}"}}',
                    idempotency_key=f"idemp-{prefix}-{index}",
                ),
            )

    try:
        await asyncio.gather(
            append_batch(first_store, "a"),
            append_batch(second_store, "b"),
        )
        events = await first_store.get_events_for_run("run_multi_instance")
        assert [event.seq for event in events] == list(range(1, 41))
    finally:
        await first_store.close()
        await second_store.close()


@pytest.mark.asyncio
async def test_get_model_cost_sum_for_run_aggregates_only_model_completed(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    try:
        await store.append_event(
            run_id="run_cost",
            tenant_id="tenant_cost",
            event_type=EventType.MODEL_COMPLETED,
            payload=ModelCompletedPayload(
                model="gpt-4o-mini",
                output_json='{"approved":true}',
                prompt_tokens=10,
                completion_tokens=5,
                cost_usd=0.04,
            ),
        )
        await store.append_event(
            run_id="run_cost",
            tenant_id="tenant_cost",
            event_type=EventType.TOOL_REQUESTED,
            payload=ToolRequestedPayload(
                tool_name="noop",
                arguments_json='{"x":"1"}',
                idempotency_key="idemp-cost-2",
            ),
        )
        await store.append_event(
            run_id="run_cost",
            tenant_id="tenant_cost",
            event_type=EventType.MODEL_COMPLETED,
            payload=ModelCompletedPayload(
                model="gpt-4o-mini",
                output_json='{"approved":false}',
                prompt_tokens=3,
                completion_tokens=2,
                cost_usd=0.06,
            ),
        )

        total = await store.get_model_cost_sum_for_run("run_cost")
        assert total == pytest.approx(0.10)
    finally:
        await store.close()
