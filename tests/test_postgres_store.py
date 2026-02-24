from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from artana.events import (
    ChatMessage,
    EventType,
    ModelCompletedPayload,
    ModelRequestedPayload,
    RunSummaryPayload,
    ToolRequestedPayload,
)
from artana.store import PostgresStore

_POSTGRES_DSN = os.getenv("ARTANA_TEST_POSTGRES_DSN")

pytestmark = pytest.mark.skipif(
    _POSTGRES_DSN is None,
    reason="Set ARTANA_TEST_POSTGRES_DSN to run Postgres store integration tests.",
)


@pytest.fixture
async def store() -> AsyncIterator[PostgresStore]:
    postgres_store = PostgresStore(_dsn_for_tests())
    try:
        yield postgres_store
    finally:
        await postgres_store.close()


def _dsn_for_tests() -> str:
    if _POSTGRES_DSN is None:
        raise RuntimeError("ARTANA_TEST_POSTGRES_DSN is required for Postgres tests.")
    return _POSTGRES_DSN


def _run_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@pytest.mark.asyncio
async def test_append_and_get_events_for_run(store: PostgresStore) -> None:
    run_id = _run_id("run_pg_append")

    first = await store.append_event(
        run_id=run_id,
        tenant_id="tenant_pg",
        event_type=EventType.MODEL_REQUESTED,
        payload=ModelRequestedPayload(
            model="gpt-4o-mini",
            prompt="hello",
            messages=[ChatMessage(role="user", content="hello")],
        ),
    )
    second = await store.append_event(
        run_id=run_id,
        tenant_id="tenant_pg",
        event_type=EventType.TOOL_REQUESTED,
        payload=ToolRequestedPayload(
            tool_name="lookup_balance",
            arguments_json='{"account_id":"abc"}',
            idempotency_key=f"idemp-{run_id}-2",
        ),
    )

    events = await store.get_events_for_run(run_id)
    assert [event.seq for event in events] == [1, 2]
    assert events[0].event_id == first.event_id
    assert events[1].event_id == second.event_id
    assert await store.verify_run_chain(run_id)


@pytest.mark.asyncio
async def test_append_is_sequential_under_concurrency(store: PostgresStore) -> None:
    run_id = _run_id("run_pg_concurrent")

    async def append_tool_event(index: int) -> None:
        await store.append_event(
            run_id=run_id,
            tenant_id="tenant_pg_concurrent",
            event_type=EventType.TOOL_REQUESTED,
            payload=ToolRequestedPayload(
                tool_name="noop",
                arguments_json=f'{{"index":"{index}"}}',
                idempotency_key=f"idemp-{run_id}-{index}",
            ),
        )

    await asyncio.gather(*(append_tool_event(index) for index in range(30)))
    events = await store.get_events_for_run(run_id)
    assert [event.seq for event in events] == list(range(1, 31))


@pytest.mark.asyncio
async def test_append_is_sequential_across_store_instances() -> None:
    run_id = _run_id("run_pg_multi")
    first_store = PostgresStore(_dsn_for_tests())
    second_store = PostgresStore(_dsn_for_tests())

    async def append_batch(store: PostgresStore, prefix: str) -> None:
        for index in range(20):
            await store.append_event(
                run_id=run_id,
                tenant_id="tenant_pg_multi",
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
        events = await first_store.get_events_for_run(run_id)
        assert [event.seq for event in events] == list(range(1, 41))
    finally:
        await first_store.close()
        await second_store.close()


@pytest.mark.asyncio
async def test_summary_and_cost_queries_match_expected_payloads(store: PostgresStore) -> None:
    run_id = _run_id("run_pg_summary_cost")

    observed: list[tuple[int, str]] = []

    async def on_event(event: object) -> None:
        seq = getattr(event, "seq")
        event_type = getattr(event, "event_type").value
        observed.append((seq, event_type))

    callback_store = PostgresStore(_dsn_for_tests(), on_event=on_event)
    try:
        await callback_store.append_event(
            run_id=run_id,
            tenant_id="tenant_pg_cost",
            event_type=EventType.MODEL_COMPLETED,
            payload=ModelCompletedPayload(
                model="gpt-4o-mini",
                output_json='{"approved":true}',
                prompt_tokens=10,
                completion_tokens=5,
                cost_usd=0.04,
            ),
        )
        await callback_store.append_event(
            run_id=run_id,
            tenant_id="tenant_pg_cost",
            event_type=EventType.RUN_SUMMARY,
            payload=RunSummaryPayload(
                summary_type="task_progress",
                summary_json='{"units":[{"id":"t1","state":"pending"}]}',
                step_key="task_progress_1",
            ),
        )
        await callback_store.append_event(
            run_id=run_id,
            tenant_id="tenant_pg_cost",
            event_type=EventType.MODEL_COMPLETED,
            payload=ModelCompletedPayload(
                model="gpt-4o-mini",
                output_json='{"approved":false}',
                prompt_tokens=3,
                completion_tokens=2,
                cost_usd=0.06,
            ),
        )
        await callback_store.append_event(
            run_id=run_id,
            tenant_id="tenant_pg_cost",
            event_type=EventType.RUN_SUMMARY,
            payload=RunSummaryPayload(
                summary_type="task_progress",
                summary_json='{"units":[{"id":"t1","state":"done"}]}',
                step_key="task_progress_2",
            ),
        )

        total = await callback_store.get_model_cost_sum_for_run(run_id)
        latest_summary = await callback_store.get_latest_run_summary(run_id, "task_progress")

        assert total == pytest.approx(0.10)
        assert latest_summary is not None
        assert latest_summary.step_key == "task_progress_2"
        assert latest_summary.summary_json == '{"units":[{"id":"t1","state":"done"}]}'
        assert observed == [
            (1, EventType.MODEL_COMPLETED.value),
            (2, EventType.RUN_SUMMARY.value),
            (3, EventType.MODEL_COMPLETED.value),
            (4, EventType.RUN_SUMMARY.value),
        ]
    finally:
        await callback_store.close()
