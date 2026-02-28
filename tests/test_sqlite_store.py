from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from artana.events import (
    ChatMessage,
    EventType,
    HarnessSleepPayload,
    ModelRequestedPayload,
    ModelTerminalPayload,
    PauseRequestedPayload,
    ResumeRequestedPayload,
    RunStartedPayload,
    RunSummaryPayload,
    ToolCompletedPayload,
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
async def test_get_model_cost_sum_for_run_aggregates_only_model_terminal(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    try:
        await store.append_event(
            run_id="run_cost",
            tenant_id="tenant_cost",
            event_type=EventType.MODEL_TERMINAL,
            payload=ModelTerminalPayload(
                outcome="completed",
                model="gpt-4o-mini",
                model_cycle_id="cycle_cost_1",
                source_model_requested_event_id="event_req_cost_1",
                elapsed_ms=1,
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
            event_type=EventType.MODEL_TERMINAL,
            payload=ModelTerminalPayload(
                outcome="completed",
                model="gpt-4o-mini",
                model_cycle_id="cycle_cost_2",
                source_model_requested_event_id="event_req_cost_2",
                elapsed_ms=1,
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


@pytest.mark.asyncio
async def test_get_latest_run_summary_returns_latest_by_summary_type(
    tmp_path: Path,
) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    try:
        await store.append_event(
            run_id="run_summary_lookup",
            tenant_id="tenant_summary",
            event_type=EventType.RUN_SUMMARY,
            payload=RunSummaryPayload(
                summary_type="task_progress",
                summary_json='{"units":[{"id":"t1","state":"pending"}]}',
                step_key="task_progress_1",
            ),
        )
        await store.append_event(
            run_id="run_summary_lookup",
            tenant_id="tenant_summary",
            event_type=EventType.RUN_SUMMARY,
            payload=RunSummaryPayload(
                summary_type="artifact::report",
                summary_json='{"url":"s3://bucket/report.json"}',
                step_key="artifact_1",
            ),
        )
        await store.append_event(
            run_id="run_summary_lookup",
            tenant_id="tenant_summary",
            event_type=EventType.RUN_SUMMARY,
            payload=RunSummaryPayload(
                summary_type="task_progress",
                summary_json='{"units":[{"id":"t1","state":"done"}]}',
                step_key="task_progress_2",
            ),
        )

        latest_task = await store.get_latest_run_summary(
            "run_summary_lookup",
            "task_progress",
        )
        latest_artifact = await store.get_latest_run_summary(
            "run_summary_lookup",
            "artifact::report",
        )
        missing = await store.get_latest_run_summary(
            "run_summary_lookup",
            "artifact::missing",
        )

        assert latest_task is not None
        assert latest_task.step_key == "task_progress_2"
        assert latest_task.summary_json == '{"units":[{"id":"t1","state":"done"}]}'

        assert latest_artifact is not None
        assert latest_artifact.step_key == "artifact_1"
        assert latest_artifact.summary_json == '{"url":"s3://bucket/report.json"}'
        assert missing is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_on_event_callback_receives_appended_events(tmp_path: Path) -> None:
    observed: list[tuple[int, str]] = []

    async def on_event(event: object) -> None:
        seq = getattr(event, "seq")
        event_type = getattr(event, "event_type").value
        observed.append((seq, event_type))

    store = SQLiteStore(str(tmp_path / "state.db"), on_event=on_event)
    try:
        await store.append_event(
            run_id="run_on_event",
            tenant_id="tenant_on_event",
            event_type=EventType.MODEL_REQUESTED,
            payload=ModelRequestedPayload(
                model="gpt-4o-mini",
                prompt="hello",
                messages=[ChatMessage(role="user", content="hello")],
            ),
        )
        await store.append_event(
            run_id="run_on_event",
            tenant_id="tenant_on_event",
            event_type=EventType.TOOL_REQUESTED,
            payload=ToolRequestedPayload(
                tool_name="noop",
                arguments_json='{"index":"2"}',
                idempotency_key="idemp-on-event-2",
            ),
        )

        assert observed == [
            (1, EventType.MODEL_REQUESTED.value),
            (2, EventType.TOOL_REQUESTED.value),
        ]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_tool_policy_columns_and_aggregates_are_queryable(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    store = SQLiteStore(str(database_path))
    try:
        request = await store.append_event(
            run_id="run_policy",
            tenant_id="tenant_policy",
            event_type=EventType.TOOL_REQUESTED,
            payload=ToolRequestedPayload(
                tool_name="send_invoice",
                arguments_json='{"billing_period":"2026-02"}',
                idempotency_key="idemp-policy-1",
                step_key="invoice_1",
                semantic_idempotency_key="send_invoice:tenant_policy:2026-02",
                amount_usd=42.5,
            ),
        )
        await store.append_event(
            run_id="run_policy",
            tenant_id="tenant_policy",
            event_type=EventType.TOOL_COMPLETED,
            payload=ToolCompletedPayload(
                tool_name="send_invoice",
                result_json='{"ok":true}',
                outcome="success",
                request_id=request.event_id,
            ),
        )

        count_for_run = await store.get_tool_request_count_for_run(
            run_id="run_policy",
            tool_name="send_invoice",
        )
        assert count_for_run == 1

        count_for_tenant = await store.get_tool_request_count_for_tenant_since(
            tenant_id="tenant_policy",
            tool_name="send_invoice",
            since=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        assert count_for_tenant == 1

        latest = await store.get_latest_tool_semantic_outcome(
            tenant_id="tenant_policy",
            tool_name="send_invoice",
            semantic_idempotency_key="send_invoice:tenant_policy:2026-02",
        )
        assert latest is not None
        assert latest.run_id == "run_policy"
        assert latest.outcome == "success"
        assert latest.request_step_key == "invoice_1"
        assert latest.request_arguments_json == '{"billing_period":"2026-02"}'

        await asyncio.sleep(0.001)
        second_request = await store.append_event(
            run_id="run_policy_second",
            tenant_id="tenant_policy",
            event_type=EventType.TOOL_REQUESTED,
            payload=ToolRequestedPayload(
                tool_name="send_invoice",
                arguments_json='{"billing_period":"2026-02"}',
                idempotency_key="idemp-policy-2",
                step_key="invoice_2",
                semantic_idempotency_key="send_invoice:tenant_policy:2026-02",
            ),
        )
        await store.append_event(
            run_id="run_policy_second",
            tenant_id="tenant_policy",
            event_type=EventType.TOOL_COMPLETED,
            payload=ToolCompletedPayload(
                tool_name="send_invoice",
                result_json='{"ok":false}',
                outcome="unknown_outcome",
                request_id=second_request.event_id,
            ),
        )
        latest_after_second = await store.get_latest_tool_semantic_outcome(
            tenant_id="tenant_policy",
            tool_name="send_invoice",
            semantic_idempotency_key="send_invoice:tenant_policy:2026-02",
        )
        assert latest_after_second is not None
        assert latest_after_second.run_id == "run_policy_second"
        assert latest_after_second.outcome == "unknown_outcome"
        assert latest_after_second.request_step_key == "invoice_2"

        with sqlite3.connect(database_path) as connection:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(kernel_events)")
            }
            assert {
                "tool_name",
                "tool_outcome",
                "tool_request_id",
                "tool_semantic_key",
                "tool_amount_usd",
            }.issubset(columns)
            indexes = {
                str(row[1]) for row in connection.execute("PRAGMA index_list(kernel_events)")
            }
            assert {
                "idx_kernel_events_tenant_tool_time",
                "idx_kernel_events_run_tool_seq",
                "idx_kernel_events_tool_semantic",
            }.issubset(indexes)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_run_state_snapshots_are_queryable(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    store = SQLiteStore(str(database_path))
    try:
        await store.append_event(
            run_id="run_snapshot_main",
            tenant_id="tenant_snapshot",
            event_type=EventType.RUN_STARTED,
            payload=RunStartedPayload(),
        )
        await store.append_event(
            run_id="run_snapshot_main",
            tenant_id="tenant_snapshot",
            event_type=EventType.PAUSE_REQUESTED,
            payload=PauseRequestedPayload(
                reason="approval needed",
                context_json='{"approval_key":"approval_1"}',
            ),
        )
        await store.append_event(
            run_id="run_snapshot_main",
            tenant_id="tenant_snapshot",
            event_type=EventType.RESUME_REQUESTED,
            payload=ResumeRequestedPayload(),
        )
        await store.append_event(
            run_id="run_snapshot_main",
            tenant_id="tenant_snapshot",
            event_type=EventType.MODEL_TERMINAL,
            payload=ModelTerminalPayload(
                outcome="completed",
                model="gpt-4o-mini",
                model_cycle_id="cycle_snapshot_1",
                source_model_requested_event_id="event_req_snapshot_1",
                elapsed_ms=1,
                output_json='{"ok":true}',
                prompt_tokens=3,
                completion_tokens=2,
                cost_usd=0.2,
            ),
        )
        await store.append_event(
            run_id="run_snapshot_done",
            tenant_id="tenant_snapshot",
            event_type=EventType.RUN_STARTED,
            payload=RunStartedPayload(),
        )
        await store.append_event(
            run_id="run_snapshot_done",
            tenant_id="tenant_snapshot",
            event_type=EventType.HARNESS_SLEEP,
            payload=HarnessSleepPayload(status="completed"),
        )

        snapshot = await store.get_run_state_snapshot(run_id="run_snapshot_main")
        assert snapshot is not None
        assert snapshot.status == "active"
        assert snapshot.blocked_on is None
        assert snapshot.model_cost_total == pytest.approx(0.2)
        assert snapshot.last_event_type == EventType.MODEL_TERMINAL.value

        active = await store.list_run_state_snapshots(
            tenant_id="tenant_snapshot",
            status="active",
        )
        assert [item.run_id for item in active] == ["run_snapshot_main"]

        completed = await store.list_run_state_snapshots(
            tenant_id="tenant_snapshot",
            status="completed",
        )
        assert [item.run_id for item in completed] == ["run_snapshot_done"]

        recent = await store.list_run_state_snapshots(
            tenant_id="tenant_snapshot",
            since=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        assert {item.run_id for item in recent} == {"run_snapshot_main", "run_snapshot_done"}

        with sqlite3.connect(database_path) as connection:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(run_state_snapshots)")
            }
            assert {
                "run_id",
                "tenant_id",
                "last_event_seq",
                "last_event_type",
                "updated_at",
                "status",
                "blocked_on",
                "failure_reason",
                "last_step_key",
                "drift_count",
                "last_stage",
                "last_tool",
                "model_cost_total",
                "open_pause_count",
                "explain_status",
                "explain_failure_reason",
                "explain_failure_step",
            }.issubset(columns)
            indexes = {
                str(row[1])
                for row in connection.execute("PRAGMA index_list(run_state_snapshots)")
            }
            assert {
                "idx_run_state_snapshots_tenant_updated",
                "idx_run_state_snapshots_tenant_status_updated",
            }.issubset(indexes)
    finally:
        await store.close()
