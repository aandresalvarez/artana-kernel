from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from artana.events import ChatMessage, ModelRequestedPayload, ToolRequestedPayload
from artana.store import SQLiteStore


@pytest.mark.asyncio
async def test_event_hash_chain_is_persisted_and_verifiable(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    store = SQLiteStore(str(database_path))
    try:
        await store.append_event(
            run_id="run_audit",
            tenant_id="tenant_audit",
            event_type="model_requested",
            payload=ModelRequestedPayload(
                model="gpt-4o-mini",
                prompt="hello",
                messages=[ChatMessage(role="user", content="hello")],
                allowed_tools=[],
            ),
        )
        await store.append_event(
            run_id="run_audit",
            tenant_id="tenant_audit",
            event_type="tool_requested",
            payload=ToolRequestedPayload(
                tool_name="noop",
                arguments_json='{"x":"1"}',
                idempotency_key="idemp-run_audit-2",
            ),
        )
        events = await store.get_events_for_run("run_audit")
        assert len(events) == 2
        assert events[0].prev_event_hash is None
        assert events[1].prev_event_hash == events[0].event_hash
        assert await store.verify_run_chain("run_audit") is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_event_hash_chain_detects_tampering(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    store = SQLiteStore(str(database_path))
    try:
        await store.append_event(
            run_id="run_tamper",
            tenant_id="tenant_tamper",
            event_type="model_requested",
            payload=ModelRequestedPayload(
                model="gpt-4o-mini",
                prompt="hello",
                messages=[ChatMessage(role="user", content="hello")],
                allowed_tools=[],
            ),
        )
        await store.append_event(
            run_id="run_tamper",
            tenant_id="tenant_tamper",
            event_type="tool_requested",
            payload=ToolRequestedPayload(
                tool_name="noop",
                arguments_json='{"x":"1"}',
                idempotency_key="idemp-run_tamper-2",
            ),
        )
        assert await store.verify_run_chain("run_tamper") is True
    finally:
        await store.close()

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            UPDATE kernel_events
            SET event_hash = 'tampered_hash'
            WHERE run_id = ? AND seq = ?
            """,
            ("run_tamper", 2),
        )
        connection.commit()

    tampered_store = SQLiteStore(str(database_path))
    try:
        assert await tampered_store.verify_run_chain("run_tamper") is False
    finally:
        await tampered_store.close()
