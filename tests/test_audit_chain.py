from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from artana.canonicalization import canonical_json_dumps
from artana.events import (
    ChatMessage,
    EventType,
    ModelRequestedPayload,
    ToolRequestedPayload,
)
from artana.store import SQLiteStore


@pytest.mark.asyncio
async def test_event_hash_chain_is_persisted_and_verifiable(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    store = SQLiteStore(str(database_path))
    try:
        await store.append_event(
            run_id="run_audit",
            tenant_id="tenant_audit",
            event_type=EventType.MODEL_REQUESTED,
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
            event_type=EventType.TOOL_REQUESTED,
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
            event_type=EventType.MODEL_REQUESTED,
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
            event_type=EventType.TOOL_REQUESTED,
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


def _legacy_event_hash(
    *,
    event_id: str,
    run_id: str,
    tenant_id: str,
    seq: int,
    event_type: EventType,
    prev_event_hash: str | None,
    timestamp_iso: str,
    parent_step_key: str | None,
    payload_json: str,
) -> str:
    hash_fields = [
        event_id,
        run_id,
        tenant_id,
        str(seq),
        event_type.value,
        prev_event_hash or "",
        timestamp_iso,
    ]
    if parent_step_key is not None:
        hash_fields.append(parent_step_key)
    hash_fields.append(payload_json)
    return hashlib.sha256("|".join(hash_fields).encode("utf-8")).hexdigest()


@pytest.mark.asyncio
async def test_event_hash_chain_verifies_legacy_tool_requested_hash(tmp_path: Path) -> None:
    database_path = tmp_path / "state_legacy.db"
    store = SQLiteStore(str(database_path))
    try:
        await store.append_event(
            run_id="run_legacy_hash",
            tenant_id="tenant_legacy",
            event_type=EventType.MODEL_REQUESTED,
            payload=ModelRequestedPayload(
                model="gpt-4o-mini",
                prompt="hello",
                messages=[ChatMessage(role="user", content="hello")],
                allowed_tools=[],
            ),
        )
        await store.append_event(
            run_id="run_legacy_hash",
            tenant_id="tenant_legacy",
            event_type=EventType.TOOL_REQUESTED,
            payload=ToolRequestedPayload(
                tool_name="noop",
                arguments_json='{"x":"1"}',
                idempotency_key="idemp-run_legacy_hash-2",
            ),
        )
        events = await store.get_events_for_run("run_legacy_hash")
    finally:
        await store.close()

    assert len(events) == 2
    second_event = events[1]
    payload = second_event.payload
    assert isinstance(payload, ToolRequestedPayload)
    legacy_payload_json = canonical_json_dumps(
        {
            "kind": "tool_requested",
            "tool_name": payload.tool_name,
            "arguments_json": payload.arguments_json,
            "idempotency_key": payload.idempotency_key,
            "tool_version": payload.tool_version,
            "schema_version": payload.schema_version,
            "step_key": payload.step_key,
        }
    )
    legacy_hash = _legacy_event_hash(
        event_id=second_event.event_id,
        run_id=second_event.run_id,
        tenant_id=second_event.tenant_id,
        seq=second_event.seq,
        event_type=second_event.event_type,
        prev_event_hash=second_event.prev_event_hash,
        timestamp_iso=second_event.timestamp.isoformat(),
        parent_step_key=second_event.parent_step_key,
        payload_json=legacy_payload_json,
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            UPDATE kernel_events
            SET payload_json = ?, event_hash = ?
            WHERE run_id = ? AND seq = ?
            """,
            (legacy_payload_json, legacy_hash, "run_legacy_hash", 2),
        )
        connection.commit()

    legacy_store = SQLiteStore(str(database_path))
    try:
        assert await legacy_store.verify_run_chain("run_legacy_hash") is True
    finally:
        await legacy_store.close()
