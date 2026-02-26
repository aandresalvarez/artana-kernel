from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import aiosqlite
from pydantic import TypeAdapter

from artana.events import (
    EventPayload,
    EventType,
    KernelEvent,
    RunSummaryPayload,
    ToolCompletedPayload,
    ToolRequestedPayload,
    compute_event_hash,
)
from artana.store.base import (
    EventStore,
    RunLeaseRecord,
    RunStateLifecycleStatus,
    RunStateSnapshotRecord,
    ToolSemanticOutcomeRecord,
)
from artana.store.snapshot_state import (
    apply_event_to_run_state_snapshot,
    initialize_run_state_snapshot,
)

_PAYLOAD_ADAPTER: TypeAdapter[EventPayload] = TypeAdapter(EventPayload)


class SQLiteStore(EventStore):
    def __init__(
        self,
        database_path: str,
        *,
        busy_timeout_ms: int = 5000,
        max_retry_attempts: int = 5,
        retry_backoff_seconds: float = 0.02,
        on_event: Callable[[KernelEvent], Awaitable[None]] | None = None,
    ) -> None:
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be > 0")
        if max_retry_attempts <= 0:
            raise ValueError("max_retry_attempts must be > 0")
        if retry_backoff_seconds <= 0:
            raise ValueError("retry_backoff_seconds must be > 0")

        self._database_path = Path(database_path)
        self._busy_timeout_ms = busy_timeout_ms
        self._max_retry_attempts = max_retry_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._connection: aiosqlite.Connection | None = None
        self._connection_lock = asyncio.Lock()
        self._append_lock = asyncio.Lock()
        self._lease_lock = asyncio.Lock()
        self._on_event = on_event

    async def append_event(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
        parent_step_key: str | None = None,
    ) -> KernelEvent:
        connection = await self._ensure_connection()

        async with self._append_lock:
            for attempt in range(self._max_retry_attempts):
                try:
                    event = await self._append_event_once(
                        connection=connection,
                        run_id=run_id,
                        tenant_id=tenant_id,
                        event_type=event_type,
                        payload=payload,
                        parent_step_key=parent_step_key,
                    )
                    if self._on_event is not None:
                        await self._on_event(event)
                    return event
                except aiosqlite.IntegrityError as exc:
                    if not _is_run_seq_conflict_error(exc):
                        raise
                    if attempt >= self._max_retry_attempts - 1:
                        raise
                except aiosqlite.OperationalError as exc:
                    if not _is_locked_error(exc):
                        raise
                    if attempt >= self._max_retry_attempts - 1:
                        raise
                await asyncio.sleep(self._retry_backoff(attempt))
        raise RuntimeError(
            "Failed to append event due to repeated SQLite write contention."
        )

    async def get_events_for_run(self, run_id: str) -> list[KernelEvent]:
        connection = await self._ensure_connection()
        cursor = await connection.execute(
            """
            SELECT run_id, seq, event_id, tenant_id, event_type, prev_event_hash,
                   event_hash, parent_step_key, timestamp, payload_json
            FROM kernel_events
            WHERE run_id = ?
            ORDER BY seq ASC
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()

        events: list[KernelEvent] = []
        for row in rows:
            run_id_raw = row["run_id"]
            seq_raw = row["seq"]
            event_id_raw = row["event_id"]
            tenant_id_raw = row["tenant_id"]
            event_type_raw = row["event_type"]
            prev_event_hash_raw = row["prev_event_hash"]
            event_hash_raw = row["event_hash"]
            parent_step_key_raw = row["parent_step_key"]
            timestamp_raw = row["timestamp"]
            payload_json_raw = row["payload_json"]

            if not isinstance(run_id_raw, str):
                raise TypeError(f"Invalid run_id row type: {type(run_id_raw)!r}")
            if not isinstance(seq_raw, int):
                raise TypeError(f"Invalid seq row type: {type(seq_raw)!r}")
            if not isinstance(event_id_raw, str):
                raise TypeError(f"Invalid event_id row type: {type(event_id_raw)!r}")
            if not isinstance(tenant_id_raw, str):
                raise TypeError(f"Invalid tenant_id row type: {type(tenant_id_raw)!r}")
            if not isinstance(event_type_raw, str):
                raise TypeError(
                    f"Invalid event_type row type: {type(event_type_raw)!r}"
                )
            if prev_event_hash_raw is not None and not isinstance(prev_event_hash_raw, str):
                raise TypeError(
                    f"Invalid prev_event_hash row type: {type(prev_event_hash_raw)!r}"
                )
            if not isinstance(event_hash_raw, str):
                raise TypeError(
                    f"Invalid event_hash row type: {type(event_hash_raw)!r}"
                )
            if parent_step_key_raw is not None and not isinstance(
                parent_step_key_raw, str
            ):
                raise TypeError(
                    "Invalid parent_step_key row type: "
                    f"{type(parent_step_key_raw)!r}"
                )
            if not isinstance(timestamp_raw, str):
                raise TypeError(
                    f"Invalid timestamp row type: {type(timestamp_raw)!r}"
                )
            if not isinstance(payload_json_raw, str):
                raise TypeError(
                    f"Invalid payload_json row type: {type(payload_json_raw)!r}"
                )
            try:
                event_type = EventType(event_type_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown event_type in store: {event_type_raw!r}"
                ) from exc

            payload_dict_raw = json.loads(payload_json_raw)
            if not isinstance(payload_dict_raw, dict):
                raise TypeError("Stored payload_json did not decode to an object.")
            payload = _PAYLOAD_ADAPTER.validate_python(payload_dict_raw)

            events.append(
                KernelEvent(
                    event_id=event_id_raw,
                    run_id=run_id_raw,
                    tenant_id=tenant_id_raw,
                    seq=seq_raw,
                    event_type=event_type,
                    prev_event_hash=prev_event_hash_raw,
                    event_hash=event_hash_raw,
                    parent_step_key=parent_step_key_raw,
                    timestamp=datetime.fromisoformat(timestamp_raw),
                    payload=payload,
                )
            )
        return events

    async def get_latest_run_summary(
        self,
        run_id: str,
        summary_type: str,
    ) -> RunSummaryPayload | None:
        connection = await self._ensure_connection()
        try:
            cursor = await connection.execute(
                """
                SELECT payload_json
                FROM kernel_events
                WHERE run_id = ?
                  AND event_type = ?
                  AND json_extract(payload_json, '$.summary_type') = ?
                ORDER BY seq DESC
                LIMIT 1
                """,
                (run_id, EventType.RUN_SUMMARY.value, summary_type),
            )
        except aiosqlite.OperationalError as exc:
            if _is_missing_json_extract_error(exc):
                return await self._latest_run_summary_via_events(
                    run_id=run_id,
                    summary_type=summary_type,
                )
            raise

        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None

        payload_json_raw = row["payload_json"]
        if not isinstance(payload_json_raw, str):
            raise TypeError(
                "Invalid payload_json row type for latest run summary lookup: "
                f"{type(payload_json_raw)!r}"
            )
        payload_dict_raw = json.loads(payload_json_raw)
        if not isinstance(payload_dict_raw, dict):
            raise TypeError("Stored payload_json did not decode to an object.")
        payload = _PAYLOAD_ADAPTER.validate_python(payload_dict_raw)
        if not isinstance(payload, RunSummaryPayload):
            raise TypeError(
                "Latest run summary lookup returned non-summary payload "
                f"{type(payload)!r}."
            )
        if payload.summary_type != summary_type:
            raise TypeError(
                "Latest run summary lookup returned unexpected summary_type "
                f"{payload.summary_type!r}; expected {summary_type!r}."
            )
        return payload

    async def get_model_cost_sum_for_run(self, run_id: str) -> float:
        connection = await self._ensure_connection()
        try:
            cursor = await connection.execute(
                """
                SELECT COALESCE(
                    SUM(CAST(json_extract(payload_json, '$.cost_usd') AS REAL)),
                    0.0
                ) AS total_cost
                FROM kernel_events
                WHERE run_id = ?
                  AND event_type = ?
                """,
                (run_id, EventType.MODEL_COMPLETED.value),
            )
        except aiosqlite.OperationalError as exc:
            if _is_missing_json_extract_error(exc):
                return await self._sum_model_cost_via_events(run_id=run_id)
            raise

        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return 0.0

        total_cost_obj: object = row["total_cost"]
        if total_cost_obj is None:
            return 0.0
        if isinstance(total_cost_obj, int):
            return float(total_cost_obj)
        if isinstance(total_cost_obj, float):
            return total_cost_obj
        raise TypeError(
            f"Invalid total_cost row type for model cost aggregate: {type(total_cost_obj)!r}"
        )

    async def get_tool_request_count_for_run(self, *, run_id: str, tool_name: str) -> int:
        connection = await self._ensure_connection()
        cursor = await connection.execute(
            """
            SELECT COUNT(*) AS total_count
            FROM kernel_events
            WHERE run_id = ?
              AND event_type = ?
              AND tool_name = ?
            """,
            (run_id, EventType.TOOL_REQUESTED.value, tool_name),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return 0
        return _coerce_count(row["total_count"])

    async def get_tool_request_count_for_tenant_since(
        self,
        *,
        tenant_id: str,
        tool_name: str,
        since: datetime,
    ) -> int:
        connection = await self._ensure_connection()
        cursor = await connection.execute(
            """
            SELECT COUNT(*) AS total_count
            FROM kernel_events
            WHERE tenant_id = ?
              AND event_type = ?
              AND tool_name = ?
              AND timestamp >= ?
            """,
            (
                tenant_id,
                EventType.TOOL_REQUESTED.value,
                tool_name,
                since.isoformat(),
            ),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return 0
        return _coerce_count(row["total_count"])

    async def get_latest_tool_semantic_outcome(
        self,
        *,
        tenant_id: str,
        tool_name: str,
        semantic_idempotency_key: str,
    ) -> ToolSemanticOutcomeRecord | None:
        connection = await self._ensure_connection()
        cursor = await connection.execute(
            """
            SELECT
                c.run_id AS run_id,
                c.tool_request_id AS request_id,
                c.tool_outcome AS outcome,
                r.payload_json AS request_payload_json
            FROM kernel_events c
            JOIN kernel_events r
                ON c.tool_request_id = r.event_id
            WHERE c.event_type = ?
              AND r.event_type = ?
              AND r.tenant_id = ?
              AND r.tool_name = ?
              AND r.tool_semantic_key = ?
            ORDER BY c.timestamp DESC, c.seq DESC
            LIMIT 1
            """,
            (
                EventType.TOOL_COMPLETED.value,
                EventType.TOOL_REQUESTED.value,
                tenant_id,
                tool_name,
                semantic_idempotency_key,
            ),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None

        run_id_obj: object = row["run_id"]
        request_id_obj: object = row["request_id"]
        outcome_obj: object = row["outcome"]
        request_payload_json_obj: object = row["request_payload_json"]
        if not isinstance(run_id_obj, str):
            raise TypeError(f"Invalid run_id row type: {type(run_id_obj)!r}")
        if not isinstance(request_id_obj, str):
            raise TypeError(f"Invalid request_id row type: {type(request_id_obj)!r}")
        if not isinstance(outcome_obj, str):
            raise TypeError(f"Invalid outcome row type: {type(outcome_obj)!r}")
        if not isinstance(request_payload_json_obj, str):
            raise TypeError(
                "Invalid request_payload_json row type: "
                f"{type(request_payload_json_obj)!r}"
            )
        request_payload = _load_request_payload(request_payload_json_obj)
        return ToolSemanticOutcomeRecord(
            run_id=run_id_obj,
            request_id=request_id_obj,
            outcome=outcome_obj,
            request_step_key=request_payload.step_key,
            request_arguments_json=request_payload.arguments_json,
        )

    async def list_run_ids(
        self,
        *,
        tenant_id: str | None = None,
        since: datetime | None = None,
    ) -> list[str]:
        connection = await self._ensure_connection()
        where_clauses: list[str] = []
        parameters: list[object] = []
        if tenant_id is not None:
            where_clauses.append("tenant_id = ?")
            parameters.append(tenant_id)
        if since is not None:
            where_clauses.append("timestamp >= ?")
            parameters.append(since.isoformat())
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
        cursor = await connection.execute(
            f"""
            SELECT run_id, MAX(timestamp) AS latest_timestamp
            FROM kernel_events
            {where_sql}
            GROUP BY run_id
            ORDER BY latest_timestamp DESC
            """,
            tuple(parameters),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        run_ids: list[str] = []
        for row in rows:
            run_id_obj: object = row["run_id"]
            if not isinstance(run_id_obj, str):
                raise TypeError(f"Invalid run_id row type: {type(run_id_obj)!r}")
            run_ids.append(run_id_obj)
        return run_ids

    async def get_run_state_snapshot(
        self,
        *,
        run_id: str,
    ) -> RunStateSnapshotRecord | None:
        connection = await self._ensure_connection()
        cursor = await connection.execute(
            """
            SELECT
                run_id,
                tenant_id,
                last_event_seq,
                last_event_type,
                updated_at,
                status,
                blocked_on,
                failure_reason,
                last_step_key,
                drift_count,
                last_stage,
                last_tool,
                model_cost_total,
                open_pause_count,
                explain_status,
                explain_failure_reason,
                explain_failure_step
            FROM run_state_snapshots
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return _snapshot_from_row(row)

    async def list_run_state_snapshots(
        self,
        *,
        tenant_id: str,
        since: datetime | None = None,
        status: RunStateLifecycleStatus | None = None,
    ) -> list[RunStateSnapshotRecord]:
        connection = await self._ensure_connection()
        where_clauses = ["tenant_id = ?"]
        parameters: list[object] = [tenant_id]
        if since is not None:
            where_clauses.append("updated_at >= ?")
            parameters.append(since.isoformat())
        if status is not None:
            where_clauses.append("status = ?")
            parameters.append(status)
        where_sql = " AND ".join(where_clauses)
        cursor = await connection.execute(
            f"""
            SELECT
                run_id,
                tenant_id,
                last_event_seq,
                last_event_type,
                updated_at,
                status,
                blocked_on,
                failure_reason,
                last_step_key,
                drift_count,
                last_stage,
                last_tool,
                model_cost_total,
                open_pause_count,
                explain_status,
                explain_failure_reason,
                explain_failure_step
            FROM run_state_snapshots
            WHERE {where_sql}
            ORDER BY updated_at DESC
            """,
            tuple(parameters),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_snapshot_from_row(row) for row in rows]

    async def stream_events(
        self,
        run_id: str,
        *,
        since_seq: int = 0,
        follow: bool = False,
        poll_interval_seconds: float = 0.5,
        idle_timeout_seconds: float | None = None,
    ) -> AsyncIterator[KernelEvent]:
        if since_seq < 0:
            raise ValueError("since_seq must be >= 0.")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0.")
        if idle_timeout_seconds is not None and idle_timeout_seconds <= 0:
            raise ValueError("idle_timeout_seconds must be > 0 when provided.")

        last_seq = since_seq
        idle_started_at = datetime.now(timezone.utc)
        while True:
            events = await self.get_events_for_run(run_id)
            emitted = False
            for event in events:
                if event.seq <= last_seq:
                    continue
                emitted = True
                last_seq = event.seq
                yield event
            if not follow:
                return
            if emitted:
                idle_started_at = datetime.now(timezone.utc)
            elif idle_timeout_seconds is not None:
                idle_elapsed = datetime.now(timezone.utc) - idle_started_at
                if idle_elapsed >= timedelta(seconds=idle_timeout_seconds):
                    return
            await asyncio.sleep(poll_interval_seconds)

    async def acquire_run_lease(
        self,
        *,
        run_id: str,
        worker_id: str,
        ttl_seconds: int,
    ) -> bool:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0.")
        connection = await self._ensure_connection()
        now = datetime.now(timezone.utc)
        lease_expires_at = now + timedelta(seconds=ttl_seconds)
        now_iso = now.isoformat()
        expires_iso = lease_expires_at.isoformat()
        async with self._lease_lock:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await connection.execute(
                    """
                    INSERT INTO run_leases (run_id, worker_id, lease_expires_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        worker_id = excluded.worker_id,
                        lease_expires_at = excluded.lease_expires_at,
                        updated_at = excluded.updated_at
                    WHERE run_leases.worker_id = excluded.worker_id
                       OR run_leases.lease_expires_at <= excluded.updated_at
                    """,
                    (run_id, worker_id, expires_iso, now_iso),
                )
                cursor = await connection.execute(
                    "SELECT worker_id, lease_expires_at FROM run_leases WHERE run_id = ?",
                    (run_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                await connection.commit()
            except Exception:
                await _rollback_quietly(connection)
                raise
        if row is None:
            return False
        row_worker_obj: object = row["worker_id"]
        row_expiry_obj: object = row["lease_expires_at"]
        if not isinstance(row_worker_obj, str):
            raise TypeError(f"Invalid worker_id row type: {type(row_worker_obj)!r}")
        if not isinstance(row_expiry_obj, str):
            raise TypeError(f"Invalid lease_expires_at row type: {type(row_expiry_obj)!r}")
        return row_worker_obj == worker_id and row_expiry_obj == expires_iso

    async def renew_run_lease(
        self,
        *,
        run_id: str,
        worker_id: str,
        ttl_seconds: int,
    ) -> bool:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0.")
        connection = await self._ensure_connection()
        now = datetime.now(timezone.utc)
        lease_expires_at = now + timedelta(seconds=ttl_seconds)
        cursor = await connection.execute(
            """
            UPDATE run_leases
            SET lease_expires_at = ?, updated_at = ?
            WHERE run_id = ?
              AND worker_id = ?
              AND lease_expires_at > ?
            """,
            (
                lease_expires_at.isoformat(),
                now.isoformat(),
                run_id,
                worker_id,
                now.isoformat(),
            ),
        )
        await cursor.close()
        changes_cursor = await connection.execute("SELECT changes() AS change_count")
        changes_row = await changes_cursor.fetchone()
        await changes_cursor.close()
        await connection.commit()
        if changes_row is None:
            return False
        return _coerce_count(changes_row["change_count"]) > 0

    async def release_run_lease(
        self,
        *,
        run_id: str,
        worker_id: str,
    ) -> bool:
        connection = await self._ensure_connection()
        cursor = await connection.execute(
            """
            DELETE FROM run_leases
            WHERE run_id = ?
              AND worker_id = ?
            """,
            (run_id, worker_id),
        )
        await cursor.close()
        changes_cursor = await connection.execute("SELECT changes() AS change_count")
        changes_row = await changes_cursor.fetchone()
        await changes_cursor.close()
        await connection.commit()
        if changes_row is None:
            return False
        return _coerce_count(changes_row["change_count"]) > 0

    async def get_run_lease(self, *, run_id: str) -> RunLeaseRecord | None:
        connection = await self._ensure_connection()
        cursor = await connection.execute(
            """
            SELECT run_id, worker_id, lease_expires_at
            FROM run_leases
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        run_id_obj: object = row["run_id"]
        worker_id_obj: object = row["worker_id"]
        expires_obj: object = row["lease_expires_at"]
        if not isinstance(run_id_obj, str):
            raise TypeError(f"Invalid run_id row type: {type(run_id_obj)!r}")
        if not isinstance(worker_id_obj, str):
            raise TypeError(f"Invalid worker_id row type: {type(worker_id_obj)!r}")
        if not isinstance(expires_obj, str):
            raise TypeError(f"Invalid lease_expires_at row type: {type(expires_obj)!r}")
        expires_at = datetime.fromisoformat(expires_obj)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return RunLeaseRecord(
            run_id=run_id_obj,
            worker_id=worker_id_obj,
            lease_expires_at=expires_at,
        )

    async def verify_run_chain(self, run_id: str) -> bool:
        try:
            events = await self.get_events_for_run(run_id)
        except Exception:
            return False

        previous_hash: str | None = None
        for event in events:
            if event.prev_event_hash != previous_hash:
                return False
            expected_hash = compute_event_hash(
                event_id=event.event_id,
                run_id=event.run_id,
                tenant_id=event.tenant_id,
                seq=event.seq,
                event_type=event.event_type,
                prev_event_hash=event.prev_event_hash,
                parent_step_key=event.parent_step_key,
                timestamp=event.timestamp,
                payload=event.payload,
            )
            if expected_hash != event.event_hash:
                return False
            previous_hash = event.event_hash
        return True

    async def close(self) -> None:
        if self._connection is None:
            return
        await self._connection.close()
        self._connection = None

    async def _ensure_connection(self) -> aiosqlite.Connection:
        if self._connection is not None:
            return self._connection

        async with self._connection_lock:
            if self._connection is None:
                self._database_path.parent.mkdir(parents=True, exist_ok=True)
                connection = await aiosqlite.connect(self._database_path)
                connection.row_factory = aiosqlite.Row
                try:
                    await self._initialize_connection(connection)
                except Exception:
                    await connection.close()
                    raise
                self._connection = connection
        if self._connection is None:
            raise RuntimeError("Failed to initialize SQLite connection.")
        return self._connection

    async def _initialize_connection(self, connection: aiosqlite.Connection) -> None:
        for attempt in range(self._max_retry_attempts):
            try:
                await connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms};")
                await connection.execute("PRAGMA journal_mode = WAL;")
                await connection.execute("PRAGMA synchronous = NORMAL;")
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kernel_events (
                        run_id TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        event_id TEXT NOT NULL UNIQUE,
                        tenant_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        prev_event_hash TEXT,
                        event_hash TEXT NOT NULL,
                        parent_step_key TEXT,
                        timestamp TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        tool_name TEXT,
                        tool_outcome TEXT,
                        tool_request_id TEXT,
                        tool_semantic_key TEXT,
                        tool_amount_usd REAL,
                        PRIMARY KEY (run_id, seq)
                    )
                    """
                )
                cursor = await connection.execute("PRAGMA table_info(kernel_events);")
                columns = await cursor.fetchall()
                await cursor.close()
                column_names = [column["name"] for column in columns]
                if "parent_step_key" not in column_names:
                    try:
                        await connection.execute(
                            "ALTER TABLE kernel_events ADD COLUMN parent_step_key TEXT"
                        )
                    except aiosqlite.OperationalError as exc:
                        if "duplicate column name: parent_step_key" not in str(
                            exc
                        ).lower():
                            raise
                for column_name, column_type in (
                    ("tool_name", "TEXT"),
                    ("tool_outcome", "TEXT"),
                    ("tool_request_id", "TEXT"),
                    ("tool_semantic_key", "TEXT"),
                    ("tool_amount_usd", "REAL"),
                ):
                    if column_name in column_names:
                        continue
                    try:
                        await connection.execute(
                            f"ALTER TABLE kernel_events ADD COLUMN {column_name} {column_type}"
                        )
                    except aiosqlite.OperationalError as exc:
                        duplicate_message = f"duplicate column name: {column_name}"
                        if duplicate_message not in str(exc).lower():
                            raise
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kernel_events_run_seq
                    ON kernel_events (run_id, seq)
                    """
                )
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kernel_events_run_type_seq
                    ON kernel_events (run_id, event_type, seq DESC)
                    """
                )
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kernel_events_tenant_tool_time
                    ON kernel_events (tenant_id, tool_name, timestamp DESC)
                    """
                )
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kernel_events_run_tool_seq
                    ON kernel_events (run_id, tool_name, seq DESC)
                    """
                )
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kernel_events_tool_semantic
                    ON kernel_events (tenant_id, tool_name, tool_semantic_key, seq DESC)
                    """
                )
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS run_leases (
                        run_id TEXT PRIMARY KEY,
                        worker_id TEXT NOT NULL,
                        lease_expires_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_run_leases_expiry
                    ON run_leases (lease_expires_at)
                    """
                )
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS run_state_snapshots (
                        run_id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        last_event_seq INTEGER NOT NULL,
                        last_event_type TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        blocked_on TEXT,
                        failure_reason TEXT,
                        last_step_key TEXT,
                        drift_count INTEGER NOT NULL,
                        last_stage TEXT,
                        last_tool TEXT,
                        model_cost_total REAL NOT NULL,
                        open_pause_count INTEGER NOT NULL,
                        explain_status TEXT NOT NULL,
                        explain_failure_reason TEXT,
                        explain_failure_step TEXT
                    )
                    """
                )
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_run_state_snapshots_tenant_updated
                    ON run_state_snapshots (tenant_id, updated_at DESC)
                    """
                )
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_run_state_snapshots_tenant_status_updated
                    ON run_state_snapshots (tenant_id, status, updated_at DESC)
                    """
                )
                await connection.commit()
                return
            except aiosqlite.OperationalError as exc:
                await _rollback_quietly(connection)
                if _is_locked_error(exc) and attempt < self._max_retry_attempts - 1:
                    await asyncio.sleep(self._retry_backoff(attempt))
                    continue
                raise RuntimeError(
                    "Failed to initialize SQLiteStore due to database lock contention. "
                    "For multi-worker deployments, prefer a Postgres-backed store."
                ) from exc
            except Exception:
                await _rollback_quietly(connection)
                raise

    async def _next_sequence_and_prev_hash(
        self, connection: aiosqlite.Connection, *, run_id: str
    ) -> tuple[int, str | None]:
        cursor = await connection.execute(
            """
            SELECT seq, event_hash
            FROM kernel_events
            WHERE run_id = ?
            ORDER BY seq DESC
            LIMIT 1
            """,
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return 1, None
        seq_raw = row["seq"]
        event_hash_raw = row["event_hash"]
        if not isinstance(seq_raw, int):
            raise TypeError(
                f"Expected integer seq from database, got {type(seq_raw)!r}"
            )
        if not isinstance(event_hash_raw, str):
            raise TypeError(
                f"Expected string event_hash from database, got {type(event_hash_raw)!r}"
            )
        return seq_raw + 1, event_hash_raw

    async def _append_event_once(
        self,
        *,
        connection: aiosqlite.Connection,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
        parent_step_key: str | None = None,
    ) -> KernelEvent:
        await connection.execute("BEGIN IMMEDIATE")
        try:
            next_seq, prev_event_hash = await self._next_sequence_and_prev_hash(
                connection, run_id=run_id
            )
            timestamp = datetime.now(timezone.utc)
            event_id = uuid4().hex
            event = KernelEvent(
                event_id=event_id,
                run_id=run_id,
                tenant_id=tenant_id,
                seq=next_seq,
                event_type=event_type,
                prev_event_hash=prev_event_hash,
                event_hash=compute_event_hash(
                    event_id=event_id,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    seq=next_seq,
                    event_type=event_type,
                    prev_event_hash=prev_event_hash,
                    parent_step_key=parent_step_key,
                    timestamp=timestamp,
                    payload=payload,
                ),
                parent_step_key=parent_step_key,
                timestamp=timestamp,
                payload=payload,
            )
            tool_name, tool_outcome, tool_request_id, tool_semantic_key, tool_amount_usd = (
                _tool_columns_from_payload(payload)
            )
            await connection.execute(
                """
                INSERT INTO kernel_events (
                    run_id, seq, event_id, tenant_id, event_type, prev_event_hash,
                    event_hash, parent_step_key, timestamp, payload_json, tool_name,
                    tool_outcome, tool_request_id, tool_semantic_key, tool_amount_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    event.seq,
                    event.event_id,
                    event.tenant_id,
                    event.event_type.value,
                    event.prev_event_hash,
                    event.event_hash,
                    event.parent_step_key,
                    event.timestamp.isoformat(),
                    json.dumps(event.payload.model_dump(mode="json")),
                    tool_name,
                    tool_outcome,
                    tool_request_id,
                    tool_semantic_key,
                    tool_amount_usd,
                ),
            )
            await self._upsert_run_state_snapshot(connection=connection, event=event)
            await connection.commit()
            return event
        except Exception:
            await _rollback_quietly(connection)
            raise

    async def _upsert_run_state_snapshot(
        self,
        *,
        connection: aiosqlite.Connection,
        event: KernelEvent,
    ) -> None:
        existing_cursor = await connection.execute(
            """
            SELECT
                run_id,
                tenant_id,
                last_event_seq,
                last_event_type,
                updated_at,
                status,
                blocked_on,
                failure_reason,
                last_step_key,
                drift_count,
                last_stage,
                last_tool,
                model_cost_total,
                open_pause_count,
                explain_status,
                explain_failure_reason,
                explain_failure_step
            FROM run_state_snapshots
            WHERE run_id = ?
            LIMIT 1
            """,
            (event.run_id,),
        )
        row = await existing_cursor.fetchone()
        await existing_cursor.close()

        next_snapshot = (
            apply_event_to_run_state_snapshot(snapshot=_snapshot_from_row(row), event=event)
            if row is not None
            else initialize_run_state_snapshot(event=event)
        )
        await connection.execute(
            """
            INSERT INTO run_state_snapshots (
                run_id,
                tenant_id,
                last_event_seq,
                last_event_type,
                updated_at,
                status,
                blocked_on,
                failure_reason,
                last_step_key,
                drift_count,
                last_stage,
                last_tool,
                model_cost_total,
                open_pause_count,
                explain_status,
                explain_failure_reason,
                explain_failure_step
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                tenant_id = excluded.tenant_id,
                last_event_seq = excluded.last_event_seq,
                last_event_type = excluded.last_event_type,
                updated_at = excluded.updated_at,
                status = excluded.status,
                blocked_on = excluded.blocked_on,
                failure_reason = excluded.failure_reason,
                last_step_key = excluded.last_step_key,
                drift_count = excluded.drift_count,
                last_stage = excluded.last_stage,
                last_tool = excluded.last_tool,
                model_cost_total = excluded.model_cost_total,
                open_pause_count = excluded.open_pause_count,
                explain_status = excluded.explain_status,
                explain_failure_reason = excluded.explain_failure_reason,
                explain_failure_step = excluded.explain_failure_step
            """,
            (
                next_snapshot.run_id,
                next_snapshot.tenant_id,
                next_snapshot.last_event_seq,
                next_snapshot.last_event_type,
                next_snapshot.updated_at.isoformat(),
                next_snapshot.status,
                next_snapshot.blocked_on,
                next_snapshot.failure_reason,
                next_snapshot.last_step_key,
                next_snapshot.drift_count,
                next_snapshot.last_stage,
                next_snapshot.last_tool,
                next_snapshot.model_cost_total,
                next_snapshot.open_pause_count,
                next_snapshot.explain_status,
                next_snapshot.explain_failure_reason,
                next_snapshot.explain_failure_step,
            ),
        )

    async def _sum_model_cost_via_events(self, *, run_id: str) -> float:
        events = await self.get_events_for_run(run_id)
        spent = 0.0
        for event in events:
            if event.event_type != EventType.MODEL_COMPLETED:
                continue
            payload = event.payload
            if payload.kind != "model_completed":
                raise RuntimeError(
                    f"Invalid event payload kind {payload.kind!r} for model_completed event."
                )
            spent += payload.cost_usd
        return spent

    async def _latest_run_summary_via_events(
        self,
        *,
        run_id: str,
        summary_type: str,
    ) -> RunSummaryPayload | None:
        events = await self.get_events_for_run(run_id)
        for event in reversed(events):
            if event.event_type != EventType.RUN_SUMMARY:
                continue
            payload = event.payload
            if not isinstance(payload, RunSummaryPayload):
                continue
            if payload.summary_type == summary_type:
                return payload
        return None

    def _retry_backoff(self, attempt: int) -> float:
        multiplier = float(2**attempt)
        delay = float(self._retry_backoff_seconds * multiplier)
        if delay > 1.0:
            return 1.0
        return float(delay)


def _snapshot_from_row(row: aiosqlite.Row) -> RunStateSnapshotRecord:
    run_id_obj: object = row["run_id"]
    tenant_id_obj: object = row["tenant_id"]
    seq_obj: object = row["last_event_seq"]
    event_type_obj: object = row["last_event_type"]
    updated_at_obj: object = row["updated_at"]
    status_obj: object = row["status"]
    blocked_on_obj: object = row["blocked_on"]
    failure_reason_obj: object = row["failure_reason"]
    last_step_key_obj: object = row["last_step_key"]
    drift_count_obj: object = row["drift_count"]
    last_stage_obj: object = row["last_stage"]
    last_tool_obj: object = row["last_tool"]
    model_cost_total_obj: object = row["model_cost_total"]
    open_pause_count_obj: object = row["open_pause_count"]
    explain_status_obj: object = row["explain_status"]
    explain_failure_reason_obj: object = row["explain_failure_reason"]
    explain_failure_step_obj: object = row["explain_failure_step"]

    if not isinstance(run_id_obj, str):
        raise TypeError(f"Invalid run_id row type: {type(run_id_obj)!r}")
    if not isinstance(tenant_id_obj, str):
        raise TypeError(f"Invalid tenant_id row type: {type(tenant_id_obj)!r}")
    if not isinstance(seq_obj, int):
        raise TypeError(f"Invalid last_event_seq row type: {type(seq_obj)!r}")
    if not isinstance(event_type_obj, str):
        raise TypeError(f"Invalid last_event_type row type: {type(event_type_obj)!r}")
    if not isinstance(updated_at_obj, str):
        raise TypeError(f"Invalid updated_at row type: {type(updated_at_obj)!r}")
    if not isinstance(status_obj, str):
        raise TypeError(f"Invalid status row type: {type(status_obj)!r}")
    if blocked_on_obj is not None and not isinstance(blocked_on_obj, str):
        raise TypeError(f"Invalid blocked_on row type: {type(blocked_on_obj)!r}")
    if failure_reason_obj is not None and not isinstance(failure_reason_obj, str):
        raise TypeError(f"Invalid failure_reason row type: {type(failure_reason_obj)!r}")
    if last_step_key_obj is not None and not isinstance(last_step_key_obj, str):
        raise TypeError(f"Invalid last_step_key row type: {type(last_step_key_obj)!r}")
    if not isinstance(drift_count_obj, int):
        raise TypeError(f"Invalid drift_count row type: {type(drift_count_obj)!r}")
    if last_stage_obj is not None and not isinstance(last_stage_obj, str):
        raise TypeError(f"Invalid last_stage row type: {type(last_stage_obj)!r}")
    if last_tool_obj is not None and not isinstance(last_tool_obj, str):
        raise TypeError(f"Invalid last_tool row type: {type(last_tool_obj)!r}")
    if not isinstance(model_cost_total_obj, (int, float)):
        raise TypeError(f"Invalid model_cost_total row type: {type(model_cost_total_obj)!r}")
    if not isinstance(open_pause_count_obj, int):
        raise TypeError(f"Invalid open_pause_count row type: {type(open_pause_count_obj)!r}")
    if not isinstance(explain_status_obj, str):
        raise TypeError(f"Invalid explain_status row type: {type(explain_status_obj)!r}")
    if explain_failure_reason_obj is not None and not isinstance(explain_failure_reason_obj, str):
        raise TypeError(
            f"Invalid explain_failure_reason row type: {type(explain_failure_reason_obj)!r}"
        )
    if explain_failure_step_obj is not None and not isinstance(explain_failure_step_obj, str):
        raise TypeError(
            f"Invalid explain_failure_step row type: {type(explain_failure_step_obj)!r}"
        )

    updated_at = datetime.fromisoformat(updated_at_obj)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    status = _coerce_snapshot_status(status_obj)
    explain_status = _coerce_explain_status(explain_status_obj)
    return RunStateSnapshotRecord(
        run_id=run_id_obj,
        tenant_id=tenant_id_obj,
        last_event_seq=seq_obj,
        last_event_type=event_type_obj,
        updated_at=updated_at,
        status=status,
        blocked_on=blocked_on_obj,
        failure_reason=failure_reason_obj,
        last_step_key=last_step_key_obj,
        drift_count=drift_count_obj,
        last_stage=last_stage_obj,
        last_tool=last_tool_obj,
        model_cost_total=float(model_cost_total_obj),
        open_pause_count=open_pause_count_obj,
        explain_status=explain_status,
        explain_failure_reason=explain_failure_reason_obj,
        explain_failure_step=explain_failure_step_obj,
    )


def _coerce_snapshot_status(status: str) -> RunStateLifecycleStatus:
    if status == "active":
        return "active"
    if status == "paused":
        return "paused"
    if status == "failed":
        return "failed"
    if status == "completed":
        return "completed"
    raise TypeError(f"Invalid snapshot status value: {status!r}")


def _coerce_explain_status(status: str) -> Literal["completed", "failed"]:
    if status == "completed":
        return "completed"
    if status == "failed":
        return "failed"
    raise TypeError(f"Invalid snapshot explain_status value: {status!r}")


def _tool_columns_from_payload(
    payload: EventPayload,
) -> tuple[str | None, str | None, str | None, str | None, float | None]:
    if isinstance(payload, ToolRequestedPayload):
        return (
            payload.tool_name,
            None,
            None,
            payload.semantic_idempotency_key,
            payload.amount_usd,
        )
    if isinstance(payload, ToolCompletedPayload):
        return (
            payload.tool_name,
            payload.outcome,
            payload.request_id,
            None,
            None,
        )
    return None, None, None, None, None


def _coerce_count(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"Invalid count row type: {type(value)!r}")


def _load_request_payload(raw_json: str) -> ToolRequestedPayload:
    payload_dict_raw = json.loads(raw_json)
    if not isinstance(payload_dict_raw, dict):
        raise TypeError("Stored request payload_json did not decode to an object.")
    payload = _PAYLOAD_ADAPTER.validate_python(payload_dict_raw)
    if not isinstance(payload, ToolRequestedPayload):
        raise TypeError(f"Expected ToolRequestedPayload, got {type(payload)!r}.")
    return payload


def _is_locked_error(exc: aiosqlite.OperationalError) -> bool:
    message = str(exc).lower()
    return "database is locked" in message or "database schema is locked" in message


def _is_run_seq_conflict_error(exc: aiosqlite.IntegrityError) -> bool:
    message = str(exc).lower()
    return "unique constraint failed: kernel_events.run_id, kernel_events.seq" in message


def _is_missing_json_extract_error(exc: aiosqlite.OperationalError) -> bool:
    return "no such function: json_extract" in str(exc).lower()


async def _rollback_quietly(connection: aiosqlite.Connection) -> None:
    try:
        await connection.rollback()
    except aiosqlite.OperationalError:
        return
