from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import aiosqlite
from pydantic import TypeAdapter

from artana.events import (
    EventPayload,
    EventType,
    KernelEvent,
    RunSummaryPayload,
    compute_event_hash,
)
from artana.store.base import EventStore

_PAYLOAD_ADAPTER: TypeAdapter[EventPayload] = TypeAdapter(EventPayload)


class SQLiteStore(EventStore):
    def __init__(
        self,
        database_path: str,
        *,
        busy_timeout_ms: int = 5000,
        max_retry_attempts: int = 5,
        retry_backoff_seconds: float = 0.02,
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

    async def append_event(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
    ) -> KernelEvent:
        connection = await self._ensure_connection()

        async with self._append_lock:
            for attempt in range(self._max_retry_attempts):
                try:
                    return await self._append_event_once(
                        connection=connection,
                        run_id=run_id,
                        tenant_id=tenant_id,
                        event_type=event_type,
                        payload=payload,
                    )
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
                   event_hash, timestamp, payload_json
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
                        timestamp TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        PRIMARY KEY (run_id, seq)
                    )
                    """
                )
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
                    timestamp=timestamp,
                    payload=payload,
                ),
                timestamp=timestamp,
                payload=payload,
            )
            await connection.execute(
                """
                INSERT INTO kernel_events (
                    run_id, seq, event_id, tenant_id, event_type, prev_event_hash,
                    event_hash, timestamp, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    event.seq,
                    event.event_id,
                    event.tenant_id,
                    event.event_type.value,
                    event.prev_event_hash,
                    event.event_hash,
                    event.timestamp.isoformat(),
                    json.dumps(event.payload.model_dump(mode="json")),
                ),
            )
            await connection.commit()
            return event
        except Exception:
            await _rollback_quietly(connection)
            raise

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
