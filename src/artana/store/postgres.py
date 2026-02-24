from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg  # type: ignore[import-untyped]
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


class PostgresStore(EventStore):
    def __init__(
        self,
        dsn: str,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        command_timeout_seconds: float = 30.0,
        max_retry_attempts: int = 5,
        retry_backoff_seconds: float = 0.02,
        on_event: Callable[[KernelEvent], Awaitable[None]] | None = None,
    ) -> None:
        if min_pool_size <= 0:
            raise ValueError("min_pool_size must be > 0")
        if max_pool_size <= 0:
            raise ValueError("max_pool_size must be > 0")
        if max_pool_size < min_pool_size:
            raise ValueError("max_pool_size must be >= min_pool_size")
        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be > 0")
        if max_retry_attempts <= 0:
            raise ValueError("max_retry_attempts must be > 0")
        if retry_backoff_seconds <= 0:
            raise ValueError("retry_backoff_seconds must be > 0")

        self._dsn = dsn
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._command_timeout_seconds = command_timeout_seconds
        self._max_retry_attempts = max_retry_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._on_event = on_event

        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    async def append_event(
        self,
        *,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
        parent_step_key: str | None = None,
    ) -> KernelEvent:
        pool = await self._ensure_pool()
        for attempt in range(self._max_retry_attempts):
            try:
                event = await self._append_event_once(
                    pool=pool,
                    run_id=run_id,
                    tenant_id=tenant_id,
                    event_type=event_type,
                    payload=payload,
                    parent_step_key=parent_step_key,
                )
                if self._on_event is not None:
                    await self._on_event(event)
                return event
            except (asyncpg.SerializationError, asyncpg.DeadlockDetectedError):
                if attempt >= self._max_retry_attempts - 1:
                    raise
            except asyncpg.UniqueViolationError as exc:
                if not _is_retryable_unique_violation(exc):
                    raise
                if attempt >= self._max_retry_attempts - 1:
                    raise
            await asyncio.sleep(self._retry_backoff(attempt))
        raise RuntimeError("Failed to append event due to repeated Postgres write contention.")

    async def get_events_for_run(self, run_id: str) -> list[KernelEvent]:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT run_id, seq, event_id, tenant_id, event_type, prev_event_hash,
                       event_hash, parent_step_key, timestamp, payload_json
                FROM kernel_events
                WHERE run_id = $1
                ORDER BY seq ASC
                """,
                run_id,
            )

        events: list[KernelEvent] = []
        for row in rows:
            events.append(_event_from_record(row))
        return events

    async def get_latest_run_summary(
        self,
        run_id: str,
        summary_type: str,
    ) -> RunSummaryPayload | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT payload_json
                FROM kernel_events
                WHERE run_id = $1
                  AND event_type = $2
                  AND (payload_json::jsonb ->> 'summary_type') = $3
                ORDER BY seq DESC
                LIMIT 1
                """,
                run_id,
                EventType.RUN_SUMMARY.value,
                summary_type,
            )

        if row is None:
            return None

        payload_json_raw: object = row["payload_json"]
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
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT COALESCE(
                    SUM((payload_json::jsonb ->> 'cost_usd')::double precision),
                    0.0
                ) AS total_cost
                FROM kernel_events
                WHERE run_id = $1
                  AND event_type = $2
                """,
                run_id,
                EventType.MODEL_COMPLETED.value,
            )

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
                parent_step_key=event.parent_step_key,
                timestamp=event.timestamp,
                payload=event.payload,
            )
            if expected_hash != event.event_hash:
                return False
            previous_hash = event.event_hash
        return True

    async def close(self) -> None:
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is not None:
            return self._pool

        async with self._pool_lock:
            if self._pool is None:
                pool = await asyncpg.create_pool(
                    dsn=self._dsn,
                    min_size=self._min_pool_size,
                    max_size=self._max_pool_size,
                    command_timeout=self._command_timeout_seconds,
                )
                try:
                    await self._initialize_pool(pool)
                except Exception:
                    await pool.close()
                    raise
                self._pool = pool

        if self._pool is None:
            raise RuntimeError("Failed to initialize Postgres connection pool.")
        return self._pool

    async def _initialize_pool(self, pool: asyncpg.Pool) -> None:
        async with pool.acquire() as connection:
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
                    timestamp TIMESTAMPTZ NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, seq)
                )
                """
            )
            await connection.execute(
                "ALTER TABLE kernel_events "
                "ADD COLUMN IF NOT EXISTS parent_step_key TEXT"
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

    async def _append_event_once(
        self,
        *,
        pool: asyncpg.Pool,
        run_id: str,
        tenant_id: str,
        event_type: EventType,
        payload: EventPayload,
        parent_step_key: str | None = None,
    ) -> KernelEvent:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock($1)",
                    _advisory_lock_key(run_id),
                )
                next_seq, prev_event_hash = await self._next_sequence_and_prev_hash(
                    connection,
                    run_id=run_id,
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
                await connection.execute(
                    """
                    INSERT INTO kernel_events (
                        run_id, seq, event_id, tenant_id, event_type, prev_event_hash,
                        event_hash, parent_step_key, timestamp, payload_json
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    event.run_id,
                    event.seq,
                    event.event_id,
                    event.tenant_id,
                    event.event_type.value,
                    event.prev_event_hash,
                    event.event_hash,
                    event.parent_step_key,
                    event.timestamp,
                    json.dumps(event.payload.model_dump(mode="json")),
                )
                return event

    async def _next_sequence_and_prev_hash(
        self,
        connection: asyncpg.Connection,
        *,
        run_id: str,
    ) -> tuple[int, str | None]:
        row = await connection.fetchrow(
            """
            SELECT seq, event_hash
            FROM kernel_events
            WHERE run_id = $1
            ORDER BY seq DESC
            LIMIT 1
            """,
            run_id,
        )
        if row is None:
            return 1, None

        seq_raw: object = row["seq"]
        event_hash_raw: object = row["event_hash"]
        if not isinstance(seq_raw, int):
            raise TypeError(f"Expected integer seq from database, got {type(seq_raw)!r}")
        if not isinstance(event_hash_raw, str):
            raise TypeError(
                f"Expected string event_hash from database, got {type(event_hash_raw)!r}"
            )
        return seq_raw + 1, event_hash_raw

    def _retry_backoff(self, attempt: int) -> float:
        multiplier = float(2**attempt)
        delay = float(self._retry_backoff_seconds * multiplier)
        if delay > 1.0:
            return 1.0
        return float(delay)


def _event_from_record(row: asyncpg.Record) -> KernelEvent:
    run_id_raw: object = row["run_id"]
    seq_raw: object = row["seq"]
    event_id_raw: object = row["event_id"]
    tenant_id_raw: object = row["tenant_id"]
    event_type_raw: object = row["event_type"]
    prev_event_hash_raw: object = row["prev_event_hash"]
    event_hash_raw: object = row["event_hash"]
    parent_step_key_raw: object = row["parent_step_key"]
    timestamp_raw: object = row["timestamp"]
    payload_json_raw: object = row["payload_json"]

    if not isinstance(run_id_raw, str):
        raise TypeError(f"Invalid run_id row type: {type(run_id_raw)!r}")
    if not isinstance(seq_raw, int):
        raise TypeError(f"Invalid seq row type: {type(seq_raw)!r}")
    if not isinstance(event_id_raw, str):
        raise TypeError(f"Invalid event_id row type: {type(event_id_raw)!r}")
    if not isinstance(tenant_id_raw, str):
        raise TypeError(f"Invalid tenant_id row type: {type(tenant_id_raw)!r}")
    if not isinstance(event_type_raw, str):
        raise TypeError(f"Invalid event_type row type: {type(event_type_raw)!r}")
    if prev_event_hash_raw is not None and not isinstance(prev_event_hash_raw, str):
        raise TypeError(f"Invalid prev_event_hash row type: {type(prev_event_hash_raw)!r}")
    if not isinstance(event_hash_raw, str):
        raise TypeError(f"Invalid event_hash row type: {type(event_hash_raw)!r}")
    if parent_step_key_raw is not None and not isinstance(parent_step_key_raw, str):
        raise TypeError(f"Invalid parent_step_key row type: {type(parent_step_key_raw)!r}")
    if not isinstance(timestamp_raw, datetime):
        raise TypeError(f"Invalid timestamp row type: {type(timestamp_raw)!r}")
    if not isinstance(payload_json_raw, str):
        raise TypeError(f"Invalid payload_json row type: {type(payload_json_raw)!r}")

    try:
        event_type = EventType(event_type_raw)
    except ValueError as exc:
        raise ValueError(f"Unknown event_type in store: {event_type_raw!r}") from exc

    payload_dict_raw = json.loads(payload_json_raw)
    if not isinstance(payload_dict_raw, dict):
        raise TypeError("Stored payload_json did not decode to an object.")
    payload = _PAYLOAD_ADAPTER.validate_python(payload_dict_raw)

    timestamp = timestamp_raw
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    return KernelEvent(
        event_id=event_id_raw,
        run_id=run_id_raw,
        tenant_id=tenant_id_raw,
        seq=seq_raw,
        event_type=event_type,
        prev_event_hash=prev_event_hash_raw,
        event_hash=event_hash_raw,
        parent_step_key=parent_step_key_raw,
        timestamp=timestamp,
        payload=payload,
    )


def _advisory_lock_key(run_id: str) -> int:
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


def _is_retryable_unique_violation(exc: asyncpg.UniqueViolationError) -> bool:
    constraint_name = getattr(exc, "constraint_name", None)
    if constraint_name is None:
        return True
    return constraint_name in {"kernel_events_pkey", "kernel_events_run_id_seq_key"}
