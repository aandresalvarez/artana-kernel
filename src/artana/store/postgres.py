from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import asyncpg  # type: ignore[import-untyped]
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
from artana.store.base import EventStore, RunLeaseRecord, ToolSemanticOutcomeRecord

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

    async def get_tool_request_count_for_run(self, *, run_id: str, tool_name: str) -> int:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT COUNT(*) AS total_count
                FROM kernel_events
                WHERE run_id = $1
                  AND event_type = $2
                  AND tool_name = $3
                """,
                run_id,
                EventType.TOOL_REQUESTED.value,
                tool_name,
            )
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
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT COUNT(*) AS total_count
                FROM kernel_events
                WHERE tenant_id = $1
                  AND event_type = $2
                  AND tool_name = $3
                  AND timestamp >= $4
                """,
                tenant_id,
                EventType.TOOL_REQUESTED.value,
                tool_name,
                since,
            )
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
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT
                    c.run_id AS run_id,
                    c.tool_request_id AS request_id,
                    c.tool_outcome AS outcome,
                    r.payload_json AS request_payload_json
                FROM kernel_events c
                JOIN kernel_events r
                    ON c.tool_request_id = r.event_id
                WHERE c.event_type = $1
                  AND r.event_type = $2
                  AND r.tenant_id = $3
                  AND r.tool_name = $4
                  AND r.tool_semantic_key = $5
                ORDER BY c.timestamp DESC, c.seq DESC
                LIMIT 1
                """,
                EventType.TOOL_COMPLETED.value,
                EventType.TOOL_REQUESTED.value,
                tenant_id,
                tool_name,
                semantic_idempotency_key,
            )
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
        pool = await self._ensure_pool()
        where_clauses: list[str] = []
        params: list[object] = []
        if tenant_id is not None:
            params.append(tenant_id)
            where_clauses.append(f"tenant_id = ${len(params)}")
        if since is not None:
            params.append(since)
            where_clauses.append(f"timestamp >= ${len(params)}")
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        async with pool.acquire() as connection:
            rows = await connection.fetch(
                f"""
                SELECT run_id, MAX(timestamp) AS latest_timestamp
                FROM kernel_events
                {where_sql}
                GROUP BY run_id
                ORDER BY latest_timestamp DESC
                """,
                *params,
            )
        run_ids: list[str] = []
        for row in rows:
            run_id_obj: object = row["run_id"]
            if not isinstance(run_id_obj, str):
                raise TypeError(f"Invalid run_id row type: {type(run_id_obj)!r}")
            run_ids.append(run_id_obj)
        return run_ids

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
                if datetime.now(timezone.utc) - idle_started_at >= timedelta(
                    seconds=idle_timeout_seconds
                ):
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
        pool = await self._ensure_pool()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT worker_id, lease_expires_at
                    FROM run_leases
                    WHERE run_id = $1
                    FOR UPDATE
                    """,
                    run_id,
                )
                if row is None:
                    await connection.execute(
                        """
                        INSERT INTO run_leases (run_id, worker_id, lease_expires_at, updated_at)
                        VALUES ($1, $2, $3, $4)
                        """,
                        run_id,
                        worker_id,
                        expires_at,
                        now,
                    )
                    return True
                current_worker_obj: object = row["worker_id"]
                current_expiry_obj: object = row["lease_expires_at"]
                if not isinstance(current_worker_obj, str):
                    raise TypeError(
                        f"Invalid worker_id row type: {type(current_worker_obj)!r}"
                    )
                if not isinstance(current_expiry_obj, datetime):
                    raise TypeError(
                        f"Invalid lease_expires_at row type: {type(current_expiry_obj)!r}"
                    )
                current_expiry = current_expiry_obj
                if current_expiry.tzinfo is None:
                    current_expiry = current_expiry.replace(tzinfo=timezone.utc)
                if current_worker_obj != worker_id and current_expiry > now:
                    return False
                await connection.execute(
                    """
                    UPDATE run_leases
                    SET worker_id = $2, lease_expires_at = $3, updated_at = $4
                    WHERE run_id = $1
                    """,
                    run_id,
                    worker_id,
                    expires_at,
                    now,
                )
                return True

    async def renew_run_lease(
        self,
        *,
        run_id: str,
        worker_id: str,
        ttl_seconds: int,
    ) -> bool:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0.")
        pool = await self._ensure_pool()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)
        async with pool.acquire() as connection:
            status_raw: object = await connection.execute(
                """
                UPDATE run_leases
                SET lease_expires_at = $3, updated_at = $4
                WHERE run_id = $1
                  AND worker_id = $2
                  AND lease_expires_at > $4
                """,
                run_id,
                worker_id,
                expires_at,
                now,
            )
        if not isinstance(status_raw, str):
            raise TypeError(f"Invalid lease renewal status type: {type(status_raw)!r}")
        status = status_raw
        return status.endswith("1")

    async def release_run_lease(
        self,
        *,
        run_id: str,
        worker_id: str,
    ) -> bool:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            status_raw: object = await connection.execute(
                """
                DELETE FROM run_leases
                WHERE run_id = $1
                  AND worker_id = $2
                """,
                run_id,
                worker_id,
            )
        if not isinstance(status_raw, str):
            raise TypeError(f"Invalid lease release status type: {type(status_raw)!r}")
        status = status_raw
        return status.endswith("1")

    async def get_run_lease(self, *, run_id: str) -> RunLeaseRecord | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT run_id, worker_id, lease_expires_at
                FROM run_leases
                WHERE run_id = $1
                LIMIT 1
                """,
                run_id,
            )
        if row is None:
            return None
        run_id_obj: object = row["run_id"]
        worker_id_obj: object = row["worker_id"]
        expiry_obj: object = row["lease_expires_at"]
        if not isinstance(run_id_obj, str):
            raise TypeError(f"Invalid run_id row type: {type(run_id_obj)!r}")
        if not isinstance(worker_id_obj, str):
            raise TypeError(f"Invalid worker_id row type: {type(worker_id_obj)!r}")
        if not isinstance(expiry_obj, datetime):
            raise TypeError(f"Invalid lease_expires_at row type: {type(expiry_obj)!r}")
        expiry = expiry_obj
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return RunLeaseRecord(
            run_id=run_id_obj,
            worker_id=worker_id_obj,
            lease_expires_at=expiry,
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
                    parent_step_key TEXT,
                    timestamp TIMESTAMPTZ NOT NULL,
                    payload_json TEXT NOT NULL,
                    tool_name TEXT,
                    tool_outcome TEXT,
                    tool_request_id TEXT,
                    tool_semantic_key TEXT,
                    tool_amount_usd DOUBLE PRECISION,
                    PRIMARY KEY (run_id, seq)
                )
                """
            )
            await connection.execute(
                "ALTER TABLE kernel_events ADD COLUMN IF NOT EXISTS parent_step_key TEXT"
            )
            await connection.execute(
                "ALTER TABLE kernel_events ADD COLUMN IF NOT EXISTS tool_name TEXT"
            )
            await connection.execute(
                "ALTER TABLE kernel_events ADD COLUMN IF NOT EXISTS tool_outcome TEXT"
            )
            await connection.execute(
                "ALTER TABLE kernel_events ADD COLUMN IF NOT EXISTS tool_request_id TEXT"
            )
            await connection.execute(
                "ALTER TABLE kernel_events ADD COLUMN IF NOT EXISTS tool_semantic_key TEXT"
            )
            await connection.execute(
                "ALTER TABLE kernel_events ADD COLUMN IF NOT EXISTS "
                "tool_amount_usd DOUBLE PRECISION"
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
                    lease_expires_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            await connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_run_leases_expiry
                ON run_leases (lease_expires_at)
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
                (
                    tool_name,
                    tool_outcome,
                    tool_request_id,
                    tool_semantic_key,
                    tool_amount_usd,
                ) = _tool_columns_from_payload(payload)
                await connection.execute(
                    """
                    INSERT INTO kernel_events (
                        run_id, seq, event_id, tenant_id, event_type, prev_event_hash,
                        event_hash, parent_step_key, timestamp, payload_json, tool_name,
                        tool_outcome, tool_request_id, tool_semantic_key, tool_amount_usd
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
                    )
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
                    tool_name,
                    tool_outcome,
                    tool_request_id,
                    tool_semantic_key,
                    tool_amount_usd,
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
