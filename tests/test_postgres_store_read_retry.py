from __future__ import annotations

import json
from datetime import datetime, timezone

import asyncpg  # type: ignore[import-untyped]
import pytest

from artana.events import EventType, RunStartedPayload, RunSummaryPayload, compute_event_hash
from artana.store import PostgresStore


class _AcquireContext:
    def __init__(self, connection: object) -> None:
        self._connection = connection

    async def __aenter__(self) -> object:
        return self._connection

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _FakePool:
    def __init__(self, connection: object) -> None:
        self._connection = connection
        self.close_calls = 0
        self.terminate_calls = 0

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self._connection)

    async def close(self) -> None:
        self.close_calls += 1

    def terminate(self) -> None:
        self.terminate_calls += 1


class _AlwaysClosedFetchConnection:
    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        _ = query, args
        raise asyncpg.ConnectionDoesNotExistError(
            "connection was closed in the middle of operation"
        )


class _StaticFetchConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        _ = query, args
        return self._rows


class _AlwaysClosedFetchRowConnection:
    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        _ = query, args
        raise asyncpg.ConnectionDoesNotExistError(
            "connection was closed in the middle of operation"
        )


class _StaticFetchRowConnection:
    def __init__(self, row: dict[str, object] | None) -> None:
        self._row = row

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        _ = query, args
        return self._row


def _script_ensure_pool(
    store: PostgresStore,
    pools: list[_FakePool],
) -> tuple[dict[str, int], object]:
    call_state = {"count": 0}

    async def _ensure_pool() -> object:
        index = min(call_state["count"], len(pools) - 1)
        pool = pools[index]
        call_state["count"] += 1
        store._pool = pool
        return pool

    return call_state, _ensure_pool


@pytest.mark.asyncio
async def test_get_events_for_run_retries_after_connection_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PostgresStore(
        "postgresql://user:pass@localhost:5432/db",
        max_retry_attempts=2,
        retry_backoff_seconds=0.0001,
    )
    payload = RunStartedPayload()
    timestamp = datetime.now(timezone.utc)
    event_id = "evt_1"
    event_row = {
        "run_id": "run_retry",
        "seq": 1,
        "event_id": event_id,
        "tenant_id": "tenant_retry",
        "event_type": EventType.RUN_STARTED.value,
        "prev_event_hash": None,
        "event_hash": compute_event_hash(
            event_id=event_id,
            run_id="run_retry",
            tenant_id="tenant_retry",
            seq=1,
            event_type=EventType.RUN_STARTED,
            prev_event_hash=None,
            parent_step_key=None,
            timestamp=timestamp,
            payload=payload,
        ),
        "parent_step_key": None,
        "timestamp": timestamp,
        "payload_json": json.dumps(payload.model_dump(mode="json")),
    }
    first_pool = _FakePool(_AlwaysClosedFetchConnection())
    second_pool = _FakePool(_StaticFetchConnection([event_row]))
    call_state, scripted_ensure_pool = _script_ensure_pool(store, [first_pool, second_pool])
    monkeypatch.setattr(store, "_ensure_pool", scripted_ensure_pool)

    events = await store.get_events_for_run("run_retry")

    assert len(events) == 1
    assert events[0].event_id == "evt_1"
    assert first_pool.close_calls == 1
    assert second_pool.close_calls == 0
    assert call_state["count"] == 2
    await store.close()


@pytest.mark.asyncio
async def test_get_latest_run_summary_retries_after_connection_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PostgresStore(
        "postgresql://user:pass@localhost:5432/db",
        max_retry_attempts=2,
        retry_backoff_seconds=0.0001,
    )
    summary_payload = RunSummaryPayload(
        summary_type="task_progress",
        summary_json='{"units":[{"id":"draft","state":"in_progress"}]}',
        step_key="task_progress_1",
    )
    summary_row: dict[str, object] = {
        "payload_json": json.dumps(summary_payload.model_dump(mode="json"))
    }
    first_pool = _FakePool(_AlwaysClosedFetchRowConnection())
    second_pool = _FakePool(_StaticFetchRowConnection(summary_row))
    call_state, scripted_ensure_pool = _script_ensure_pool(store, [first_pool, second_pool])
    monkeypatch.setattr(store, "_ensure_pool", scripted_ensure_pool)

    summary = await store.get_latest_run_summary("run_retry", "task_progress")

    assert summary is not None
    assert summary.step_key == "task_progress_1"
    assert first_pool.close_calls == 1
    assert second_pool.close_calls == 0
    assert call_state["count"] == 2
    await store.close()
