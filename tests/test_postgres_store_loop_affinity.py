from __future__ import annotations

import asyncio

import pytest

from artana.store import LoopAffinityError, PostgresStore


class _FakeConnection:
    async def fetch(self, query: str, *args: object) -> list[object]:
        return []


class _FakeAcquire:
    async def __aenter__(self) -> _FakeConnection:
        return _FakeConnection()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> bool:
        return False


class _FakePool:
    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire()

    async def close(self) -> None:
        return None


def test_postgres_store_rejects_cross_loop_close() -> None:
    store = PostgresStore("postgresql://example.invalid:5432/artana")

    asyncio.run(store.close())

    with pytest.raises(LoopAffinityError, match="PostgresStore is bound to event loop"):
        asyncio.run(store.close())


def test_postgres_store_rejects_cross_loop_reads() -> None:
    store = PostgresStore("postgresql://example.invalid:5432/artana")
    store._pool = _FakePool()  # type: ignore[assignment]

    assert asyncio.run(store.get_events_for_run("run_loop_affinity")) == []

    with pytest.raises(LoopAffinityError, match="owning loop"):
        asyncio.run(store.get_events_for_run("run_loop_affinity"))
