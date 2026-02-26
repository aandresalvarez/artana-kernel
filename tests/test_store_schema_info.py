from __future__ import annotations

from pathlib import Path

import pytest

from artana.store import PostgresStore, SQLiteStore


@pytest.mark.asyncio
async def test_sqlite_store_schema_info(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path / "state.db"))
    try:
        info = await store.get_schema_info()
    finally:
        await store.close()

    assert info.backend == "sqlite"
    assert info.schema_version == "1"


@pytest.mark.asyncio
async def test_postgres_store_schema_info() -> None:
    store = PostgresStore("postgresql://user:pass@localhost:5432/db")
    try:
        info = await store.get_schema_info()
    finally:
        await store.close()

    assert info.backend == "postgres"
    assert info.schema_version == "1"
