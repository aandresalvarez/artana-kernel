from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import aiosqlite


class MemoryStore(ABC):
    @abstractmethod
    async def load(self, run_id: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def replace(self, run_id: str, content: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def append(self, run_id: str, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(self, run_id: str, query: str, *, max_results: int = 5) -> str:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    async def load(self, run_id: str) -> str:
        return self._values.get(run_id, "")

    async def replace(self, run_id: str, content: str) -> None:
        self._values[run_id] = content

    async def append(self, run_id: str, text: str) -> None:
        current = self._values.get(run_id, "")
        if current:
            self._values[run_id] = f"{current}\n{text}"
        else:
            self._values[run_id] = text

    async def search(self, run_id: str, query: str, *, max_results: int = 5) -> str:
        current = self._values.get(run_id, "")
        if not query:
            return current
        needle = query.lower()
        lines = [line for line in current.splitlines() if needle in line.lower()]
        return json.dumps(lines[:max(1, max_results)])

    async def close(self) -> None:
        return None


class SQLiteMemoryStore(MemoryStore):
    def __init__(
        self,
        db_path: str | Path,
        *,
        table_name: str = "agent_memory",
    ) -> None:
        self._path = Path(db_path)
        self._table_name = table_name
        self._connection: aiosqlite.Connection | None = None

    async def load(self, run_id: str) -> str:
        connection = await self._ensure_connection()
        cursor = await connection.execute(
            f"""
            SELECT memory
            FROM {self._table_name}
            WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return ""
        value = row[0]
        if not isinstance(value, str):
            return ""
        return value

    async def replace(self, run_id: str, content: str) -> None:
        connection = await self._ensure_connection()
        await connection.execute(
            f"""
            INSERT INTO {self._table_name} (run_id, memory)
            VALUES (?, ?)
            ON CONFLICT(run_id) DO UPDATE SET memory = excluded.memory
            """,
            (run_id, content),
        )
        await connection.commit()

    async def append(self, run_id: str, text: str) -> None:
        current = await self.load(run_id)
        if current:
            await self.replace(run_id=run_id, content=f"{current}\n{text}")
        else:
            await self.replace(run_id=run_id, content=text)

    async def search(self, run_id: str, query: str, *, max_results: int = 5) -> str:
        memory_text = await self.load(run_id)
        if not query:
            return self._serialize_results([memory_text]) if memory_text else "[]"

        needle = query.lower()
        matches = [
            line
            for line in memory_text.splitlines()
            if needle in line.lower()
        ]
        return self._serialize_results(matches[: max(1, max_results)])

    async def close(self) -> None:
        if self._connection is None:
            return
        await self._connection.close()
        self._connection = None

    async def _ensure_connection(self) -> aiosqlite.Connection:
        if self._connection is not None:
            return self._connection

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self._path)
        await self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                run_id TEXT PRIMARY KEY,
                memory TEXT NOT NULL DEFAULT ''
            )
            """
        )
        await self._connection.commit()
        return self._connection

    def _serialize_results(self, results: list[str]) -> str:
        return json.dumps(results, ensure_ascii=False)


# Backwards-compatible alias used by examples and tests.
MemoryStoreLike = MemoryStore
