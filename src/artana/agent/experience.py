from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import aiosqlite
from pydantic import BaseModel, Field


class RuleType(str, Enum):
    WIN_PATTERN = "win_pattern"
    ANTI_PATTERN = "anti_pattern"
    FACT = "fact"


class ExperienceRule(BaseModel):
    rule_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    task_category: str = Field(min_length=1)
    rule_type: RuleType
    content: str = Field(min_length=1)
    success_count: int = Field(default=0, ge=0)
    fail_count: int = Field(default=0, ge=0)


class ReflectionResult(BaseModel):
    extracted_rules: list[ExperienceRule] = Field(default_factory=list)


class ExperienceStore(ABC):
    @abstractmethod
    async def save_rules(self, rules: list[ExperienceRule]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_rules(
        self,
        tenant_id: str,
        task_category: str,
        *,
        limit: int = 10,
    ) -> list[ExperienceRule]:
        raise NotImplementedError

    @abstractmethod
    async def reinforce_rule(self, rule_id: str, *, positive: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


class SQLiteExperienceStore(ExperienceStore):
    def __init__(
        self,
        db_path: str | Path,
        *,
        table_name: str = "experience_rules",
    ) -> None:
        self._path = Path(db_path)
        self._table_name = table_name
        self._connection: aiosqlite.Connection | None = None

    async def save_rules(self, rules: list[ExperienceRule]) -> None:
        if not rules:
            return
        connection = await self._ensure_connection()
        rows: list[tuple[str, str, str, str, str, int, int]] = [
            (
                rule.rule_id,
                rule.tenant_id,
                rule.task_category,
                rule.rule_type.value,
                rule.content,
                rule.success_count,
                rule.fail_count,
            )
            for rule in rules
        ]
        await connection.executemany(
            f"""
            INSERT INTO {self._table_name} (
                rule_id,
                tenant_id,
                task_category,
                rule_type,
                content,
                success_count,
                fail_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rule_id) DO UPDATE SET
                tenant_id = excluded.tenant_id,
                task_category = excluded.task_category,
                rule_type = excluded.rule_type,
                content = excluded.content,
                success_count = excluded.success_count,
                fail_count = excluded.fail_count
            """,
            rows,
        )
        await connection.commit()

    async def get_rules(
        self,
        tenant_id: str,
        task_category: str,
        *,
        limit: int = 10,
    ) -> list[ExperienceRule]:
        if limit <= 0:
            return []
        connection = await self._ensure_connection()
        cursor = await connection.execute(
            f"""
            SELECT
                rule_id,
                tenant_id,
                task_category,
                rule_type,
                content,
                success_count,
                fail_count
            FROM {self._table_name}
            WHERE tenant_id = ? AND task_category = ?
            ORDER BY
                (success_count - fail_count) DESC,
                success_count DESC,
                fail_count ASC,
                rowid DESC
            LIMIT ?
            """,
            (tenant_id, task_category, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            ExperienceRule(
                rule_id=row[0],
                tenant_id=row[1],
                task_category=row[2],
                rule_type=RuleType(row[3]),
                content=row[4],
                success_count=row[5],
                fail_count=row[6],
            )
            for row in rows
        ]

    async def reinforce_rule(self, rule_id: str, *, positive: bool) -> None:
        connection = await self._ensure_connection()
        column = "success_count" if positive else "fail_count"
        await connection.execute(
            f"""
            UPDATE {self._table_name}
            SET {column} = {column} + 1
            WHERE rule_id = ?
            """,
            (rule_id,),
        )
        await connection.commit()

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
                rule_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                task_category TEXT NOT NULL,
                rule_type TEXT NOT NULL,
                content TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                fail_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        await self._connection.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_tenant_task
            ON {self._table_name} (tenant_id, task_category)
            """
        )
        await self._connection.commit()
        return self._connection


__all__ = [
    "ExperienceRule",
    "ExperienceStore",
    "ReflectionResult",
    "RuleType",
    "SQLiteExperienceStore",
]
