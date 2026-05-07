from __future__ import annotations

# simd_agent/repositories/base.py
"""Abstract repository interface and Postgres implementation.

To swap databases, implement BaseRepository for your backend and register
it in the dependency injection setup (see services/__init__.py).
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from simd_agent.db import get_session

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """Abstract data access interface — one implementation per database backend."""

    @abstractmethod
    async def get_by_id(self, id: UUID) -> dict[str, Any] | None: ...

    @abstractmethod
    async def list(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str = "created_at DESC",
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def create(self, data: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    async def update(self, id: UUID, data: dict[str, Any]) -> dict[str, Any] | None: ...

    @abstractmethod
    async def delete(self, id: UUID) -> bool: ...

    @abstractmethod
    async def upsert(
        self,
        data: dict[str, Any],
        conflict_keys: list[str],
        update_keys: list[str] | None = None,
    ) -> dict[str, Any]: ...


class PostgresRepository(BaseRepository):
    """Postgres implementation using SQLAlchemy async sessions.

    Each subclass sets:
      - table: str           — table name
      - columns: list[str]   — columns to SELECT (use '::text' for timestamps)
      - pk: str              — primary key column name (default 'id')
      - json_columns: set    — columns that need json.dumps() on write
    """

    table: str
    columns: list[str]
    pk: str = "id"
    json_columns: set[str] = set()

    @property
    def _select_cols(self) -> str:
        return ", ".join(self.columns)

    def _serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        """Serialize JSON columns before writing."""
        out = {}
        for k, v in data.items():
            if k in self.json_columns and v is not None:
                out[k] = json.dumps(v) if not isinstance(v, str) else v
            else:
                out[k] = v
        return out

    async def get_by_id(self, id: UUID) -> dict[str, Any] | None:
        async with get_session() as session:
            result = await session.execute(
                text(f"SELECT {self._select_cols} FROM {self.table} WHERE {self.pk} = :id"),
                {"id": id},
            )
            row = result.mappings().one_or_none()
            return dict(row) if row else None

    async def list(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str = "created_at DESC",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        conditions = []
        params: dict[str, Any] = {}

        if filters:
            for key, val in filters.items():
                conditions.append(f"{key} = :{key}")
                params[key] = val

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT :_limit" if limit else ""
        if limit:
            params["_limit"] = limit

        async with get_session() as session:
            result = await session.execute(
                text(f"SELECT {self._select_cols} FROM {self.table} {where} ORDER BY {order_by} {limit_clause}"),
                params,
            )
            return [dict(row) for row in result.mappings().all()]

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized = self._serialize(data)
        cols = ", ".join(serialized.keys())
        placeholders = ", ".join(f":{k}" for k in serialized.keys())

        async with get_session() as session:
            result = await session.execute(
                text(f"INSERT INTO {self.table} ({cols}) VALUES ({placeholders}) RETURNING {self._select_cols}"),
                serialized,
            )
            return dict(result.mappings().one())

    async def update(self, id: UUID, data: dict[str, Any]) -> dict[str, Any] | None:
        if not data:
            return await self.get_by_id(id)

        serialized = self._serialize(data)
        set_clauses = ", ".join(f"{k} = :{k}" for k in serialized.keys())
        serialized["_pk"] = id

        async with get_session() as session:
            result = await session.execute(
                text(f"UPDATE {self.table} SET {set_clauses} WHERE {self.pk} = :_pk RETURNING {self._select_cols}"),
                serialized,
            )
            row = result.mappings().one_or_none()
            return dict(row) if row else None

    async def delete(self, id: UUID) -> bool:
        async with get_session() as session:
            result = await session.execute(
                text(f"DELETE FROM {self.table} WHERE {self.pk} = :id RETURNING {self.pk}"),
                {"id": id},
            )
            return result.scalar() is not None

    async def upsert(
        self,
        data: dict[str, Any],
        conflict_keys: list[str],
        update_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        serialized = self._serialize(data)
        cols = ", ".join(serialized.keys())
        placeholders = ", ".join(f":{k}" for k in serialized.keys())
        conflict = ", ".join(conflict_keys)

        if update_keys is None:
            update_keys = [k for k in serialized.keys() if k not in conflict_keys]

        updates = ", ".join(
            f"{k} = COALESCE(EXCLUDED.{k}, {self.table}.{k})" for k in update_keys
        )

        async with get_session() as session:
            result = await session.execute(
                text(f"""
                    INSERT INTO {self.table} ({cols}) VALUES ({placeholders})
                    ON CONFLICT ({conflict}) DO UPDATE SET {updates}
                    RETURNING {self._select_cols}
                """),
                serialized,
            )
            return dict(result.mappings().one())

    async def execute_raw(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Escape hatch for complex queries that don't fit the standard patterns."""
        async with get_session() as session:
            result = await session.execute(text(query), params or {})
            return [dict(row) for row in result.mappings().all()]

    async def execute_raw_one(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Execute a raw query expecting zero or one result."""
        rows = await self.execute_raw(query, params)
        return rows[0] if rows else None

    async def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Execute a raw write query (INSERT/UPDATE/DELETE) with no return."""
        async with get_session() as session:
            await session.execute(text(query), params or {})
