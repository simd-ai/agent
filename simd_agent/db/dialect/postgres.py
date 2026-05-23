"""Postgres dialect — the production backend (Neon)."""

from __future__ import annotations

from .base import Dialect


_POSTGRES_SCHEMES = ("postgres://", "postgresql://", "postgresql+asyncpg://")


class PostgresDialect(Dialect):
    name = "postgres"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return url.startswith(_POSTGRES_SCHEMES)

    def translate_runtime_sql(self, sql: str) -> str:
        return sql

    def translate_ddl(self, sql: str) -> str:
        return sql
