"""Dialect abstraction for the dual-backend (SQLite + Postgres) data layer.

Every dialect-specific behavior — URL detection, SQL translation, runtime
quirks — lives behind this interface so the rest of the data layer can stay
backend-agnostic.  Today the implementations are :class:`SqliteDialect` and
:class:`PostgresDialect`; adding a third dialect means writing one subclass,
not sprinkling ``if is_sqlite():`` branches across the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Dialect(ABC):
    """Encapsulates per-backend differences for the data layer."""

    name: str

    @classmethod
    @abstractmethod
    def matches_url(cls, url: str) -> bool:
        """True if ``url`` is handled by this dialect."""

    @abstractmethod
    def translate_runtime_sql(self, sql: str) -> str:
        """Translate a Postgres-flavored runtime query for this dialect.

        Call sites author one query string that runs on every backend; the
        dialect rewrites Postgres-only fragments (``::jsonb`` casts, ``NOW()``,
        …) into the dialect's equivalents.  Postgres is a no-op.
        """

    @abstractmethod
    def translate_ddl(self, sql: str) -> str:
        """Translate a Postgres-flavored ``CREATE TABLE`` for this dialect.

        The canonical schema is authored in Postgres syntax (TIMESTAMPTZ,
        JSONB, ``gen_random_uuid()`` …) so the production deploy keeps full
        type fidelity.  SQLite gets equivalents with the same application-level
        semantics; Postgres is a no-op.
        """
