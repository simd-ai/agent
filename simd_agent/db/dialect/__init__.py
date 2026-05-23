"""Dialect registry — selects the right :class:`Dialect` for a database URL."""

from __future__ import annotations

from .base import Dialect
from .postgres import PostgresDialect
from .sqlite import SqliteDialect

# Order matters only when schemes overlap; today they don't.
_DIALECTS: tuple[type[Dialect], ...] = (SqliteDialect, PostgresDialect)


def get_dialect(url: str) -> Dialect:
    """Return the dialect that handles ``url``.

    Raises :class:`ValueError` for unrecognised schemes — surfacing config
    typos at startup rather than as confusing query failures later.
    """
    for dialect_cls in _DIALECTS:
        if dialect_cls.matches_url(url):
            return dialect_cls()
    raise ValueError(f"Unknown database URL scheme: {url!r}")


__all__ = ["Dialect", "PostgresDialect", "SqliteDialect", "get_dialect"]
