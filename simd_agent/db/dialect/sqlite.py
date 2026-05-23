"""SQLite dialect — single-file backend used for local installs and tests."""

from __future__ import annotations

import re
import sqlite3
import uuid

from .base import Dialect


# Bind ``uuid.UUID`` instances to their canonical string form when the active
# backend is SQLite.  The codebase carries UUIDs as ``UUID`` objects through
# every layer (run_id, event_id, simulation_id …); Python's stock sqlite3
# driver — which aiosqlite wraps — raises ``ProgrammingError: Error binding
# parameter`` if it sees a UUID directly.  Registering once at import time
# means all aiosqlite connections in this process get the conversion for
# free, without sprinkling ``str(...)`` casts through every call site.
sqlite3.register_adapter(uuid.UUID, str)


_RUNTIME_PARAM_CAST = re.compile(r"(:[A-Za-z_][A-Za-z0-9_]*)::(jsonb|text)")
_RUNTIME_CAST_AS_JSONB = re.compile(
    r"CAST\(\s*(:[A-Za-z_][A-Za-z0-9_]*)\s+AS\s+jsonb\s*\)",
    flags=re.IGNORECASE,
)
_RUNTIME_NOW = re.compile(r"\bNOW\(\)")


_DDL_REPL: list[tuple[str, str]] = [
    # UUID with default — drop the default; app provides uuid4().hex
    (r"UUID\s+PRIMARY\s+KEY\s+DEFAULT\s+gen_random_uuid\(\)", "TEXT PRIMARY KEY"),
    (r"UUID\s+PRIMARY\s+KEY",                                  "TEXT PRIMARY KEY"),
    (r"\bUUID\b",                                              "TEXT"),
    # Timestamps
    (r"TIMESTAMPTZ\s+NOT\s+NULL\s+DEFAULT\s+NOW\(\)",          "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"),
    (r"\bTIMESTAMPTZ\b",                                       "TEXT"),
    # JSON
    (r"\bJSONB\b",                                             "TEXT"),
    # Booleans — SQLite has no boolean type, store as 0/1 integers
    (r"BOOLEAN\s+NOT\s+NULL\s+DEFAULT\s+FALSE",                "INTEGER NOT NULL DEFAULT 0"),
    (r"BOOLEAN\s+NOT\s+NULL\s+DEFAULT\s+TRUE",                 "INTEGER NOT NULL DEFAULT 1"),
    (r"\bBOOLEAN\b",                                           "INTEGER"),
    # Numeric sizes
    (r"\bSMALLINT\b",                                          "INTEGER"),
    (r"VARCHAR\(\s*\d+\s*\)",                                  "TEXT"),
]


class SqliteDialect(Dialect):
    name = "sqlite"

    @classmethod
    def matches_url(cls, url: str) -> bool:
        return url.startswith("sqlite")

    def translate_runtime_sql(self, sql: str) -> str:
        sql = _RUNTIME_PARAM_CAST.sub(r"\1", sql)
        sql = _RUNTIME_CAST_AS_JSONB.sub(r"\1", sql)
        sql = sql.replace("::jsonb", "")
        sql = sql.replace("::text", "")
        sql = _RUNTIME_NOW.sub("CURRENT_TIMESTAMP", sql)
        return sql

    def translate_ddl(self, sql: str) -> str:
        for pat, sub in _DDL_REPL:
            sql = re.sub(pat, sub, sql, flags=re.IGNORECASE)
        return sql
