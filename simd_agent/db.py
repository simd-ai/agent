# simd_agent/db.py
"""Async database engine and session management using SQLAlchemy 2.0."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import MetaData, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from simd_agent.settings import get_settings

# Naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)

# Module-level engine and session factory (initialized lazily)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url() -> str:
    """Get async database URL from settings.
    
    Converts standard postgres URLs to asyncpg format and handles
    Neon-specific parameters that asyncpg doesn't support.
    """
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
    
    settings = get_settings()
    url = str(settings.database_url)
    
    # Convert postgres:// to postgresql+asyncpg://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    # Parse URL to handle query parameters properly
    parsed = urlparse(url)
    
    if parsed.query:
        # Parse query parameters
        params = parse_qs(parsed.query, keep_blank_values=True)
        
        # Parameters that asyncpg doesn't support (libpq-specific)
        unsupported_params = {
            "sslmode",        # asyncpg uses 'ssl' instead
            "channel_binding",  # Not supported by asyncpg
            "options",        # libpq-specific
            "application_name",  # Handled via connect_args in SQLAlchemy
        }
        
        # Convert sslmode to ssl if present
        if "sslmode" in params:
            ssl_value = params.pop("sslmode")[0]
            params["ssl"] = [ssl_value]
        
        # Remove other unsupported parameters
        for param in unsupported_params:
            params.pop(param, None)
        
        # Flatten params (parse_qs returns lists)
        flat_params = {k: v[0] for k, v in params.items()}
        new_query = urlencode(flat_params) if flat_params else ""
        
        # Rebuild URL
        url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment,
        ))
    
    return url


def get_engine() -> AsyncEngine:
    """Get or create the async database engine."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            get_database_url(),
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize database tables."""
    engine = get_engine()
    async with engine.begin() as conn:
        # Create tables using raw SQL for simplicity
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS runs (
                id UUID PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                op VARCHAR(50) NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                provider VARCHAR(50) NOT NULL,
                prompt_pack VARCHAR(100) NOT NULL,
                user_requirements TEXT NOT NULL,
                simulation_config JSONB NOT NULL DEFAULT '{}',
                validated_config JSONB,
                attempts INTEGER NOT NULL DEFAULT 0,
                result JSONB
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS events (
                id UUID PRIMARY KEY,
                run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                ts TIMESTAMPTZ NOT NULL,
                level VARCHAR(10) NOT NULL,
                type VARCHAR(100) NOT NULL,
                message TEXT NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'
            )
        """))
        
        # Create indexes
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, seq)
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)
        """))


async def close_db() -> None:
    """Close database connections.

    Uses a short timeout so that a slow or unreachable remote DB (e.g. Neon)
    during Ctrl+C shutdown doesn't block the process for 60 seconds and print
    a noisy TimeoutError traceback.  Any connection that can't be closed within
    the timeout is abandoned — the OS will clean it up.
    """
    global _engine, _session_factory
    if _engine is not None:
        try:
            await asyncio.wait_for(_engine.dispose(), timeout=3.0)
        except (asyncio.TimeoutError, Exception):
            # Best-effort: force the pool closed without waiting for TCP teardown
            try:
                _engine.sync_engine.pool.dispose()
            except Exception:
                pass
        finally:
            _engine = None
            _session_factory = None
