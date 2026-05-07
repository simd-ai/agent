# simd_agent/db.py
"""Async database engine and session management using SQLAlchemy 2.0.

Uses Neon Postgres with the asyncpg driver.  Key settings:
  - pool_pre_ping=True  — handles Neon cold-start disconnects
  - pool_recycle=300     — recycle before Neon's idle timeout (5 min)
  - pool_size=5          — small since Neon's built-in pooler handles the rest
"""

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
            pool_pre_ping=True,    # handles Neon cold-start disconnects
            pool_size=5,           # small — Neon pooler handles the rest
            max_overflow=5,
            pool_recycle=300,      # recycle before Neon's idle timeout
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


async def _ensure_columns(conn) -> None:
    """Add columns that may be missing on tables created by an older schema.

    Each statement uses ADD COLUMN IF NOT EXISTS so it is safe to run
    repeatedly.  This covers the gap between the old store.py schema
    (which only had id/op/provider/…) and the full schema above.
    """
    _run_cols = [
        ("simulation_id", "UUID REFERENCES simulations(id) ON DELETE CASCADE"),
        ("label", "TEXT"),
        ("type", "TEXT NOT NULL DEFAULT 'full'"),
        ("solver", "VARCHAR(100)"),
        ("prompt_pack", "VARCHAR(100)"),
        ("user_requirements", "TEXT"),
        ("simulation_config", "JSONB NOT NULL DEFAULT '{}'"),
        ("validated_config", "JSONB"),
        ("attempts", "INTEGER NOT NULL DEFAULT 0"),
        ("lint_result", "JSONB"),
        ("planning_result", "JSONB"),
        ("generated_files", "JSONB"),
        ("file_generation_map", "JSONB"),
        ("final_result", "JSONB"),
        ("vtk_result", "JSONB"),
        ("error_message", "TEXT"),
        ("user_prompt_snapshot", "TEXT"),
        ("started_at", "TIMESTAMPTZ NOT NULL DEFAULT NOW()"),
        ("completed_at", "TIMESTAMPTZ"),
    ]
    for col, typedef in _run_cols:
        await conn.execute(text(
            f"ALTER TABLE runs ADD COLUMN IF NOT EXISTS {col} {typedef}"
        ))

    _user_cols = [
        ("stripe_customer_id", "TEXT"),
        ("subscription_status", "TEXT NOT NULL DEFAULT 'free'"),
        ("subscription_current_period_end", "TIMESTAMPTZ"),
    ]
    for col, typedef in _user_cols:
        await conn.execute(text(
            f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {col} {typedef}"
        ))

    # sim_progress — field_ranges column for fieldMinMax data
    await conn.execute(text(
        "ALTER TABLE sim_progress ADD COLUMN IF NOT EXISTS field_ranges JSONB"
    ))


    # ── FK migration: re-point FKs from old "agent_runs" → "runs" ───
    # The old Drizzle schema created sim_progress/events/lint_reports with
    # FKs referencing agent_runs(id).  The backend canonical table is
    # "runs", so re-point the FKs.  Safe to run repeatedly.
    _fk_fixes = [
        ("sim_progress", "sim_progress_run_id_fkey", "sim_progress_run_id_agent_runs_id_fk"),
        ("events",       "events_run_id_fkey",       "events_run_id_agent_runs_id_fk"),
    ]
    for tbl, pg_name, drizzle_name in _fk_fixes:
        try:
            await conn.execute(text(
                f"ALTER TABLE {tbl} DROP CONSTRAINT IF EXISTS {pg_name}"
            ))
            await conn.execute(text(
                f"ALTER TABLE {tbl} DROP CONSTRAINT IF EXISTS {drizzle_name}"
            ))
            await conn.execute(text(
                f"ALTER TABLE {tbl} "
                f"ADD CONSTRAINT {pg_name} "
                f"FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE"
            ))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("%s FK migration: %s", tbl, e)


async def init_db() -> None:
    """Initialize database tables.

    Creates all tables needed for the backend to be the system of record.
    Schema mirrors the frontend's Drizzle schema so the frontend can drop
    its direct DB connection and use these REST endpoints instead.

    Tables:
      - users:             user accounts
      - simulations:       saved simulations (belong to a user)
      - simulation_config: CFD physics/solver/fluid/turbulence (1:1 with simulations)
      - mesh_info:         mesh metadata per simulation (1:1 with simulations)
      - patch_configs:     boundary condition per patch (many per simulation)
      - runs:              codegen+simulation attempts (many per simulation)
      - events:            streaming event log per run
      - sim_progress:      solver convergence data per run
      - chat_messages:     user/assistant conversation per simulation
      - precheck_history:  LLM precheck analysis (1:1 with simulations)
      - lint_reports:      validation results (many per simulation)
    """
    engine = get_engine()
    async with engine.begin() as conn:
        # ── Users ────────────────────────────────────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                stripe_customer_id TEXT,
                subscription_status TEXT NOT NULL DEFAULT 'free',
                subscription_current_period_end TIMESTAMPTZ
            )
        """))

        # ── Simulations ──────────────────────────────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS simulations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT 'Untitled Simulation',
                active_step SMALLINT NOT NULL DEFAULT 1,
                max_reached_step SMALLINT NOT NULL DEFAULT 1,
                selected_preset_id TEXT,
                user_prompt TEXT,
                expert_mode BOOLEAN NOT NULL DEFAULT FALSE,
                is_from_scratch_mode BOOLEAN NOT NULL DEFAULT FALSE,
                active_tab TEXT NOT NULL DEFAULT 'viewer',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))

        # ── Simulation Config (1:1) ─────────────────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS simulation_config (
                simulation_id UUID PRIMARY KEY REFERENCES simulations(id) ON DELETE CASCADE,
                case_spec JSONB,
                cfd_physics JSONB,
                cfd_solver JSONB,
                cfd_fluid JSONB,
                cfd_turbulence JSONB,
                cfd_derived JSONB
            )
        """))

        # ── Mesh Info (1:1) ──────────────────────────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS mesh_info (
                simulation_id UUID PRIMARY KEY REFERENCES simulations(id) ON DELETE CASCADE,
                mesh_id TEXT NOT NULL,
                file_name TEXT,
                uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                patches JSONB,
                viewer_artifacts JSONB,
                check_mesh JSONB
            )
        """))

        # ── Patch Configs (many per simulation) ──────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS patch_configs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                simulation_id UUID NOT NULL REFERENCES simulations(id) ON DELETE CASCADE,
                patch_name TEXT NOT NULL,
                patch_class TEXT,
                patch_config JSONB,
                patch_info JSONB,
                boundary_hint JSONB,
                status TEXT NOT NULL DEFAULT 'needs_config',
                UNIQUE (simulation_id, patch_name)
            )
        """))

        # ── Runs ─────────────────────────────────────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS runs (
                id UUID PRIMARY KEY,
                simulation_id UUID REFERENCES simulations(id) ON DELETE CASCADE,
                label TEXT,
                type TEXT NOT NULL DEFAULT 'full',
                status TEXT NOT NULL DEFAULT 'running',
                op VARCHAR(50),
                provider VARCHAR(50),
                prompt_pack VARCHAR(100),
                user_requirements TEXT,
                simulation_config JSONB NOT NULL DEFAULT '{}',
                validated_config JSONB,
                solver VARCHAR(100),
                attempts INTEGER NOT NULL DEFAULT 0,
                lint_result JSONB,
                planning_result JSONB,
                generated_files JSONB,
                file_generation_map JSONB,
                final_result JSONB,
                vtk_result JSONB,
                error_message TEXT,
                user_prompt_snapshot TEXT,
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ,
                result JSONB
            )
        """))

        # ── Events ───────────────────────────────────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS events (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                ts TIMESTAMPTZ NOT NULL,
                level VARCHAR(10) NOT NULL,
                type VARCHAR(100) NOT NULL,
                message TEXT NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'
            )
        """))

        # ── Sim Progress (convergence data per run) ──────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sim_progress (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                iteration INTEGER NOT NULL,
                sim_time REAL,
                fields JSONB,
                residuals JSONB,
                courant JSONB,
                continuity JSONB,
                execution JSONB,
                field_ranges JSONB
            )
        """))

        # ── Chat Messages ────────────────────────────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                simulation_id UUID NOT NULL REFERENCES simulations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                suggested_actions JSONB,
                artifacts JSONB,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))

        # ── Precheck History (1:1) ───────────────────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS precheck_history (
                simulation_id UUID PRIMARY KEY REFERENCES simulations(id) ON DELETE CASCADE,
                submitted_prompt TEXT,
                mesh_name TEXT,
                mesh_cells INTEGER,
                steps JSONB,
                review_thoughts TEXT,
                review_items JSONB,
                suggested_config JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))

        # ── Lint Reports ─────────────────────────────────────────────────
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS lint_reports (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                simulation_id UUID NOT NULL REFERENCES simulations(id) ON DELETE CASCADE,
                run_id UUID REFERENCES runs(id) ON DELETE SET NULL,
                is_valid BOOLEAN NOT NULL,
                issues JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))

        # ── Migrations: add columns that may be missing on older tables ──
        # (CREATE TABLE IF NOT EXISTS won't alter an existing table)
        await _ensure_columns(conn)

        # ── Indexes ──────────────────────────────────────────────────────
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_simulations_user ON simulations(user_id)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_runs_simulation ON runs(simulation_id)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, seq)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_patch_configs_simulation ON patch_configs(simulation_id)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_sim_progress_run ON sim_progress(run_id)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_simulation ON chat_messages(simulation_id)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_lint_reports_simulation ON lint_reports(simulation_id)"
        ))


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
