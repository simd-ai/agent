# simd_agent/settings.py 
"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to the repo root (one level up from this file's package dir)
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

class Settings(BaseSettings):
    """Application configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: PostgresDsn = Field(
        ...,
        description="Neon Postgres connection URL",
    )

    # Simulation Server (OpenFOAM runner — configurable endpoint)
    simulation_server_url: str = Field(
        description="Base URL for the SIMD Simulation Runner server (OpenFOAM). "
                    "Must be set via SIMULATION_SERVER_URL in .env.",
    )

    # LLM Provider — the registry reads these to configure the active provider.
    # To add a new provider, create a package under simd_agent/llm/<name>/ and
    # set default_provider to its name.
    default_provider: str = Field(
        default="gemini",
        description="LLM provider name (must match a package under simd_agent/llm/)",
    )
    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key",
    )
    gemini_model: str = Field(
        default="gemini-3-flash-preview",
        description="Default Gemini model for code generation",
    )
    gemini_super_model: str = Field(
        default="gemini-3.1-pro-preview",
        description="High-capacity Gemini model used for solver selection and verification",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # WebSocket
    ws_heartbeat_interval: int = Field(
        default=30,
        description="WebSocket heartbeat interval in seconds",
    )

    # Simulation logs
    max_log_lines_in_event: int = Field(
        default=100,
        description="Maximum number of log lines to include in events",
    )

    # Auth
    neon_auth_base_url: str | None = Field(
        default=None,
        description="Neon Auth server URL for session validation (e.g. https://ep-xxx.neonauth.us-east-1.aws.neon.tech)",
    )

    # ── Object storage (meshes, VTP results, case ZIPs) ─────────
    #
    # Two backends:
    #   local  — files under STORAGE_LOCAL_DIR (default ./storage)
    #   gcs    — blobs in STORAGE_BUCKET (Google Cloud Storage)
    #
    # Typical combos:
    #   Cloud:       STORAGE_BACKEND=gcs   + Neon Postgres
    #   Self-hosted: STORAGE_BACKEND=local + local Postgres
    storage_backend: str = Field(
        default="local",
        description="Object storage backend: 'local' (filesystem) or 'gcs' (Google Cloud Storage)",
    )
    storage_bucket: str | None = Field(
        default=None,
        description="GCS bucket name (required when STORAGE_BACKEND=gcs)",
    )
    storage_local_dir: str = Field(
        default="./storage",
        description="Root directory for local object storage (used when STORAGE_BACKEND=local)",
    )

    # Progress data — convergence residuals stored as NDJSON + zstd.
    # Buffered locally during the run, then uploaded to GCS on completion.
    progress_data_dir: str = Field(
        default="/tmp/simd_progress",
        description="Local temp directory for in-flight progress NDJSON files",
    )
    progress_gcs_bucket: str | None = Field(
        default=None,
        description="GCS bucket for finalized progress data (e.g. 'simd-progress'). "
                    "If unset, compressed files stay in progress_data_dir.",
    )

    # Google Cloud credentials file (service account key JSON).
    # google.cloud.storage.Client() reads this from os.environ, but
    # pydantic-settings doesn't export .env vars to os.environ — so we
    # read it here and inject it during app startup.
    google_application_credentials: str | None = Field(
        default=None,
        description="Path to GCS service account key JSON file",
    )

    # ── Usage / tier limits ──────────────────────────────────────
    free_max_projects: int = Field(
        default=10,
        description="Maximum number of projects for free-tier users",
    )
    free_max_runs: int = Field(
        default=20,
        description="Maximum number of simulation runs for free-tier users",
    )

    # ── Telemetry (Umami) ────────────────────────────────────
    telemetry_enabled: bool = Field(
        default=True,
        description="Enable anonymized usage telemetry via Umami. Set to false to opt out.",
    )
    umami_host_url: str = Field(
        default="https://cloud.umami.is",
        description="Umami instance base URL.",
    )
    umami_website_id: str | None = Field(
        default=None,
        description="Umami website ID. Telemetry is silently disabled if not set.",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
