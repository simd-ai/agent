# simd_agent/settings.py
"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Database
    database_url: PostgresDsn = Field(
        ...,
        description="Neon Postgres connection URL",
    )
    
    # Sandbox
    sandbox_base_url: str = Field(
        default="https://sandbox.simd.dev",
        description="Base URL for the SIMD sandbox execution service",
    )
    sandbox_timeout: int = Field(
        default=300,
        description="Timeout in seconds for sandbox operations",
    )
    sandbox_poll_interval: float = Field(
        default=2.0,
        description="Polling interval for sandbox status checks",
    )
    
    # Self-healing loop
    max_retries: int = Field(
        default=3,
        description="Maximum number of codegen+sandbox retry attempts",
    )
    
    # LLM Providers
    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key",
    )
    grok_api_key: str | None = Field(
        default=None,
        description="xAI Grok API key",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key",
    )
    
    # Default provider
    default_provider: Literal["gemini3", "grok", "openai", "anthropic", "mock"] = Field(
        default="gemini3",
        description="Default LLM provider to use",
    )
    
    # Prompt packs
    default_prompt_pack: str = Field(
        default="simd",
        description="Default prompt pack name",
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
    
    # Sandbox logs
    max_log_lines_in_event: int = Field(
        default=100,
        description="Maximum number of log lines to include in events",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
