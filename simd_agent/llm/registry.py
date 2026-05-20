# simd_agent/llm/registry.py
"""Auto-discovering LLM provider registry.

Mirrors the solver plugin pattern: drop a new directory into
``simd_agent/llm/<provider_name>/`` with an ``__init__.py`` that
exports ``provider_plugin``, and the registry picks it up.

Usage::

    from simd_agent.llm import get_provider

    provider = get_provider()            # default provider
    provider = get_provider("gemini")    # explicit name
    resp = await provider.generate(provider.models["default"], contents=...)
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from simd_agent.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_PROVIDERS_PKG = "simd_agent.llm"
_PROVIDERS_DIR = Path(__file__).resolve().parent


class LLMRegistry:
    """Registry that auto-discovers provider packages under ``simd_agent/llm/``."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}
        self._default_name: str | None = None
        self._discover()

    # ── discovery ────────────────────────────────────────────────

    def _discover(self) -> None:
        """Walk sub-packages and import those that export ``provider_plugin``."""
        for info in pkgutil.iter_modules([str(_PROVIDERS_DIR)]):
            if not info.ispkg:
                continue
            name = info.name
            try:
                mod = importlib.import_module(f"{_PROVIDERS_PKG}.{name}")
                plugin: LLMProvider | None = getattr(mod, "provider_plugin", None)
                if plugin is None:
                    continue
                self._providers[plugin.name] = plugin
                logger.debug("LLM provider discovered: %s", plugin.name)
            except Exception:
                logger.warning("Failed to load LLM provider '%s'", name, exc_info=True)

    # ── public API ───────────────────────────────────────────────

    def configure_from_settings(self) -> None:
        """Read settings and configure the appropriate provider(s).

        Called once at startup (from ``main.py`` lifespan or lazily on
        first ``get_provider()`` call).
        """
        from simd_agent.settings import get_settings

        settings = get_settings()
        self._default_name = settings.default_provider

        # Gemini (public AI Studio API key)
        gemini = self._providers.get("gemini")
        if gemini is not None:
            api_key = (
                settings.gemini_api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            if api_key:
                gemini.configure(
                    api_key=api_key,
                    default_model=settings.gemini_model,
                    super_model=settings.gemini_super_model,
                )

        # Vertex AI (Gemini via GCP).  Uses Application Default
        # Credentials — no API key.  Skipped unless VERTEX_PROJECT is set.
        vertex = self._providers.get("vertex")
        if vertex is not None and settings.vertex_project:
            vertex.configure(
                project=settings.vertex_project,
                location=settings.vertex_location,
                default_model=settings.vertex_model,
                super_model=settings.vertex_super_model,
            )

        # Ollama (local) — no authentication, just an HTTP host.
        ollama = self._providers.get("ollama")
        if ollama is not None:
            ollama.configure(
                host=settings.ollama_host,
                default_model=settings.ollama_model,
                super_model=settings.ollama_super_model,
            )

    @property
    def providers(self) -> dict[str, LLMProvider]:
        return dict(self._providers)

    def get(self, name: str | None = None) -> LLMProvider:
        """Return a configured provider by name (or the default).

        Accepts legacy aliases (e.g. ``"gemini3"`` → ``"gemini"``).
        """
        target = name or self._default_name or "gemini"
        provider = self._providers.get(target)
        # Fallback: strip trailing version digits (gemini3 → gemini)
        if provider is None:
            stripped = target.rstrip("0123456789")
            provider = self._providers.get(stripped)
        if provider is None:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise ValueError(
                f"LLM provider '{target}' not found. Available: {available}"
            )
        return provider

    def list_providers(self) -> list[str]:
        return sorted(self._providers.keys())


# ── module-level singleton ───────────────────────────────────────

_registry: LLMRegistry | None = None


def get_llm_registry() -> LLMRegistry:
    """Return the singleton registry, creating + configuring it on first call."""
    global _registry
    if _registry is None:
        _registry = LLMRegistry()
        _registry.configure_from_settings()
    return _registry


def get_provider(name: str | None = None) -> LLMProvider:
    """Convenience: ``get_provider()`` → default configured provider."""
    return get_llm_registry().get(name)
