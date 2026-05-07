# simd_agent/llm/gemini/provider.py
"""Google Gemini provider implementation.

Wraps the ``google-genai`` SDK.  All Gemini-specific configuration
(API key, model names) flows through :meth:`configure`, which the
registry calls once at startup.
"""

from __future__ import annotations

import types as _builtin_types
from typing import Any, AsyncIterator

from google import genai
from google.genai import types as genai_types

from simd_agent.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """LLM provider backed by the Google Gemini API."""

    name = "gemini"

    def __init__(self) -> None:
        self.models: dict[str, str] = {}
        self._client: genai.Client | None = None
        self._api_key: str | None = None

    # ── lifecycle ────────────────────────────────────────────────

    def configure(self, api_key: str, **kwargs: Any) -> None:
        self._api_key = api_key
        self._client = None  # reset so next access picks up new key
        self.models = {
            "default": kwargs.get("default_model", "gemini-3-flash-preview"),
            "super": kwargs.get("super_model", "gemini-3.1-pro-preview"),
        }

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "Gemini provider not configured — call configure(api_key=...) "
                    "or set GEMINI_API_KEY in .env"
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def types(self) -> _builtin_types.ModuleType:
        return genai_types

    # ── generation ───────────────────────────────────────────────

    async def generate(
        self,
        model: str,
        contents: Any,
        *,
        config: Any | None = None,
    ) -> Any:
        return await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    async def generate_stream(
        self,
        model: str,
        contents: Any,
        *,
        config: Any | None = None,
    ) -> AsyncIterator[Any]:
        return await self.client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )

    # ── error classification ────────────────────────────────────

    _RETRYABLE_STATUS_CODES = {429, 503}

    def is_retryable_error(self, exc: Exception) -> bool:
        """Gemini transient errors: 503 UNAVAILABLE, 429 RESOURCE_EXHAUSTED."""
        exc_str = str(exc)
        for code in self._RETRYABLE_STATUS_CODES:
            if str(code) in exc_str:
                return True
        return "UNAVAILABLE" in exc_str or "RESOURCE_EXHAUSTED" in exc_str
