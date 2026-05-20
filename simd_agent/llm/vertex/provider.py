# simd_agent/llm/vertex/provider.py
"""Vertex AI provider — Gemini models served via Google Cloud Vertex AI.

Uses the same ``google-genai`` SDK as the public Gemini provider but
authenticates with Application Default Credentials (a GCP service-account
JSON key) against a project + region instead of an API key.  This lifts
the strict daily request caps of the public AI Studio tier.

Setup — no ``gcloud`` CLI required:

1. In the GCP Console, create a service account and grant it the
   ``roles/aiplatform.user`` role.
2. Download a JSON key for that service account.
3. Enable the Vertex AI API on the project
   (``aiplatform.googleapis.com``).
4. Set the path to that JSON file in ``.env``::

    DEFAULT_PROVIDER=vertex
    VERTEX_PROJECT=<your-gcp-project>
    VERTEX_LOCATION=us-central1
    GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/sa-key.json

The provider injects ``GOOGLE_APPLICATION_CREDENTIALS`` into
``os.environ`` at startup; ``google-auth`` (which ``google-genai`` uses)
then picks it up automatically — no CLI login needed.
"""

from __future__ import annotations

import logging
import os
import types as _builtin_types
from typing import Any, AsyncIterator

from google import genai
from google.genai import types as genai_types

from simd_agent.llm.base import LLMProvider

logger = logging.getLogger(__name__)


def _ensure_vertex_env() -> None:
    """Bridge pydantic-settings → os.environ for the GCP service-account key.

    ``google-genai`` (via ``google-auth``) reads
    ``GOOGLE_APPLICATION_CREDENTIALS`` from ``os.environ`` directly, but
    pydantic-settings does NOT export ``.env`` vars there.  This helper
    fills the gap once per process.  Same pattern as ``storage/gcs.py``.
    """
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    from simd_agent.settings import get_settings
    cred_path = get_settings().google_application_credentials
    if cred_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
        logger.info("[LLM/VERTEX] Injected GOOGLE_APPLICATION_CREDENTIALS=%s", cred_path)


class VertexProvider(LLMProvider):
    """LLM provider backed by Gemini on Google Cloud Vertex AI."""

    name = "vertex"

    def __init__(self) -> None:
        self.models: dict[str, str] = {}
        self._client: genai.Client | None = None
        self._project: str | None = None
        self._location: str | None = None

    # ── lifecycle ────────────────────────────────────────────────

    def configure(self, **kwargs: Any) -> None:
        self._project = kwargs.get("project")
        self._location = kwargs.get("location") or "us-central1"
        self._client = None  # reset so next access picks up new credentials
        self.models = {
            "default": kwargs.get("default_model", "gemini-2.5-flash"),
            "super": kwargs.get("super_model", "gemini-2.5-pro"),
        }

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self._project:
                raise ValueError(
                    "Vertex provider not configured — set VERTEX_PROJECT in .env "
                    "and point GOOGLE_APPLICATION_CREDENTIALS at a service-account "
                    "JSON key with the roles/aiplatform.user role"
                )
            _ensure_vertex_env()
            self._client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
            )
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
        """Vertex transient errors: 503 UNAVAILABLE, 429 RESOURCE_EXHAUSTED."""
        exc_str = str(exc)
        for code in self._RETRYABLE_STATUS_CODES:
            if str(code) in exc_str:
                return True
        return "UNAVAILABLE" in exc_str or "RESOURCE_EXHAUSTED" in exc_str
