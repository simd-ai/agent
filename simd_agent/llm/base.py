# simd_agent/llm/base.py
"""Abstract base class for LLM providers.

To add a new provider, create a package under ``simd_agent/llm/<name>/``
with an ``__init__.py`` that exports ``provider_plugin = YourProvider()``.
The registry auto-discovers it — no other file needs editing.
"""

from __future__ import annotations

import types as _builtin_types
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class LLMProvider(ABC):
    """Contract every LLM provider must fulfil.

    Attributes:
        name:   Short identifier (e.g. ``"gemini"``).  Must match the
                package directory name.
        models: Mapping of logical role → concrete model ID.
                At minimum, ``"default"`` and ``"super"`` should be present.
    """

    name: str
    models: dict[str, str]

    # ── lifecycle ────────────────────────────────────────────────

    @abstractmethod
    def configure(self, **kwargs: Any) -> None:
        """Initialise the provider with credentials and optional overrides.

        Called once by the registry after reading settings.  Each provider
        defines its own required kwargs (e.g. ``api_key=`` for Gemini,
        ``project=``/``location=`` for Vertex, ``host=`` for Ollama).
        """

    @property
    @abstractmethod
    def client(self) -> Any:
        """Return the underlying SDK client (lazily created).

        Callers that need provider-specific features (tool schemas,
        streaming helpers) access the native client through this property.
        """

    @property
    @abstractmethod
    def types(self) -> _builtin_types.ModuleType:
        """Return the provider's type module.

        For Gemini this is ``google.genai.types``.  Callers that build
        ``Content`` / ``Part`` / ``Tool`` objects import the types from
        the active provider so tool schemas stay provider-specific but
        configuration is centralised.
        """

    # ── generation ───────────────────────────────────────────────

    @abstractmethod
    async def generate(
        self,
        model: str,
        contents: Any,
        *,
        config: Any | None = None,
    ) -> Any:
        """Single-shot generation.  Returns a response whose ``.text``
        attribute contains the model output."""

    @abstractmethod
    async def generate_stream(
        self,
        model: str,
        contents: Any,
        *,
        config: Any | None = None,
    ) -> AsyncIterator[Any]:
        """Streaming generation.  Yields provider-native chunk objects."""

    # ── error classification ────────────────────────────────────

    def is_retryable_error(self, exc: Exception) -> bool:
        """Return True if *exc* is a transient server error worth retrying.

        Override in provider subclasses to recognise provider-specific
        error types (e.g. Gemini 503 / Anthropic 529).  The precheck and
        codegen services use this to decide whether to retry LLM calls
        with exponential backoff.

        The default implementation returns ``False`` (no retry).
        """
        return False
