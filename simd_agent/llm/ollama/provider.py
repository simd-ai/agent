# simd_agent/llm/ollama/provider.py
"""Ollama provider — local inference via the Ollama HTTP server.

Default model is ``gemma4`` (Google Gemma 4, April 2026, Apache 2.0).
Pull it with ``ollama pull gemma4`` before first use.

Wire format note — the rest of the codebase builds prompts using
Gemini's ``types`` (Content, Part, GenerateContentConfig, Tool,
ToolConfig, …).  This provider exposes a shim ``types`` module with the
same shapes and translates them to Ollama's wire format inside
``generate()`` / ``generate_stream()``.  See ``_translate.py``.
"""

from __future__ import annotations

import logging
import types as _builtin_types
from typing import Any, AsyncIterator

import httpx

try:
    from ollama import AsyncClient
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "ollama provider requires the 'ollama' package — install with "
        "`pip install ollama>=0.4.0` or `pip install -e .`"
    ) from exc

from simd_agent.llm.base import LLMProvider
from simd_agent.llm.ollama import _types_shim as types_shim
from simd_agent.llm.ollama._translate import (
    build_options_and_format,
    from_ollama_chunk,
    from_ollama_response,
    to_ollama_messages,
    to_ollama_tools,
)

logger = logging.getLogger(__name__)


# ── client shim ─────────────────────────────────────────────────


class _ModelsAPI:
    """Mimics ``google.genai.Client().aio.models`` surface.

    The codebase reaches into the underlying SDK directly in a few
    spots (e.g. ``chat/tools.py``) via
    ``provider.client.aio.models.generate_content(...)``.  Rather than
    rewrite those call sites, we re-expose the same shape here and
    delegate to the provider's own translation pipeline.
    """

    def __init__(self, provider: "OllamaProvider") -> None:
        self._provider = provider

    async def generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: Any | None = None,
    ) -> Any:
        return await self._provider.generate(model, contents, config=config)

    async def generate_content_stream(
        self,
        *,
        model: str,
        contents: Any,
        config: Any | None = None,
    ) -> AsyncIterator[Any]:
        return await self._provider.generate_stream(model, contents, config=config)


class _AioNamespace:
    def __init__(self, provider: "OllamaProvider") -> None:
        self.models = _ModelsAPI(provider)


class _OllamaClientShim:
    """Outer client object — only ``.aio.models`` is used by callers."""

    def __init__(self, provider: "OllamaProvider") -> None:
        self._provider = provider
        self.aio = _AioNamespace(provider)

    @property
    def native(self) -> AsyncClient:
        """Escape hatch — the real ollama AsyncClient if a caller needs it."""
        return self._provider._native_client  # noqa: SLF001


# ── provider ─────────────────────────────────────────────────────


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama server."""

    name = "ollama"

    def __init__(self) -> None:
        self.models: dict[str, str] = {}
        self._host: str | None = None
        self._native_client: AsyncClient | None = None
        self._shim_client: _OllamaClientShim | None = None

    # ── lifecycle ────────────────────────────────────────────────

    def configure(self, **kwargs: Any) -> None:
        self._host = kwargs.get("host") or "http://localhost:11434"
        self._native_client = None  # reset so next access picks up new host
        self._shim_client = None
        self.models = {
            "default": kwargs.get("default_model", "gemma4"),
            "super": kwargs.get("super_model", "gemma4:31b"),
        }

    @property
    def client(self) -> _OllamaClientShim:
        if self._shim_client is None:
            if not self._host:
                raise ValueError(
                    "Ollama provider not configured — set OLLAMA_HOST in .env "
                    "or call configure(api_key=<host_url>)"
                )
            self._native_client = AsyncClient(host=self._host)
            self._shim_client = _OllamaClientShim(self)
        return self._shim_client

    @property
    def types(self) -> _builtin_types.ModuleType:
        # Returning the shim module gives callers access to
        # ``provider.types.GenerateContentConfig`` etc. with the same
        # constructor signatures as ``google.genai.types``.
        return types_shim  # type: ignore[return-value]

    # ── generation ───────────────────────────────────────────────

    async def generate(
        self,
        model: str,
        contents: Any,
        *,
        config: Any | None = None,
    ) -> Any:
        # Ensure the underlying client is ready (touches ``self.client``).
        _ = self.client
        kwargs = self._build_chat_kwargs(model, contents, config, stream=False)
        resp = await self._native_client.chat(**kwargs)  # type: ignore[union-attr]
        return from_ollama_response(resp)

    async def generate_stream(
        self,
        model: str,
        contents: Any,
        *,
        config: Any | None = None,
    ) -> AsyncIterator[Any]:
        _ = self.client
        kwargs = self._build_chat_kwargs(model, contents, config, stream=True)
        ollama_stream = await self._native_client.chat(**kwargs)  # type: ignore[union-attr]
        return _wrap_stream(ollama_stream)

    # ── translation helpers ──────────────────────────────────────

    def _build_chat_kwargs(
        self,
        model: str,
        contents: Any,
        config: Any | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        # Pull system_instruction + tools/tool_config out of config.
        system: str | None = None
        tools: list[dict[str, Any]] | None = None
        fmt: Any = None
        opts: dict[str, Any] = {}
        if config is not None:
            sys_obj = getattr(config, "system_instruction", None)
            if sys_obj is not None:
                system = sys_obj if isinstance(sys_obj, str) else _flatten_system(sys_obj)
            cfg_tools = getattr(config, "tools", None)
            if cfg_tools:
                tools = to_ollama_tools(cfg_tools)
            tool_cfg = getattr(config, "tool_config", None)
            if tool_cfg is not None:
                fcc = getattr(tool_cfg, "function_calling_config", None)
                mode = (getattr(fcc, "mode", None) or "AUTO").upper() if fcc else "AUTO"
                if mode == "NONE":
                    tools = None
            opts, fmt = build_options_and_format(config)

        messages = to_ollama_messages(contents, system_instruction=system)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
        if opts:
            kwargs["options"] = opts
        if fmt is not None:
            kwargs["format"] = fmt
        return kwargs

    # ── error classification ────────────────────────────────────

    def is_retryable_error(self, exc: Exception) -> bool:
        """Ollama transient errors: connection refused, timeouts, 5xx."""
        if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)):
            return True
        msg = str(exc)
        if any(s in msg for s in ("503", "502", "504", "Connection refused", "timeout")):
            return True
        return False


# ── streaming helpers ────────────────────────────────────────────


async def _wrap_stream(ollama_stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Wrap an Ollama async generator so each chunk is Gemini-shaped."""
    async for chunk in ollama_stream:
        yield from_ollama_chunk(chunk)


def _flatten_system(obj: Any) -> str:
    """Accept Content / Part / str for system_instruction."""
    parts = getattr(obj, "parts", None)
    if parts:
        return "\n".join(getattr(p, "text", "") or "" for p in parts)
    return str(obj)
