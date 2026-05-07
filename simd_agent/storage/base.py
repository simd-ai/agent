"""Abstract storage backend for binary object storage.

All large binary data (meshes, VTP results, case ZIPs) goes through this
interface.  Implementations live alongside this file:

    - ``LocalBackend``  — local filesystem (dev / self-hosted)
    - ``GCSBackend``    — Google Cloud Storage (cloud deployment)

The active backend is selected by the ``STORAGE_BACKEND`` env var.
"""

from __future__ import annotations

import abc
from typing import AsyncIterator


class StorageBackend(abc.ABC):
    """Minimal contract for object storage."""

    # ── Core operations ───────────────────────────────────────────

    @abc.abstractmethod
    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Store *data* under *key*.  Overwrites if key already exists."""

    @abc.abstractmethod
    async def download(self, key: str) -> bytes | None:
        """Return the bytes at *key*, or ``None`` if it does not exist."""

    @abc.abstractmethod
    async def exists(self, key: str) -> bool:
        """Return ``True`` if *key* exists in the store."""

    @abc.abstractmethod
    async def delete(self, key: str) -> None:
        """Delete *key*.  No-op if it does not exist."""

    @abc.abstractmethod
    async def list_keys(self, prefix: str) -> list[str]:
        """Return all keys that start with *prefix*."""
