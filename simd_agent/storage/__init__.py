"""Pluggable object storage for SIMD Agent.

Usage::

    from simd_agent.storage import get_storage

    storage = get_storage()
    await storage.upload("meshes/abc/surface.vtp", vtp_bytes)
    data = await storage.download("meshes/abc/surface.vtp")

The active backend is selected by ``STORAGE_BACKEND`` in settings:

    - ``local`` (default) — files under ``STORAGE_LOCAL_DIR`` (default ``./storage``)
    - ``gcs``             — blobs in ``STORAGE_BUCKET``
"""

from __future__ import annotations

import logging
from functools import lru_cache

from .base import StorageBackend

logger = logging.getLogger(__name__)

__all__ = ["StorageBackend", "get_storage"]


@lru_cache(maxsize=1)
def get_storage() -> StorageBackend:
    """Return the singleton storage backend configured by settings."""
    from simd_agent.settings import get_settings
    settings = get_settings()

    backend = settings.storage_backend

    if backend == "gcs":
        bucket = settings.storage_bucket
        if not bucket:
            raise ValueError(
                "STORAGE_BACKEND=gcs requires STORAGE_BUCKET to be set"
            )
        from .gcs import GCSBackend
        logger.info("[STORAGE] Using GCS backend — bucket=%s", bucket)
        return GCSBackend(bucket)

    if backend == "local":
        from .local import LocalBackend
        logger.info("[STORAGE] Using local backend — dir=%s", settings.storage_local_dir)
        return LocalBackend(settings.storage_local_dir)

    raise ValueError(f"Unknown STORAGE_BACKEND: {backend!r} (expected 'local' or 'gcs')")
