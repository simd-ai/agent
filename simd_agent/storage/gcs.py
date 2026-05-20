"""Google Cloud Storage backend.

Stores objects as blobs in a GCS bucket.  Suitable for cloud deployments
where data must survive server restarts and be accessible across instances.

Credentials are loaded from ``GOOGLE_APPLICATION_CREDENTIALS`` (env var or
pydantic-settings).  The bucket name comes from ``STORAGE_BUCKET``.
"""

from __future__ import annotations

import asyncio
import logging
import os

from .base import StorageBackend

logger = logging.getLogger(__name__)


def _ensure_gcs_env() -> None:
    """Bridge pydantic-settings → os.environ for GCS credentials.

    ``google.cloud.storage.Client()`` reads ``GOOGLE_APPLICATION_CREDENTIALS``
    from ``os.environ`` directly, but pydantic-settings does NOT export ``.env``
    vars there.  This helper fills the gap once per process.
    """
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    from simd_agent.settings import get_settings
    cred_path = get_settings().google_application_credentials
    if cred_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
        logger.info("[STORAGE/GCS] Injected GOOGLE_APPLICATION_CREDENTIALS=%s", cred_path)


class GCSBackend(StorageBackend):
    """Store objects in a Google Cloud Storage bucket."""

    def __init__(self, bucket_name: str) -> None:
        _ensure_gcs_env()
        from google.cloud import storage as gcs_storage
        self._client = gcs_storage.Client()
        # Multi-region VTK cache fills can hit GCS with 60+ parallel
        # uploads (per-region VTPs + merged VTPs).  The default urllib3
        # connection pool size is 10, which produces ``Connection pool
        # is full, discarding connection`` warnings and serialises the
        # uploads — pushing the cache-fill time from a few seconds to
        # tens of seconds.  Bump the pool so the bursts fit.
        try:
            adapter = self._client._http.adapters.get("https://")
            if adapter is not None:
                from urllib3.poolmanager import PoolManager
                adapter.poolmanager = PoolManager(
                    num_pools=10, maxsize=100, block=False,
                )
        except Exception as e:
            logger.warning(f"[STORAGE/GCS] Could not enlarge HTTP pool: {e}")
        self._bucket = self._client.bucket(bucket_name)
        logger.info(
            "[STORAGE/GCS] Initialized — project=%s bucket=%s",
            self._client.project,
            bucket_name,
        )

    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        blob = self._bucket.blob(key)
        await asyncio.to_thread(
            blob.upload_from_string, data, content_type=content_type,
        )
        logger.debug(
            "[STORAGE/GCS] Uploaded gs://%s/%s (%d bytes)",
            self._bucket.name, key, len(data),
        )

    async def download(self, key: str) -> bytes | None:
        blob = self._bucket.blob(key)
        if not await asyncio.to_thread(blob.exists):
            return None
        return await asyncio.to_thread(blob.download_as_bytes)

    async def exists(self, key: str) -> bool:
        blob = self._bucket.blob(key)
        return await asyncio.to_thread(blob.exists)

    async def delete(self, key: str) -> None:
        blob = self._bucket.blob(key)
        if await asyncio.to_thread(blob.exists):
            await asyncio.to_thread(blob.delete)

    async def list_keys(self, prefix: str) -> list[str]:
        blobs = await asyncio.to_thread(
            lambda: list(self._client.list_blobs(self._bucket, prefix=prefix)),
        )
        return [b.name for b in blobs]
