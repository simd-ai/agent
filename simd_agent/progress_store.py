# simd_agent/progress_store.py
"""
Non-blocking progress storage: in-memory buffer → local temp → GCS + zstd.

Design
──────
Simulation convergence data (residuals, Courant, continuity, field ranges)
arrives every solver iteration — thousands of entries for a typical run,
hundreds of thousands for a 5-hour simulation.  Storing each row in Postgres
with an individual INSERT blocks the SSE→WebSocket relay and bloats the DB.

This module replaces that with:

1.  **In-memory buffer** — ``ProgressWriter.append()`` is O(1), never blocks.
2.  **Periodic flush** — a background asyncio task writes the buffer to a
    local temp NDJSON file every ``flush_interval`` seconds.
3.  **GCS upload on finalize** — when the run ends the raw NDJSON is
    compressed with zstd (~10-20× smaller) and uploaded to a GCS bucket.
    The local temp file is then deleted, keeping the server stateless.
4.  **Fast reads** — ``read_progress()`` downloads from GCS and streams
    decompressed NDJSON back to dicts.  Falls back to local temp (active
    run) then Postgres (legacy data).

GCS blob layout::

    gs://{bucket}/progress/{run_id}.ndjson.zst

Local temp (only during active run)::

    {PROGRESS_DIR}/{run_id}.ndjson
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

_DEFAULT_PROGRESS_DIR = "/tmp/simd_progress"

# ── Lazy imports ───────────────────────────────────────────────────────

_zstd = None


def _get_zstd():
    global _zstd
    if _zstd is None:
        try:
            import zstandard
            _zstd = zstandard
        except ImportError:
            logger.warning("zstandard not installed — progress files will not be compressed")
            _zstd = False  # type: ignore[assignment]
    return _zstd if _zstd is not False else None


_gcs_client = None


def _ensure_gcs_env():
    """Inject GOOGLE_APPLICATION_CREDENTIALS into os.environ from .env."""
    import os
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    from simd_agent.settings import get_settings
    cred_path = get_settings().google_application_credentials
    if cred_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path


def _get_gcs_client():
    """Lazy-init a GCS client (one per process)."""
    global _gcs_client
    if _gcs_client is None:
        _ensure_gcs_env()
        try:
            from google.cloud import storage
            _gcs_client = storage.Client()
        except Exception as exc:
            logger.warning("GCS client init failed — progress stays local: %s", exc)
            _gcs_client = False  # type: ignore[assignment]
    return _gcs_client if _gcs_client is not False else None


def _progress_dir() -> Path:
    from simd_agent.settings import get_settings
    d = get_settings().progress_data_dir
    return Path(d)


def _gcs_bucket_name() -> str | None:
    from simd_agent.settings import get_settings
    return get_settings().progress_gcs_bucket or None


def _gcs_blob_key(run_id: str) -> str:
    return f"progress/{run_id}.ndjson.zst"


# ────────────────────────────────────────────────────────────────────────
# Writer
# ────────────────────────────────────────────────────────────────────────

class ProgressWriter:
    """Non-blocking, append-only progress writer for one run.

    Usage::

        writer = ProgressWriter(run_id)
        writer.start()                    # spawns background flush task

        # In the SSE callback — instant, never blocks:
        writer.append(items)

        # When the run finishes:
        await writer.finalize()           # flush → compress → GCS upload
    """

    def __init__(
        self,
        run_id: UUID | str,
        *,
        flush_interval: float = 2.0,
    ):
        self.run_id = str(run_id)
        self._flush_interval = flush_interval

        self._buf: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._closed = False
        self._total_written = 0

        base = _progress_dir()
        base.mkdir(parents=True, exist_ok=True)
        self._path_ndjson = base / f"{self.run_id}.ndjson"

    # ── public API ────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the background flush loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._flush_loop(),
            name=f"progress-flush-{self.run_id[:8]}",
        )

    def append(self, items: list[dict[str, Any]]) -> None:
        """Add items to the in-memory buffer.  Never blocks, never awaits."""
        if self._closed:
            return
        self._buf.extend(items)

    async def finalize(self) -> None:
        """Final flush → zstd compress → GCS upload → local cleanup."""
        self._closed = True

        # Stop the periodic flush task
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Flush remaining buffer
        await self._flush()

        if self._total_written == 0:
            self._path_ndjson.unlink(missing_ok=True)
            return

        # Compress + upload in a thread (blocking I/O)
        await asyncio.to_thread(self._compress_and_upload)

    async def clear(self) -> None:
        """Delete progress data for this run (called on retry)."""
        async with self._lock:
            self._buf.clear()
            self._total_written = 0
        await asyncio.to_thread(self._remove_all)

    @property
    def count(self) -> int:
        return self._total_written + len(self._buf)

    # ── internals ─────────────────────────────────────────────────

    async def _flush_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(self._flush_interval)
                if self._buf:
                    await self._flush()
        except asyncio.CancelledError:
            pass

    async def _flush(self) -> None:
        async with self._lock:
            if not self._buf:
                return
            batch = self._buf
            self._buf = []

        lines = "".join(
            json.dumps(item, separators=(",", ":")) + "\n" for item in batch
        )
        await asyncio.to_thread(self._append_to_file, lines)
        self._total_written += len(batch)

    def _append_to_file(self, data: str) -> None:
        with open(self._path_ndjson, "a") as f:
            f.write(data)

    def _compress_and_upload(self) -> None:
        """Compress NDJSON with zstd → upload to GCS → delete local file."""
        if not self._path_ndjson.exists():
            return

        raw_size = self._path_ndjson.stat().st_size

        # ── Compress ──────────────────────────────────────────────
        zstd = _get_zstd()
        if zstd is not None:
            cctx = zstd.ZstdCompressor(level=3)
            with open(self._path_ndjson, "rb") as f_in:
                compressed = cctx.compress(f_in.read())
        else:
            # Fallback: store uncompressed
            with open(self._path_ndjson, "rb") as f_in:
                compressed = f_in.read()

        comp_size = len(compressed)
        ratio = raw_size / comp_size if comp_size > 0 else 0
        logger.info(
            "[ProgressWriter] run=%s: %d → %d bytes (%.1f×), %d entries",
            self.run_id[:8], raw_size, comp_size, ratio, self._total_written,
        )

        # ── Upload to GCS ─────────────────────────────────────────
        bucket_name = _gcs_bucket_name()
        client = _get_gcs_client()
        if client and bucket_name:
            try:
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(_gcs_blob_key(self.run_id))
                blob.upload_from_string(
                    compressed,
                    content_type="application/zstd",
                )
                logger.info(
                    "[ProgressWriter] Uploaded run=%s to gs://%s/%s",
                    self.run_id[:8], bucket_name, _gcs_blob_key(self.run_id),
                )
                # GCS is the source of truth — remove local file
                self._path_ndjson.unlink(missing_ok=True)
            except Exception as exc:
                logger.error(
                    "[ProgressWriter] GCS upload failed for run=%s, keeping local: %s",
                    self.run_id[:8], exc,
                )
                # Keep compressed data locally as fallback
                zst_path = self._path_ndjson.with_suffix(".ndjson.zst")
                zst_path.write_bytes(compressed)
                self._path_ndjson.unlink(missing_ok=True)
        else:
            # No GCS configured — keep compressed file locally
            zst_path = self._path_ndjson.with_suffix(".ndjson.zst")
            zst_path.write_bytes(compressed)
            self._path_ndjson.unlink(missing_ok=True)
            logger.info(
                "[ProgressWriter] No GCS bucket — saved locally run=%s",
                self.run_id[:8],
            )

    def _remove_all(self) -> None:
        """Remove local files and GCS blob."""
        self._path_ndjson.unlink(missing_ok=True)
        self._path_ndjson.with_suffix(".ndjson.zst").unlink(missing_ok=True)

        bucket_name = _gcs_bucket_name()
        client = _get_gcs_client()
        if client and bucket_name:
            try:
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(_gcs_blob_key(self.run_id))
                blob.delete()
            except Exception:
                pass  # Not found is fine


# ────────────────────────────────────────────────────────────────────────
# Reader
# ────────────────────────────────────────────────────────────────────────

def _parse_ndjson(data: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def _decompress_and_parse(raw: bytes) -> list[dict[str, Any]]:
    zstd = _get_zstd()
    if zstd is not None:
        dctx = zstd.ZstdDecompressor()
        text = dctx.decompress(raw).decode("utf-8")
    else:
        # Might be uncompressed if zstd wasn't available at write time
        text = raw.decode("utf-8")
    return _parse_ndjson(text)


def _read_from_gcs(run_id: str) -> list[dict[str, Any]] | None:
    """Download + decompress from GCS.  Caches the compressed file locally
    so subsequent reads skip the network round-trip.  Returns None if not found."""
    bucket_name = _gcs_bucket_name()
    client = _get_gcs_client()
    if not client or not bucket_name:
        return None

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(_gcs_blob_key(run_id))
        if not blob.exists():
            return None
        raw = blob.download_as_bytes()

        # Cache compressed file locally — _read_from_local will pick it up next time
        cache_path = _progress_dir() / f"{run_id}.ndjson.zst"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
        logger.info(
            "[ProgressReader] Cached run=%s from GCS (%d bytes)",
            run_id[:8], len(raw),
        )

        return _decompress_and_parse(raw)
    except Exception as exc:
        logger.warning("[ProgressReader] GCS read failed for run=%s: %s", run_id[:8], exc)
        return None


def _read_from_local(run_id: str) -> list[dict[str, Any]] | None:
    """Read from local temp file (active run) or compressed fallback."""
    base = _progress_dir()
    path_raw = base / f"{run_id}.ndjson"
    path_zst = base / f"{run_id}.ndjson.zst"

    if path_raw.exists():
        with open(path_raw, "r") as f:
            return _parse_ndjson(f.read())
    elif path_zst.exists():
        return _decompress_and_parse(path_zst.read_bytes())
    return None


async def read_progress(run_id: UUID | str) -> list[dict[str, Any]] | None:
    """Read progress for a run.

    Priority: local temp (active run) → GCS (finalized) → None (caller
    falls back to Postgres for legacy data).
    """
    rid = str(run_id)

    # 1. Local temp file (run is still active)
    local = await asyncio.to_thread(_read_from_local, rid)
    if local is not None:
        return local

    # 2. GCS (finalized run)
    gcs = await asyncio.to_thread(_read_from_gcs, rid)
    if gcs is not None:
        return gcs

    return None


async def delete_progress_files(run_id: UUID | str) -> None:
    """Remove progress data for a run from all locations."""
    rid = str(run_id)
    base = _progress_dir()
    for suffix in (".ndjson", ".ndjson.zst"):
        p = base / f"{rid}{suffix}"
        p.unlink(missing_ok=True)

    bucket_name = _gcs_bucket_name()
    client = _get_gcs_client()
    if client and bucket_name:
        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(_gcs_blob_key(rid))
            blob.delete()
        except Exception:
            pass
