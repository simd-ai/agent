"""Local filesystem storage backend.

Stores objects as plain files under a configurable root directory.
Suitable for development, self-hosting, and Docker Compose deployments.

Layout example::

    {root}/meshes/{simulation_id}/original.msh
    {root}/meshes/{simulation_id}/surface.vtp
    {root}/results/{run_id}/surface.vtp
    {root}/results/{run_id}/timesteps/t_0_1.vtp
    {root}/cases/{run_id}/case.zip
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .base import StorageBackend

logger = logging.getLogger(__name__)


class LocalBackend(StorageBackend):
    """Store objects on the local filesystem."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("[STORAGE/LOCAL] root=%s", self._root)

    def _path(self, key: str) -> Path:
        # Prevent path traversal
        resolved = (self._root / key).resolve()
        if not str(resolved).startswith(str(self._root)):
            raise ValueError(f"Invalid storage key: {key}")
        return resolved

    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)
        logger.debug("[STORAGE/LOCAL] Uploaded %s (%d bytes)", key, len(data))

    async def download(self, key: str) -> bytes | None:
        path = self._path(key)
        if not path.exists():
            return None
        return await asyncio.to_thread(path.read_bytes)

    async def exists(self, key: str) -> bool:
        return self._path(key).exists()

    async def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()

    async def list_keys(self, prefix: str) -> list[str]:
        prefix_path = self._path(prefix)
        # If prefix points to a dir, list everything under it
        # If it points to a partial filename, list siblings that match
        if prefix_path.is_dir():
            search_dir = prefix_path
            pattern = "**/*"
        else:
            search_dir = prefix_path.parent
            pattern = f"{prefix_path.name}*"
            if not search_dir.exists():
                return []
            # Also search recursively under matching dirs
            results: list[str] = []
            for p in search_dir.iterdir():
                rel = str(p.relative_to(self._root))
                if not rel.startswith(prefix):
                    continue
                if p.is_file():
                    results.append(rel)
                elif p.is_dir():
                    for child in p.rglob("*"):
                        if child.is_file():
                            results.append(str(child.relative_to(self._root)))
            return sorted(results)

        if not search_dir.exists():
            return []

        return sorted(
            str(p.relative_to(self._root))
            for p in search_dir.rglob("*")
            if p.is_file()
        )
