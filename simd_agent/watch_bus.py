# simd_agent/watch_bus.py
"""In-memory pub/sub for /ws/watch/{run_id} live event delivery.

The EventBus publishes every emitted event here (after DB persistence).
/ws/watch/{run_id} subscribers receive live events in real time.
"""

import asyncio
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class WatchBus:
    """Module-level singleton that distributes live events to watch subscribers."""

    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def subscribe(self, run_id: str) -> "asyncio.Queue[dict]":
        """Register a new watcher queue for run_id. Returns the queue."""
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=2000)
        async with self._lock:
            self._queues[run_id].append(q)
        return q

    async def unsubscribe(self, run_id: str, q: "asyncio.Queue[dict]") -> None:
        """Remove a watcher queue. Safe to call even if already removed."""
        async with self._lock:
            try:
                self._queues[run_id].remove(q)
            except ValueError:
                pass
            if not self._queues.get(run_id):
                self._queues.pop(run_id, None)

    def publish_nowait(self, run_id: str, message: dict) -> None:
        """Non-blocking publish — drops silently if a watcher queue is full."""
        queues = self._queues.get(run_id)
        if not queues:
            return
        for q in list(queues):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning("[WatchBus] Queue full for run %s — dropping event", run_id)

    def has_watchers(self, run_id: str) -> bool:
        return bool(self._queues.get(run_id))


# ── Module-level singleton ────────────────────────────────────────────────────

_bus: WatchBus | None = None


def get_watch_bus() -> WatchBus:
    """Return the process-wide WatchBus instance (created on first call)."""
    global _bus
    if _bus is None:
        _bus = WatchBus()
    return _bus
