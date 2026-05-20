"""Thin HTTP + WebSocket client around the simd-agent API.

The CLI's only contact point with the agent.  Every subcommand goes
through this module — keeps the wire format in one place, makes the
subcommand modules testable with stubs.

Designed for one short-lived invocation per CLI call.  Re-use of the
underlying ``httpx.AsyncClient`` would be nice for ``simd ls`` / ``simd
watch`` but the marginal cost of a fresh connection is small compared
to the LLM calls we're waiting on, and statelessness is easier to
reason about.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import websockets

from simd_agent.cli.config import CliConfig


class AgentClient:
    """Wraps the agent's REST + WebSocket surfaces.

    All methods are async.  All return decoded JSON (or raise an
    ``httpx.HTTPStatusError`` / ``websockets`` exception on failure).
    """

    def __init__(self, config: CliConfig, *, timeout_seconds: float = 60.0) -> None:
        self._config = config
        self._timeout = httpx.Timeout(timeout_seconds, connect=10.0)

    # ── HTTP helpers ───────────────────────────────────────────

    async def _get(self, path: str, **params: Any) -> Any:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self._config.agent_url}{path}",
                params={k: v for k, v in params.items() if v is not None},
                headers=self._config.auth_header(),
            )
            resp.raise_for_status()
            return resp.json()

    async def _post(self, path: str, json_body: Any = None, **kwargs: Any) -> Any:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._config.agent_url}{path}",
                json=json_body,
                headers=self._config.auth_header(),
                **kwargs,
            )
            resp.raise_for_status()
            # 204 No Content / empty body — return None rather than raise.
            return resp.json() if resp.content else None

    # ── Health ─────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """``GET /health`` — quick reachability check used by ``simd ls``."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get(f"{self._config.agent_url}/health")
            resp.raise_for_status()
            return resp.json()

    # ── Users / simulations / meshes ───────────────────────────

    async def get_or_create_user(self, email: str) -> dict[str, Any]:
        """``POST /api/users`` — upsert a user, get back a user_id."""
        return await self._post("/api/users", {"email": email})

    async def create_simulation(
        self, user_id: str, name: str = "simd-cli scratch"
    ) -> dict[str, Any]:
        """``POST /api/simulations``."""
        return await self._post(
            "/api/simulations",
            {"user_id": user_id, "name": name},
        )

    async def list_runs(self, simulation_id: str | None = None) -> list[dict[str, Any]]:
        """``GET /api/runs?simulation_id=...``."""
        params = {"simulation_id": simulation_id} if simulation_id else {}
        return await self._get("/api/runs", **params)

    async def upload_mesh(
        self, simulation_id: str, mesh_path: Path,
    ) -> dict[str, Any]:
        """``POST /api/mesh/convert`` — uploads the .msh, returns mesh info."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            with mesh_path.open("rb") as f:
                files = {"file": (mesh_path.name, f, "application/octet-stream")}
                data = {"simulation_id": simulation_id}
                resp = await client.post(
                    f"{self._config.agent_url}/api/mesh/convert",
                    files=files,
                    data=data,
                    headers=self._config.auth_header(),
                )
                resp.raise_for_status()
                return resp.json()

    # ── Precheck ───────────────────────────────────────────────

    async def precheck(
        self,
        prompt: str,
        mesh_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``POST /api/precheck`` — doesn't persist a run record."""
        body: dict[str, Any] = {
            "prompt":   prompt,
            "has_mesh": mesh_info is not None,
        }
        if mesh_info is not None:
            body["mesh_info"] = mesh_info
        return await self._post("/api/precheck", body)

    # ── Runs ───────────────────────────────────────────────────

    async def get_run_status(self, run_id: str) -> dict[str, Any]:
        return await self._get(f"/api/runs/{run_id}/status")

    async def get_run_summary(self, run_id: str) -> dict[str, Any]:
        return await self._get(f"/api/runs/{run_id}/summary")

    async def stop_run(self, run_id: str) -> dict[str, Any]:
        return await self._post(f"/api/runs/{run_id}/stop")

    # ── WebSocket: start a run ─────────────────────────────────

    async def stream_run(
        self,
        start_request: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Open ``/ws/run``, send the StartRequest, yield each event.

        Closes the socket on the first ``final`` event or on any
        ``run_*`` terminal status.  Callers iterate until the generator
        finishes — no need to break manually.
        """
        url = f"{self._config.ws_url}/ws/run"
        async with websockets.connect(
            url, max_size=2**25,  # 32 MB, large enough for generated files
        ) as ws:
            await ws.send(json.dumps(start_request))
            async for raw in ws:
                event = json.loads(raw)
                yield event
                if event.get("type") == "final":
                    return

    # ── WebSocket: re-attach to a running run ──────────────────

    async def watch_run(
        self,
        run_id: str,
        last_seq: int = 0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Open ``/ws/watch/{run_id}?last_seq=N``, replay + tail."""
        url = (
            f"{self._config.ws_url}/ws/watch/{run_id}"
            f"?last_seq={last_seq}"
        )
        async with websockets.connect(url, max_size=2**25) as ws:
            async for raw in ws:
                event = json.loads(raw)
                yield event
                if event.get("type") == "final":
                    return
