"""Process management for the ``local-docker`` backend mode.

Thin async wrapper around the ``docker compose`` CLI.  Lives behind
the subcommand modules (``init``, ``up``, ``down``, ``status``, plus
the auto-start path in ``run``) so they don't all reinvent
subprocess plumbing.

This module owns:

  - the location of the compose file (``~/.simd/docker-compose.yml``)
  - the docker-binary lookup (with a helpful error when docker is
    missing or the daemon isn't running)
  - the ``docker compose up -d`` / ``down`` / ``ps`` calls
  - polling ``GET /health`` on the agent until it responds (or
    timing out)

Nothing here knows about the agent's API — that's ``client.py``'s
job.  Nothing here knows about user-facing display — that's
``display/``.  Keeps the CLI testable by swapping this module out.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx


# ── Paths ────────────────────────────────────────────────────


def simd_home() -> Path:
    """User-scoped data directory for the bundled stack.

    Carries the compose file, the merged ``.env``, and the local
    storage volume root.  Honours ``$SIMD_HOME`` for testing.
    """
    import os
    if (env := os.environ.get("SIMD_HOME")):
        return Path(env)
    return Path.home() / ".simd"


def compose_file() -> Path:
    """Path the CLI tells ``docker compose`` to use."""
    return simd_home() / "docker-compose.yml"


def env_file() -> Path:
    """Path to the merged ``.env`` for the bundled stack."""
    return simd_home() / ".env"


# ── docker binary discovery ──────────────────────────────────


class DockerMissing(Exception):
    """Raised when the docker CLI isn't on $PATH or the daemon is down."""


def _resolve_docker() -> str:
    """Return the path to the ``docker`` binary, or raise DockerMissing."""
    path = shutil.which("docker")
    if not path:
        raise DockerMissing(
            "the `docker` command isn't on your $PATH.  install Docker "
            "Desktop (https://www.docker.com/get-started) or use the "
            "bare-metal backend mode instead."
        )
    return path


async def docker_daemon_alive() -> bool:
    """``docker info`` — quick check that the daemon is actually running."""
    try:
        docker = _resolve_docker()
    except DockerMissing:
        return False
    proc = await asyncio.create_subprocess_exec(
        docker, "info",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return (await proc.wait()) == 0


# ── compose commands ────────────────────────────────────────


async def _compose(*args: str, capture: bool = False) -> tuple[int, str, str]:
    """Run ``docker compose -f <file> <args>``.

    Returns ``(exit_code, stdout, stderr)``.  When ``capture`` is
    False, stdout/stderr stream to the user's terminal (best for
    ``up``/``down`` where the progress is useful).  When True they
    get captured for the caller (best for ``ps`` / ``logs``).
    """
    docker = _resolve_docker()
    cf = compose_file()
    if not cf.is_file():
        raise FileNotFoundError(
            f"compose file not found at {cf}.  run `simd init` first."
        )

    cmd = [docker, "compose", "-f", str(cf), *args]
    if capture:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode or 0, stdout.decode(), stderr.decode()
    else:
        proc = await asyncio.create_subprocess_exec(*cmd)
        return (await proc.wait()) or 0, "", ""


async def compose_up(detach: bool = True) -> int:
    """Start the bundled stack.  Returns the process exit code."""
    args = ["up"]
    if detach:
        args.append("-d")
    rc, _, _ = await _compose(*args)
    return rc


async def compose_down() -> int:
    """Stop the bundled stack (preserve volumes by default)."""
    rc, _, _ = await _compose("down")
    return rc


async def compose_ps() -> list[dict[str, Any]]:
    """Return one dict per service: ``{name, state, ports}``."""
    rc, stdout, _ = await _compose("ps", "--format", "json", capture=True)
    if rc != 0:
        return []
    import json
    rows: list[dict[str, Any]] = []
    # Modern docker compose emits one JSON object per line; older
    # versions emit a single JSON array.  Handle both.
    text = stdout.strip()
    if not text:
        return rows
    if text.startswith("["):
        try:
            rows = json.loads(text)
        except json.JSONDecodeError:
            return []
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ── /health polling ──────────────────────────────────────────


async def wait_for_health(
    url: str,
    *,
    timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 1.0,
) -> bool:
    """Poll ``GET <url>/health`` until 200 OK or timeout.

    Used after ``compose up`` to delay returning to the caller until
    the agent is actually accepting requests.  Returns ``True`` on
    health, ``False`` on timeout — caller decides how to surface.
    """
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    timeout = httpx.Timeout(2.0, connect=1.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await client.get(f"{url.rstrip('/')}/health")
                if r.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                pass
            await asyncio.sleep(poll_interval_seconds)
    return False


# ── port-conflict detection ──────────────────────────────────


def port_in_use(host: str, port: int) -> bool:
    """Check if a TCP port is already bound on the local machine.

    Used at ``simd init`` time to warn the user before we even try
    ``docker compose up`` and get an unhelpful EADDRINUSE.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.connect((host, port))
            return True
        except (OSError, ConnectionRefusedError):
            return False
