"""``simd up`` — start the bundled docker stack.

Only valid when ``agent_mode == "local-docker"``.  Other modes refuse
because the CLI doesn't own those processes (the user runs uvicorn
themselves, or the agent is on a remote machine).
"""

from __future__ import annotations

import argparse

from simd_agent.cli.config import CliConfig
from simd_agent.cli.display import console, err
from simd_agent.cli.process import (
    DockerMissing,
    compose_file,
    compose_up,
    docker_daemon_alive,
    env_file,
    wait_for_health,
)


async def run(args: argparse.Namespace, config: CliConfig) -> int:
    mode = config.extras.get("agent_mode", "local-docker")

    if mode != "local-docker":
        err(
            f"  agent_mode is {mode!r} — `simd up` only manages the "
            f"bundled docker stack."
        )
        if mode == "local-bare-metal":
            err("  start the agent with `uvicorn simd_agent.main:app --port 8000`.")
        else:
            err(f"  the remote agent at {config.agent_url} isn't ours to start.")
        return 1

    if not env_file().is_file() or not compose_file().is_file():
        err("  no setup found — run `simd init` first.")
        return 1

    if not await docker_daemon_alive():
        err(
            "  docker daemon isn't running.  start Docker Desktop (or "
            "your daemon) and retry."
        )
        return 1

    console.print("  starting bundled stack …")
    rc = await compose_up(detach=True)
    if rc != 0:
        err(f"  `docker compose up` failed (exit {rc}).")
        return rc

    console.print("  waiting for agent to come up …")
    if not await wait_for_health(config.agent_url):
        err(
            f"  agent at {config.agent_url} didn't respond within 60s.  "
            f"check `docker compose logs agent`."
        )
        return 1

    console.print(f"  [bold green]✓[/] agent up at {config.agent_url}")
    return 0
