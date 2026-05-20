"""``simd status`` — show backend, runner, and config at a glance.

Works in every backend mode.  For local-docker it also reports
container health via ``docker compose ps``.
"""

from __future__ import annotations

import argparse

import httpx

from simd_agent.cli.client import AgentClient
from simd_agent.cli.config import CliConfig
from simd_agent.cli.display import console
from simd_agent.cli.process import compose_file, compose_ps, env_file


async def run(args: argparse.Namespace, config: CliConfig) -> int:
    console.print("[bold]simd status[/]")

    agent_mode  = config.extras.get("agent_mode",  "?")
    runner_mode = config.extras.get("runner_mode", "?")
    runner_url  = config.extras.get("runner_url",  "?")

    console.print(f"  agent_mode    {agent_mode}")
    console.print(f"  agent_url     {config.agent_url}    {await _health_dot(config.agent_url)}")
    console.print(f"  runner_mode   {runner_mode}")
    console.print(f"  runner_url    {runner_url}    {await _health_dot(runner_url)}")

    # Bundled mode: also show per-service container state.
    if agent_mode == "local-docker" and compose_file().is_file():
        rows = await compose_ps()
        if rows:
            console.print("\n  containers:")
            for r in rows:
                name  = r.get("Service") or r.get("Name") or "?"
                state = r.get("State") or "?"
                health = r.get("Health") or ""
                tag = "[green]✓[/]" if state == "running" else "[red]✗[/]"
                extra = f" ({health})" if health and health != "healthy" else ""
                console.print(f"    {tag} {name}  {state}{extra}")

    # Config locations.
    console.print(f"\n  env file      {env_file()}")
    console.print(f"  config file   ~/.config/simd/config.toml")

    if config.user_id:
        console.print(f"  user_id       {config.user_id}")
    if config.simulation_id:
        console.print(f"  simulation_id {config.simulation_id}")
    if config.last_run_id:
        console.print(f"  last_run_id   {config.last_run_id}")

    return 0


async def _health_dot(url: str) -> str:
    """Return a single-character colored marker for a /health probe."""
    if not url or url == "?":
        return "[dim]—[/]"
    timeout = httpx.Timeout(2.0, connect=1.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{url.rstrip('/')}/health")
            if r.status_code == 200:
                return "[green]✓ healthy[/]"
            return f"[yellow]HTTP {r.status_code}[/]"
    except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
        return "[red]✗ unreachable[/]"
