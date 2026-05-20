"""``simd ls`` — list recent runs in the active simulation."""

from __future__ import annotations

import argparse

import httpx
from rich.table import Table

from simd_agent.cli.client import AgentClient
from simd_agent.cli.config import CliConfig
from simd_agent.cli.display import console, err


_STATUS_COLOR = {
    "succeeded": "green",
    "failed":    "red",
    "stopped":   "yellow",
    "cancelled": "white",
    "running":   "cyan",
    "pending":   "white",
}


async def run(args: argparse.Namespace, config: CliConfig) -> int:
    simulation_id = args.simulation or config.simulation_id
    if not simulation_id:
        err(
            "no simulation_id known — pass --simulation <UUID> or "
            "run `simd run …` once to create one."
        )
        return 1

    client = AgentClient(config)
    try:
        rows = await client.list_runs(simulation_id)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            err(f"simulation {simulation_id} not found")
            return 1
        err(f"list_runs failed (HTTP {e.response.status_code}): {e.response.text}")
        return 1

    if not rows:
        console.print("  (no runs)")
        return 0

    rows = rows[: args.limit]

    t = Table(show_header=True, header_style="bold")
    t.add_column("run_id", style="cyan", overflow="fold")
    t.add_column("status")
    t.add_column("solver", style="bold")
    t.add_column("op")
    t.add_column("started")

    for r in rows:
        status = r.get("status") or "?"
        color = _STATUS_COLOR.get(status, "white")
        t.add_row(
            r.get("id") or "?",
            f"[{color}]{status}[/]",
            r.get("solver") or "—",
            r.get("op") or "—",
            r.get("started_at") or "—",
        )
    console.print(t)
    return 0
