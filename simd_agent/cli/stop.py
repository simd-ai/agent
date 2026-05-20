"""``simd stop RUN_ID`` — gracefully stop a running simulation."""

from __future__ import annotations

import argparse

import httpx

from simd_agent.cli.client import AgentClient
from simd_agent.cli.config import CliConfig
from simd_agent.cli.display import console, err


async def run(args: argparse.Namespace, config: CliConfig) -> int:
    client = AgentClient(config)
    try:
        result = await client.stop_run(args.run_id)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            err(f"run {args.run_id} not found")
            return 1
        err(f"stop request failed (HTTP {e.response.status_code}): {e.response.text}")
        return 1

    if result.get("stopped"):
        console.print(f"  ✓ stopped {args.run_id}")
        return 0
    reason = result.get("reason") or "unknown"
    console.print(f"  ■ {args.run_id}: {reason}")
    return 0
