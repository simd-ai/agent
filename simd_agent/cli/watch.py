"""``simd watch RUN_ID`` — re-attach to an in-progress run."""

from __future__ import annotations

import argparse
import json
import sys

import httpx
import websockets

from simd_agent.cli.client import AgentClient
from simd_agent.cli.config import CliConfig
from simd_agent.cli.display import console, err
from simd_agent.cli.display.stages import render_event, render_verbose
from simd_agent.cli.display.summary import render_summary


async def run(args: argparse.Namespace, config: CliConfig) -> int:
    client = AgentClient(config)

    # Sanity check: does the run exist?
    try:
        status = await client.get_run_status(args.run_id)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            err(f"run {args.run_id} not found")
            return 1
        err(f"status check failed (HTTP {e.response.status_code}): {e.response.text}")
        return 1

    last_seq = status.get("last_seq") or 0

    if getattr(args, "verbose", False):
        console.print(f"  reconnecting to {args.run_id} (last_seq={last_seq})")

    try:
        async for event in client.watch_run(args.run_id, last_seq=last_seq):
            if getattr(args, "json", False):
                print(json.dumps(event), flush=True)
                continue
            if getattr(args, "verbose", False):
                console.print(render_verbose(event))
                continue
            line = render_event(event)
            if line is not None:
                console.print(line)
    except websockets.ConnectionClosedError as e:
        err(f"connection closed: {e}")
        return 1

    # On clean termination, fetch and print the summary.
    try:
        summary = await client.get_run_summary(args.run_id)
        render_summary(summary)
        return 0 if summary.get("status") == "succeeded" else 1
    except httpx.HTTPStatusError:
        return 0
