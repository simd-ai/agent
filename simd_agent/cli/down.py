"""``simd down`` — stop the bundled docker stack.

Mirror of ``simd up`` — only operates when ``agent_mode`` is
``local-docker``.  Preserves volumes by default so postgres state
and case ZIPs survive across restarts.
"""

from __future__ import annotations

import argparse

from simd_agent.cli.config import CliConfig
from simd_agent.cli.display import console, err
from simd_agent.cli.process import compose_down, compose_file


async def run(args: argparse.Namespace, config: CliConfig) -> int:
    mode = config.extras.get("agent_mode", "local-docker")
    if mode != "local-docker":
        err(
            f"  agent_mode is {mode!r} — `simd down` only manages the "
            f"bundled docker stack."
        )
        return 1

    if not compose_file().is_file():
        err("  no compose file — nothing to stop.")
        return 1

    console.print("  stopping bundled stack …")
    rc = await compose_down()
    if rc == 0:
        console.print("  [bold green]✓[/] stopped.")
    return rc
