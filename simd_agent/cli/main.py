"""Top-level CLI dispatch.

``simd`` is a small dispatcher that hands off to per-subcommand
modules.  The subcommands are kept in separate files because the
interactive ones (``run``, ``watch``) carry a lot of display logic
that would crowd this file.

Subcommands shipped in v0.1:

    simd init                  # interactive wizard — backend, runner,
                               # LLM provider, storage
    simd up                    # start the bundled docker stack
    simd down                  # stop the bundled docker stack
    simd status                # show backend / runner health + config
    simd run PROMPT MESH       # end-to-end interactive run
                               # (auto-runs `init` + `up` if needed)
    simd watch RUN_ID          # re-attach to a running run
    simd ls                    # list recent runs
    simd stop RUN_ID           # gracefully stop a running run

No login command — the CLI talks to whatever local agent the user
has running, and creates the local-postgres records it needs on
first ``simd run``.  No account, no tracking, no telemetry.

``--help`` on any subcommand prints its specific flags.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Callable, Awaitable

from simd_agent.cli.config import CliConfig


# Type alias for subcommand handler: takes parsed args, returns exit code.
_HandlerSync = Callable[[argparse.Namespace, CliConfig], int]
_HandlerAsync = Callable[[argparse.Namespace, CliConfig], Awaitable[int]]


def _build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser with every subcommand."""
    parser = argparse.ArgumentParser(
        prog="simd",
        description="simd-agent — AI-native CFD simulation agent.",
    )
    parser.add_argument(
        "--agent",
        metavar="URL",
        help="Override agent URL (default: $SIMD_AGENT or http://localhost:8000).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="simd 0.1.0",
    )

    sub = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        required=True,
    )

    # ── simd init ──────────────────────────────────────────────
    p_init = sub.add_parser(
        "init",
        help="Interactive wizard — pick backend, runner, LLM, storage.",
        description=(
            "Walk through the questions needed to configure the CLI "
            "and write the resulting ~/.simd/.env + ~/.config/simd/config.toml."
        ),
    )

    # ── simd up ────────────────────────────────────────────────
    p_up = sub.add_parser(
        "up",
        help="Start the bundled docker stack (local-docker mode only).",
    )

    # ── simd down ──────────────────────────────────────────────
    p_down = sub.add_parser(
        "down",
        help="Stop the bundled docker stack (local-docker mode only).",
    )

    # ── simd status ────────────────────────────────────────────
    p_status = sub.add_parser(
        "status",
        help="Show backend / runner health + config.",
    )

    # ── simd run ───────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help="Upload mesh, run precheck, review patches, start the run.",
        description=(
            "End-to-end interactive run.  Uploads the mesh, runs the "
            "precheck, lets you review and edit the proposed boundary "
            "conditions, then kicks off codegen + simulation and "
            "streams the result back."
        ),
    )
    p_run.add_argument("prompt_file", help="Path to a text file with the natural-language prompt.")
    p_run.add_argument("mesh_file",   help="Path to the mesh file (.msh).")
    p_run.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip interactive patch review, accept precheck's defaults.",
    )
    p_run.add_argument(
        "--max-retries", type=int, default=7, metavar="N",
        help="Self-healing retry budget (default: 7).",
    )
    p_run.add_argument(
        "--solver", metavar="NAME",
        help="Force a solver name, skip LLM selection (e.g. --solver pimpleFoam).",
    )
    p_run.add_argument(
        "--detach", action="store_true",
        help="Return immediately after launch (don't stream).",
    )
    p_run.add_argument(
        "--no-run", action="store_true",
        help="Stop after precheck + patch review (dry run).",
    )

    # ── simd watch ─────────────────────────────────────────────
    p_watch = sub.add_parser(
        "watch",
        help="Re-attach to an in-progress run.",
    )
    p_watch.add_argument("run_id", help="The run UUID.")

    # ── simd ls ────────────────────────────────────────────────
    p_ls = sub.add_parser(
        "ls",
        help="List recent runs.",
    )
    p_ls.add_argument(
        "--simulation", metavar="UUID",
        help="Scope to a specific simulation (default: the one in config).",
    )
    p_ls.add_argument(
        "--limit", type=int, default=10, metavar="N",
        help="Maximum runs to show (default: 10).",
    )

    # ── simd stop ──────────────────────────────────────────────
    p_stop = sub.add_parser(
        "stop",
        help="Gracefully stop a running simulation.",
    )
    p_stop.add_argument("run_id", help="The run UUID.")

    # ── global verbosity ───────────────────────────────────────
    # These apply to every subcommand; declared on the top-level
    # parser so they work before OR after the subcommand name.
    for p in (parser, p_run, p_watch, p_ls, p_stop, p_init, p_up, p_down, p_status):
        verbosity = p.add_mutually_exclusive_group()
        verbosity.add_argument(
            "--quiet", action="store_true",
            help="Only print errors and the final result.",
        )
        verbosity.add_argument(
            "--verbose", action="store_true",
            help="Stream every AgentEvent — for debugging the agent.",
        )
        p.add_argument(
            "--json", action="store_true",
            help="Machine-readable output, one JSON event per line.",
        )

    return parser


def _dispatch(args: argparse.Namespace, config: CliConfig) -> int:
    """Route the parsed args to the right subcommand handler.

    Imports are lazy so a failing dependency in one subcommand doesn't
    break ``simd --help``.
    """
    cmd = args.command
    if cmd == "init":
        from simd_agent.cli.init import run as run_init
        return asyncio.run(run_init(args, config))
    if cmd == "up":
        from simd_agent.cli.up import run as run_up
        return asyncio.run(run_up(args, config))
    if cmd == "down":
        from simd_agent.cli.down import run as run_down
        return asyncio.run(run_down(args, config))
    if cmd == "status":
        from simd_agent.cli.status import run as run_status
        return asyncio.run(run_status(args, config))
    if cmd == "run":
        from simd_agent.cli.run import run as run_run
        return asyncio.run(run_run(args, config))
    if cmd == "watch":
        from simd_agent.cli.watch import run as run_watch
        return asyncio.run(run_watch(args, config))
    if cmd == "ls":
        from simd_agent.cli.ls import run as run_ls
        return asyncio.run(run_ls(args, config))
    if cmd == "stop":
        from simd_agent.cli.stop import run as run_stop
        return asyncio.run(run_stop(args, config))

    # Should be unreachable because the parser requires a subcommand.
    print(f"simd: unknown command {cmd!r}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    """Entry point — the ``simd`` console script lands here."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = CliConfig.load()
    if getattr(args, "agent", None):
        config.agent_url = args.agent

    return _dispatch(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
