"""``simd run PROMPT MESH`` — end-to-end interactive run.

The CLI's flagship command.  Walks the user through:

  1. uploading the mesh        (POST /api/mesh/convert)
  2. running precheck          (POST /api/precheck)
  3. reviewing patches         (interactive REPL — see display/patches.py)
  4. starting the run          (open /ws/run, send StartRequest)
  5. streaming progress        (5-stage display from display/stages.py)
  6. printing the summary      (display/summary.py)

The flow is sequential and stateful — each step depends on the
previous one's output.  This module is the orchestrator; the moving
parts live in ``client.py`` and ``display/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from simd_agent.cli.client import AgentClient
from simd_agent.cli.config import CliConfig
from simd_agent.cli.display import console, err
from simd_agent.cli.display.patches import review
from simd_agent.cli.display.stages import (
    render_event,
    render_verbose,
    stage_banner,
)
from simd_agent.cli.display.summary import render_summary


# ── Public entry point ───────────────────────────────────────


async def run(args: argparse.Namespace, config: CliConfig) -> int:
    """Top-level run-subcommand handler.

    Returns 0 on success, 1 on user-cancel or backend failure, 2 on
    argument errors.
    """
    prompt_path = Path(args.prompt_file)
    mesh_path = Path(args.mesh_file)

    if not prompt_path.is_file():
        err(f"prompt file not found: {prompt_path}")
        return 2
    if not mesh_path.is_file():
        err(f"mesh file not found: {mesh_path}")
        return 2

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        err(f"prompt file is empty: {prompt_path}")
        return 2

    # ── Bootstrap: inline `simd init` if no setup, then ensure
    # the backend is actually up.  Returning 1 here is correct
    # because the user hasn't even gotten to stage 1 yet.
    if not await _ensure_setup(config):
        return 1
    if not await _ensure_backend_up(config):
        return 1

    # Re-read the config — `simd init` may have mutated user_id /
    # agent_url since the top-level main.py loaded it.
    config = CliConfig.load()

    client = AgentClient(config)

    # ── Bootstrap: confirm we have a user_id, create simulation ──
    try:
        simulation_id = await _ensure_simulation(client, config)
    except _CliError as e:
        err(str(e))
        return 1

    # ── Stage 1: upload mesh ───────────────────────────────────
    stage_banner(1, 5, "uploading mesh")
    try:
        mesh_info = await client.upload_mesh(simulation_id, mesh_path)
    except httpx.HTTPStatusError as e:
        err(f"mesh upload failed (HTTP {e.response.status_code}): {e.response.text}")
        return 1
    size_mb = mesh_path.stat().st_size / 1024 / 1024
    console.print(f"  {mesh_path.name} ({size_mb:.1f} MB) … done")
    console.print(f"  mesh_id   {mesh_info.get('mesh_id', simulation_id)}")
    console.print(f"  patches   {_count_patches(mesh_info)}")

    # ── Stage 2: precheck ──────────────────────────────────────
    stage_banner(2, 5, "running precheck")
    try:
        precheck = await client.precheck(prompt=prompt, mesh_info=mesh_info)
    except httpx.HTTPStatusError as e:
        err(f"precheck failed (HTTP {e.response.status_code}): {e.response.text}")
        return 1

    suggested = precheck.get("suggestedConfig") or {}
    solver = (suggested.get("solver") or {}).get("openfoamSolver") or "(unknown)"
    regime = suggested.get("flowRegime") or "?"
    console.print(f"  solver    {solver}")
    console.print(f"  regime    {regime}")

    patches = _patches_from_precheck(precheck, mesh_info)
    if not patches:
        console.print(
            "  [yellow](no patches found — proceeding without "
            "interactive review)[/]"
        )
    else:
        console.print(f"  {len(patches)} patches found:")
        overrides = review(patches, auto_accept=args.yes)
        if overrides is None:
            console.print("  cancelled.")
            return 1
        if overrides:
            console.print(f"  ✓ {len(overrides)} patch override(s) applied")
        # Merge overrides back into the suggested config so the run
        # uses the user's edited values.
        _apply_overrides(suggested, overrides)

    if args.no_run:
        console.print("\n  --no-run set; stopping after precheck.")
        return 0

    # ── Build StartRequest ─────────────────────────────────────
    start_request = _build_start_request(
        prompt=prompt,
        mesh_info=mesh_info,
        suggested_config=suggested,
        simulation_id=simulation_id,
        user_id=config.user_id,
        max_retries=args.max_retries,
        forced_solver=args.solver,
    )

    # ── Stage 3 + 4: stream events ─────────────────────────────
    stage_banner(3, 5, "generating case files")
    run_id: str | None = None
    final_payload: dict[str, Any] | None = None
    sim_stage_started = False

    try:
        async for event in client.stream_run(start_request):
            # Catch the run_id as it streams by — first event carries it.
            if not run_id and event.get("run_id"):
                run_id = event["run_id"]

            # Switch from stage 3 to stage 4 when sim-server events start.
            etype = event.get("type", "")
            if not sim_stage_started and etype in (
                "sim_submitted", "sim_extract_started",
                "mesh_conversion_started",
            ):
                sim_stage_started = True
                stage_banner(4, 5, "running on simulation server")

            if etype == "final":
                final_payload = event.get("payload")
                break

            if args.json:
                print(json.dumps(event), flush=True)
                continue
            if args.verbose:
                console.print(render_verbose(event))
                continue
            if args.quiet:
                continue

            line = render_event(event)
            if line is not None:
                console.print(line)

    except Exception as e:
        err(f"stream interrupted: {type(e).__name__}: {e}")
        if run_id:
            # Remember the run_id so the user can `simd watch` later.
            config.last_run_id = run_id
            config.save()
        return 1

    if run_id:
        config.last_run_id = run_id
        config.simulation_id = simulation_id
        config.save()

    # ── Stage 5: summary ───────────────────────────────────────
    stage_banner(5, 5, "results")
    if run_id is None:
        err("did not receive a run_id from the agent")
        return 1
    try:
        summary = await client.get_run_summary(run_id)
        render_summary(summary)
        status = summary.get("status")
    except httpx.HTTPStatusError:
        status = (final_payload or {}).get("status")

    return 0 if status == "succeeded" else 1


# ── Helpers ──────────────────────────────────────────────────


class _CliError(Exception):
    """Sentinel for user-facing CLI errors that should stop the run."""


async def _ensure_setup(config: CliConfig) -> bool:
    """Inline the init wizard if no setup file exists yet.

    Returns False when the wizard was cancelled or failed — the
    caller should propagate that as the run's exit code.
    """
    from simd_agent.cli.init import maybe_init
    return await maybe_init(config)


async def _ensure_backend_up(config: CliConfig) -> bool:
    """If the agent is unreachable AND the backend is local-docker,
    start the stack silently and wait for /health.

    For local-bare-metal or remote modes we refuse with a clear
    explanation — those backends aren't ours to start.
    """
    from simd_agent.cli.process import (
        compose_file, compose_up, docker_daemon_alive,
        wait_for_health,
    )

    # Quick probe first — if it's already up we have nothing to do.
    if await wait_for_health(config.agent_url, timeout_seconds=2.0):
        return True

    mode = config.extras.get("agent_mode", "local-docker")

    if mode == "local-bare-metal":
        err(
            f"  agent at {config.agent_url} isn't responding.\n"
            f"  start it with `uvicorn simd_agent.main:app --port 8000` "
            f"and retry."
        )
        return False
    if mode == "remote":
        err(
            f"  remote agent at {config.agent_url} isn't reachable.\n"
            f"  check the URL or your network."
        )
        return False

    # local-docker: try to start it.
    if not compose_file().is_file():
        err(
            "  bundled stack is configured but the compose file is "
            "missing.  re-run `simd init`."
        )
        return False
    if not await docker_daemon_alive():
        err(
            "  docker daemon isn't running.  start Docker Desktop "
            "(or your daemon) and retry."
        )
        return False

    console.print("  agent isn't running.  starting bundled stack …")
    rc = await compose_up(detach=True)
    if rc != 0:
        err(f"  `docker compose up` failed (exit {rc}).")
        return False
    if not await wait_for_health(config.agent_url, timeout_seconds=60.0):
        err(
            f"  agent at {config.agent_url} didn't come up within 60s.  "
            f"check `docker compose logs agent`."
        )
        return False
    console.print(f"  [bold green]✓[/] agent up at {config.agent_url}\n")
    return True


async def _ensure_simulation(client: AgentClient, config: CliConfig) -> str:
    """Get a simulation_id, creating one if we don't have one cached.

    The agent's data model attaches simulations to a "user" record for
    relational integrity in its local Postgres.  In open mode that
    record is purely internal bookkeeping — no account, no tracking,
    no data leaves the machine.  We create it once per CLI install
    with a fixed local identity and cache the resulting id in
    ``~/.config/simd/config.toml`` so it survives across runs.
    """
    if not config.user_id:
        user = await client.get_or_create_user("local@simd.local")
        uid = user.get("uid") or user.get("id")
        if not uid:
            raise _CliError(
                "agent didn't return a user id when bootstrapping the "
                "local CLI identity — check /api/users on the agent."
            )
        config.user_id = uid
        config.save()

    # Reuse the cached simulation when present — avoids a new project
    # per run, keeps `simd ls` showing all your CLI runs in one list.
    if config.simulation_id:
        return config.simulation_id

    sim = await client.create_simulation(config.user_id)
    sim_id = sim.get("id") or sim.get("simulation_id")
    if not sim_id:
        raise _CliError("simulation creation returned no id")
    config.simulation_id = sim_id
    config.save()
    return sim_id


def _count_patches(mesh_info: dict[str, Any]) -> int:
    p = mesh_info.get("patches") or []
    return len(p) if isinstance(p, list) else 0


def _patches_from_precheck(
    precheck: dict[str, Any],
    mesh_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalise the precheck response into a flat list of patch dicts.

    The precheck response varies in shape (boundaryHints vs
    boundaryConditions vs patch arrays); this helper picks the
    user-visible patches out of whichever shape it finds.
    """
    suggested = precheck.get("suggestedConfig") or {}
    bcs = (suggested.get("boundaryConditions")
           or precheck.get("boundaryHints")
           or {})
    if isinstance(bcs, dict) and "patches" in bcs:
        bcs = bcs["patches"]

    mesh_patches = mesh_info.get("patches") or []
    patches: list[dict[str, Any]] = []

    # Build one row per mesh patch.  The precheck's BCs are keyed by
    # patch name; we left-join into the mesh-known set so coupled
    # CHT patches and any precheck-discovered roles all show up.
    seen: set[str] = set()
    for mp in mesh_patches:
        name = mp.get("name") if isinstance(mp, dict) else None
        if not name:
            continue
        seen.add(name)
        bc = bcs.get(name, {}) if isinstance(bcs, dict) else {}
        patches.append({
            "name":   name,
            "role":   (bc.get("patch_class") or bc.get("role")
                       or mp.get("type") or "patch"),
            "fields": bc,
        })

    # Patches that exist only in the precheck (e.g. CHT couplings
    # auto-introduced by splitMeshRegions later) — append for visibility.
    if isinstance(bcs, dict):
        for name, bc in bcs.items():
            if name in seen or not isinstance(bc, dict):
                continue
            patches.append({
                "name":   name,
                "role":   bc.get("patch_class") or bc.get("role") or "patch",
                "fields": bc,
            })

    return patches


def _apply_overrides(
    suggested_config: dict[str, Any],
    overrides: dict[str, Any],
) -> None:
    """Merge the user's edits back into ``suggested_config`` in place.

    Overrides are keyed by patch name → field → {type, value}.
    """
    if not overrides:
        return
    bcs = suggested_config.setdefault("boundaryConditions", {})
    for patch_name, fields in overrides.items():
        patch_bc = bcs.setdefault(patch_name, {})
        for field_name, change in fields.items():
            target = patch_bc.setdefault(field_name, {})
            if "type" in change:
                target["type"] = change["type"]
            if "value" in change:
                target["value"] = change["value"]


def _build_start_request(
    *,
    prompt: str,
    mesh_info: dict[str, Any],
    suggested_config: dict[str, Any],
    simulation_id: str,
    user_id: str | None,
    max_retries: int,
    forced_solver: str | None,
) -> dict[str, Any]:
    """Assemble the StartRequest the agent expects on /ws/run."""
    # Fold the mesh_info into the config block so the agent can find
    # mesh_id, patches, check_mesh stats without a separate lookup.
    config = dict(suggested_config)
    config["mesh"] = {
        "mesh_id":    mesh_info.get("mesh_id") or simulation_id,
        "patches":    mesh_info.get("patches") or [],
        "check_mesh": mesh_info.get("check_mesh") or {},
        "cell_zones": mesh_info.get("cell_zones") or [],
    }
    if forced_solver:
        solver_block = config.setdefault("solver", {})
        solver_block["openfoamSolver"] = forced_solver

    return {
        "op":                "CFD_CODEGEN_RUN",
        "user_requirements": prompt,
        "simulation_config": config,
        "constraints":       {"max_retries": max_retries},
        "metadata":          {
            "simulation_id": simulation_id,
            "user_id":       user_id,
            "source":        "cli",
        },
    }
