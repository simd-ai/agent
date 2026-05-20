"""Final summary block printed at end of ``simd run``."""

from __future__ import annotations

from typing import Any

from simd_agent.cli.display import console


def render_summary(summary: dict[str, Any]) -> None:
    """One-screen post-run report.

    Carries the run_id, status, the case directory the sim-server wrote
    to, and the VTK URL when available.  Matches the layout the agent's
    /api/runs/{id}/summary endpoint returns.
    """
    console.print()
    status = summary.get("status", "?")
    if status == "succeeded":
        head = f"[bold green]✓ run {summary.get('run_id', '?')} succeeded[/]"
    elif status == "stopped":
        head = f"[bold yellow]■ run {summary.get('run_id', '?')} stopped[/]"
    elif status == "failed":
        head = f"[bold red]✗ run {summary.get('run_id', '?')} failed[/]"
    else:
        head = f"  run {summary.get('run_id', '?')} ({status})"
    console.print(head)

    started = summary.get("started_at")
    completed = summary.get("completed_at")
    if started:
        console.print(f"  started     {started}")
    if completed:
        console.print(f"  completed   {completed}")
    if (solver := summary.get("solver")):
        console.print(f"  solver      {solver}")
    if (sim_run := summary.get("sim_run_id")):
        console.print(f"  sim_run     {sim_run}")
    if (vtk := summary.get("vtk_url")):
        console.print(f"  vtk         {vtk}")
    if (err := summary.get("error")):
        console.print(f"  [red]error[/]       {err}")
