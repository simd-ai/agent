"""5-stage progress display for ``simd run``.

The CLI translates the agent's noisy ``AgentEvent`` stream into a few
human-meaningful checkpoints.  This module owns the mapping from event
type → user-visible line.

Stages:
    1. uploading mesh         (CLI-internal, before any AgentEvent)
    2. running precheck       (CLI-internal, before any AgentEvent)
    3. generating case files  (codegen_*, file_generating, file_generated)
    4. running on sim-server  (sim_*, mesh_*, checkmesh_*, run_progress_*)
    5. results                (final + run summary)
"""

from __future__ import annotations

from typing import Any

from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

from simd_agent.cli.display import console, is_tty


# ── Stage banners ─────────────────────────────────────────────


def stage_banner(n: int, total: int, label: str) -> None:
    """Print a ``Stage 3/5 — generating case files`` heading."""
    console.print()
    console.print(f"[bold]Stage {n}/{total}[/] — {label}")


# ── Event → user line mapping ─────────────────────────────────


def render_event(event: dict[str, Any]) -> str | None:
    """Translate one ``AgentEvent`` into a CLI line.

    Returns ``None`` when the event is internal noise we don't want to
    show in default mode.  Use ``--verbose`` to surface everything.
    """
    etype = event.get("type", "")
    payload = event.get("payload") or {}

    # ── codegen ───────────────────────────────────────────────
    if etype == "file_generating":
        return f"  ► {payload.get('path', '?')}"
    if etype == "file_generated":
        path = payload.get("path", "?")
        chars = payload.get("char_count", 0)
        return f"  ✓ {path}  ({chars} chars)"
    if etype == "codegen_verification_complete":
        passed = payload.get("passed", False)
        n_issues = len(payload.get("issues") or [])
        if passed:
            return f"  validation: ✓  ({n_issues} issues, 0 errors)"
        return f"  validation: ✗  ({n_issues} issues)"
    if etype == "codegen_complete":
        size = payload.get("case_zip_size", 0)
        return f"  case packaged: {size:,} bytes"

    # ── sim-server pipeline ──────────────────────────────────
    if etype == "sim_extract_started":
        return "  ⠧ extracting case bundle"
    if etype == "mesh_conversion_started":
        return "  ⠧ converting mesh (gmshToFoam)"
    if etype == "split_mesh_started":
        return "  ⠧ splitting mesh into per-region trees"
    if etype == "split_mesh_complete":
        regions = ", ".join(payload.get("regions") or [])
        return f"  ✓ regions: {regions}"
    if etype == "boundary_types_fixed":
        n = payload.get("patches_fixed", 0)
        if payload.get("multi_region"):
            return f"  ✓ boundary types fixed: {n} patch(es) across regions"
        return f"  ✓ boundary types fixed: {n} patch(es)"
    if etype == "checkmesh_complete":
        return "  ✓ checkMesh OK"
    if etype == "sim_run_started":
        solver = payload.get("solver", "?")
        mode = payload.get("mode", "")
        return f"  ⠧ running {solver} ({mode})"

    # ── self-healing ──────────────────────────────────────────
    if etype == "sim_run_failed":
        exit_code = payload.get("exit_code", "?")
        return f"  ✗ solver failed (exit {exit_code}) — diagnosing…"
    if etype == "error_summary":
        root = (payload.get("root_cause") or "").split(".", 1)[0]
        return f"    ↪ {root}"
    if etype == "retrying":
        attempt = payload.get("attempt", "?")
        mx = payload.get("max_retries", "?")
        return f"  ↻ retrying ({attempt}/{mx})"

    # ── progress (only show milestones) ──────────────────────
    if etype == "run_progress_batch":
        items = payload.get("items") or []
        if items:
            last = items[-1]
            t = last.get("time", "?")
            return f"    iter {t}"
        return None

    # ── solver done ──────────────────────────────────────────
    if etype == "sim_run_complete":
        return "  ✓ solver finished"

    return None  # silent for the default mode


def render_verbose(event: dict[str, Any]) -> str:
    """Verbose mode: dump the full event as a single line."""
    seq = event.get("seq", "?")
    etype = event.get("type", "?")
    msg = event.get("message", "")
    return f"[{seq:>4}] {etype}: {msg}"


# ── Live spinner helper ──────────────────────────────────────


class StageSpinner:
    """A single-line spinner that updates as new sub-events arrive.

    Used inside Stage 4 to show ``running pimpleFoam … iter 250`` and
    update in place.  In a non-TTY context (CI, piped output) we fall
    back to one line per state change rather than animating.
    """

    def __init__(self, initial: str = "") -> None:
        self._current = initial
        self._live: Live | None = None
        if is_tty():
            self._live = Live(
                Spinner("dots", text=Text(initial)),
                console=console,
                transient=True,
                refresh_per_second=8,
            )

    def __enter__(self) -> "StageSpinner":
        if self._live is not None:
            self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)

    def update(self, text: str) -> None:
        """Change what the spinner is showing."""
        self._current = text
        if self._live is not None:
            self._live.update(Spinner("dots", text=Text(text)))
        else:
            # Non-TTY: just print the state change.
            console.print(text)
