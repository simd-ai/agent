# simd_agent/chat_tools.py
"""Tools the CFD chat assistant can invoke via Gemini function-calling.

Each tool receives a dict of arguments (populated by the LLM) plus a shared
``SimulationSnapshot`` that carries all simulation state available for the
current session.  Tools return a plain dict which is serialised as the
``tool_result.data`` payload sent to the frontend.
"""

from __future__ import annotations

import json
import logging
import math
import textwrap
import traceback
from typing import Any

from google.genai import types

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation snapshot — everything the tools can read
# ---------------------------------------------------------------------------

class SimulationSnapshot:
    """Immutable bag of data assembled once per chat turn from the DB + the
    context the frontend sends.  Every tool receives this."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        simulation_id: str | None = None,
        mesh_id: str | None = None,
        physics: dict[str, Any] | None = None,
        solver: dict[str, Any] | None = None,
        fluid: dict[str, Any] | None = None,
        turbulence: dict[str, Any] | None = None,
        patches: dict[str, Any] | None = None,
        final_result: dict[str, Any] | None = None,
        vtk_result: dict[str, Any] | None = None,
        lint_result: dict[str, Any] | None = None,
        generated_files: dict[str, str] | None = None,
        sim_progress: list[dict[str, Any]] | None = None,
        simulation_config: dict[str, Any] | None = None,
        agent_run: dict[str, Any] | None = None,
        mesh_info: dict[str, Any] | None = None,
    ):
        self.run_id = run_id
        self.simulation_id = simulation_id
        self.mesh_id = mesh_id
        self.physics = physics or {}
        self.solver = solver or {}
        self.fluid = fluid or {}
        self.turbulence = turbulence or {}
        self.patches = patches or {}
        self.final_result = final_result or {}
        self.vtk_result = vtk_result or {}
        self.lint_result = lint_result or {}
        self.generated_files = generated_files or {}
        self.sim_progress = sim_progress or []
        self.simulation_config = simulation_config or {}
        self.agent_run = agent_run or {}
        self.mesh_info = mesh_info or {}

    def summary_dict(self) -> dict[str, Any]:
        """Compact representation injected into the LLM system prompt."""
        d: dict[str, Any] = {}
        if self.physics:
            d["physics"] = self.physics
        if self.solver:
            d["solver"] = self.solver
        if self.fluid:
            d["fluid"] = self.fluid
        if self.turbulence:
            d["turbulence"] = self.turbulence
        if self.patches:
            d["patches"] = self.patches
        if self.lint_result:
            d["lint_result"] = self.lint_result
        if self.vtk_result:
            d["vtk_result"] = self.vtk_result
        if self.final_result:
            d["final_result"] = self.final_result
        if self.mesh_info:
            d["mesh_info"] = self.mesh_info
        if self.sim_progress:
            prog = self.sim_progress
            total = len(prog)
            # Give LLM a representative sample: first 2, middle 1, last 5
            sample_steps = prog[:2]
            if total > 7:
                sample_steps += [prog[total // 2]]
            sample_steps += prog[-5:]
            # Summarise each sample step to only the most useful fields:
            # sim_time, iteration, and per-field final+initial residuals.
            def _slim_step(s: dict) -> dict:
                out: dict = {}
                if s.get("sim_time") is not None:
                    out["sim_time"] = s["sim_time"]
                out["iteration"] = s.get("iteration")
                res = s.get("residuals", {})
                if res:
                    out["residuals"] = {
                        f: {"initial": r.get("initial"), "final": r.get("final")}
                        for f, r in res.items()
                    }
                return out
            d["sim_progress_sample"] = [_slim_step(s) for s in sample_steps]
            d["sim_progress_total_steps"] = total
            # Surface the last step's final residuals prominently so the LLM
            # immediately sees the actual inner-loop convergence quality.
            last_res = prog[-1].get("residuals", {})
            if last_res:
                d["last_final_residuals"] = {
                    f: r.get("final") for f, r in last_res.items()
                }
                d["last_sim_time"] = prog[-1].get("sim_time")
        if self.generated_files:
            d["generated_file_paths"] = list(self.generated_files.keys())
        if self.agent_run:
            d["agent_run"] = {
                k: v for k, v in self.agent_run.items()
                if k in ("status", "label", "type", "error_message", "started_at", "completed_at")
            }
        return d


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def compute_field_stats(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Return available statistics for a field, mining all DB sources.

    Data sources (in priority order):
    1. vtk_result.fields — contains min/mean/max if the sim server computed them
    2. sim_progress residuals — Ux/Uy/Uz/p initial+final per time step → trend stats
    3. patch_configs (boundary conditions) — inlet/outlet set values

    Note: true spatial min/mean/max over the full mesh requires VTK post-processing.
    What Neon stores is residual history and boundary condition set-points, which
    gives a good picture of solver behaviour and boundary velocities.
    """
    field = args.get("field", "U")
    patch = args.get("patch")

    # ── Source 1: vtk_result ─────────────────────────────────────────────────
    vtk = snap.vtk_result or {}
    vtk_fields = vtk.get("fields", [])
    if isinstance(vtk_fields, list):
        for fdata in vtk_fields:
            if not isinstance(fdata, dict):
                continue
            fname = fdata.get("name") or fdata.get("field") or ""
            # Match "U", "Ux"/"Uy"/"Uz" or any requested name
            if fname == field or fname.lower() == field.lower():
                mn   = fdata.get("min")
                mx   = fdata.get("max")
                mean = fdata.get("mean")
                if any(v is not None for v in (mn, mx, mean)):
                    return {
                        "field": field,
                        "patch": patch,
                        "sim_time": vtk.get("time"),
                        "min": mn, "max": mx, "mean": mean,
                        "source": "vtk_result",
                        "chart": {
                            "type": "bar",
                            "title": f"{field} Statistics (t={vtk.get('time')} s)",
                            "xKey": "stat",
                            "yLabel": field,
                            "lines": [field],   # frontend uses "lines" for value key
                            "data": [
                                {"stat": "min",  field: mn},
                                {"stat": "mean", field: mean},
                                {"stat": "max",  field: mx},
                            ],
                        },
                    }

    # ── Source 2: sim_progress residuals ─────────────────────────────────────
    # The residuals table stores Ux, Uy, Uz (and p, k, …) with initial/final
    # per time step.  We derive: last-step final, min-final, max-final, mean-final
    # across all time steps — this is the best velocity "trend" data in Neon.
    if snap.sim_progress:
        # Normalise requested field: "U" → look for Ux/Uy/Uz components
        component_fields: list[str] = []
        if field.upper() == "U":
            component_fields = ["Ux", "Uy", "Uz"]
        else:
            component_fields = [field]

        results: dict[str, Any] = {}
        trend_data: list[dict[str, Any]] = []
        has_sim_time = snap.sim_progress[0].get("sim_time") is not None

        for comp in component_fields:
            finals: list[float] = []
            for step in snap.sim_progress:
                r = step.get("residuals", {}).get(comp)
                if r and r.get("final") is not None:
                    finals.append(_safe_float(r["final"], None) or 0)

            if finals:
                results[comp] = {
                    "final_residual_last": finals[-1],
                    "final_residual_min":  min(finals),
                    "final_residual_max":  max(finals),
                    "final_residual_mean": sum(finals) / len(finals),
                    "time_steps_sampled": len(finals),
                }

        # Build time-series trend chart.
        # Use the SAME format as compute_residual_trend:
        #   xKey = "sim_time" (or "iteration"), lines = [comp, ...],
        #   each row = { sim_time: 0.1, Ux: 6.8e-6, Uy: 5.2e-6 }
        x_key = "sim_time" if has_sim_time else "iteration"
        x_label = "Simulation Time (s)" if has_sim_time else "Iteration"

        # Collect all found components into merged rows
        x_to_row: dict[Any, dict[str, Any]] = {}
        for comp in component_fields:
            for step in snap.sim_progress:
                xv = step.get("sim_time") if has_sim_time else step.get("iteration")
                r = step.get("residuals", {}).get(comp)
                if r and xv is not None:
                    final_val = r.get("final")
                    if final_val is not None and final_val > 0:
                        if xv not in x_to_row:
                            x_to_row[xv] = {x_key: xv}
                        x_to_row[xv][comp] = final_val

        all_rows = [x_to_row[xv] for xv in sorted(x_to_row.keys())]
        all_rows = [r for r in all_rows if len(r) > 1]

        # Downsample to ≤ 300 points
        if len(all_rows) > 300:
            step_s = len(all_rows) / 300
            indices = {0, len(all_rows) - 1} | {int(i * step_s) for i in range(1, 299)}
            trend_data = [all_rows[i] for i in sorted(indices)]
        else:
            trend_data = all_rows

        if results:
            # Boundary velocity from patch_configs (the set-point, not computed)
            boundary_info: dict[str, Any] = {}
            if patch and snap.patches and patch in snap.patches:
                bc = snap.patches[patch]
                if isinstance(bc, dict):
                    u_bc = bc.get("U") or bc.get("velocity") or {}
                    if u_bc:
                        boundary_info = {"patch": patch, "bc_type": u_bc.get("type"), "set_value": u_bc.get("value")}

            active_lines = [c for c in component_fields if c in results]
            return {
                "field": field,
                "patch": patch,
                "source": "sim_progress_residuals",
                "components": results,
                "boundary_condition": boundary_info or None,
                "note": (
                    "Statistics derived from inner-loop (PISO/PIMPLE) final residuals "
                    "in sim_progress. True spatial min/mean/max require VTK post-processing."
                ),
                "chart": {
                    "type": "line",
                    "title": f"{field} Inner-Loop Residuals over Time",
                    "xKey": x_key,
                    "xLabel": x_label,
                    "yLabel": "Final Residual (log scale)",
                    "yScale": "log",
                    "lines": active_lines,   # ← same format as compute_residual_trend
                    "data": trend_data,
                },
            }

    # ── Source 3: boundary condition set-point only ───────────────────────────
    if snap.patches:
        target_patch = patch or next(iter(snap.patches), None)
        if target_patch and target_patch in snap.patches:
            bc = snap.patches[target_patch]
            if isinstance(bc, dict):
                u_bc = bc.get("U") or bc.get("velocity") or {}
                if u_bc:
                    return {
                        "field": field,
                        "patch": target_patch,
                        "source": "boundary_condition",
                        "bc_type": u_bc.get("type"),
                        "set_value": u_bc.get("value"),
                        "note": (
                            "Only boundary condition set-points are available. "
                            "Run compute_residual_trend for solver behaviour, "
                            "or load VTK results for spatial statistics."
                        ),
                    }

    return {
        "field": field,
        "patch": patch,
        "source": "none",
        "error": (
            f"No data found for field '{field}'. "
            "Available data sources: vtk_result (field list only), "
            "sim_progress (residuals for Ux/Uy/Uz/p/k/omega), "
            "patch boundary conditions."
        ),
    }


_TRANSIENT_SOLVERS = {
    "pimplefoam", "pisofoam", "icofoam", "interfoam", "interisofoam",
    "rhopimplefoam", "compressibleinterfoam", "compressibleinterisofoam",
    "compressiblemultiphaseinterfoam",
}


def _is_transient(snap: SimulationSnapshot) -> bool:
    """Return True if this is a time-marching (transient) simulation."""
    solver = (
        snap.solver.get("solver")
        or snap.physics.get("solver")
        or snap.simulation_config.get("solver", {}).get("solver")
        or ""
    ).lower()
    time_scheme = (snap.physics.get("timeScheme") or "").lower()
    if solver in _TRANSIENT_SOLVERS or time_scheme in ("transient", "unsteady"):
        return True
    # Fallback: if sim_progress has progressing sim_time values, it's transient —
    # this catches cases where solver/timeScheme metadata wasn't forwarded but the
    # progress data itself tells us it's time-marching.
    if snap.sim_progress and len(snap.sim_progress) > 1:
        times = [
            s.get("sim_time") for s in snap.sim_progress[:5]
            if s.get("sim_time") is not None
        ]
        if len(times) >= 2 and times[-1] != times[0]:
            return True
    return False


def _assess_steady_convergence(pts: list[dict[str, Any]]) -> str:
    """Convergence check for steady-state solvers (simpleFoam, rhoSimpleFoam…).

    Steady-state initial residuals should drop monotonically.
    Criteria:
      well_converged : dropped ≥ 3 orders  OR  final ≤ 1e-5
      converging     : dropped ≥ 1 order   OR  final ≤ 1e-3
      not_converged  : everything else
    """
    vals = [_safe_float(p.get("initial"), None) for p in pts]
    vals = [v for v in vals if v is not None and v > 0]
    if not vals:
        return "not_converged"

    first_val = vals[0]
    last_val = vals[-1]
    drop_orders = math.log10(first_val) - math.log10(last_val) if last_val > 0 else 0.0

    # Stability check on the tail
    tail = vals[max(0, len(vals) - max(len(vals) // 5, 3)):]
    mean_tail = sum(tail) / len(tail) if tail else last_val
    if len(tail) > 1 and mean_tail > 0:
        variance = sum((v - mean_tail) ** 2 for v in tail) / len(tail)
        cv = (variance ** 0.5) / mean_tail
        is_stable = cv < 0.15
    else:
        is_stable = False

    if last_val <= 1e-5 or drop_orders >= 3:
        status = "well_converged"
    elif last_val <= 1e-3 or drop_orders >= 1:
        status = "converging"
    else:
        status = "not_converged"

    if is_stable and last_val <= 1e-2 and status == "not_converged":
        status = "converging"

    return status


def _assess_transient_inner_convergence(pts: list[dict[str, Any]]) -> str:
    """Convergence check for transient solvers (pimpleFoam, icoFoam…).

    For transient simulations, initial residuals OSCILLATE with the physics —
    a high initial residual at the start of a time step is normal and does NOT
    mean the simulation is diverging.

    What matters is the FINAL residual: how well the inner PISO/PIMPLE loop
    solved the equations within each time step.  We check whether the solver
    consistently achieved tight inner convergence.

    Criteria:
      well_converged : median final residual ≤ 1e-5  (inner loop fully resolved)
      converging     : median final residual ≤ 1e-3  (inner loop mostly resolved)
      not_converged  : median final residual > 1e-3  (inner loop struggling)
    """
    finals = [_safe_float(p.get("final"), None) for p in pts]
    finals = [v for v in finals if v is not None and v > 0]

    if not finals:
        # No final residuals stored — fall back to checking if initial residuals
        # are at least consistently low (weak signal for transient health).
        initials = [_safe_float(p.get("initial"), None) for p in pts]
        initials = [v for v in initials if v is not None and v > 0]
        if not initials:
            return "not_converged"
        median_init = sorted(initials)[len(initials) // 2]
        # Transient initial residuals of ~0.1–0.3 are completely normal;
        # only flag as not_converged if they're very large (> 1, diverging)
        return "converging" if median_init <= 1.0 else "not_converged"

    # Use median over ALL time steps — robust against occasional spikes and
    # correctly reflects inner-loop quality throughout the full run.
    sorted_f = sorted(finals)
    median_final = sorted_f[len(sorted_f) // 2]

    if median_final <= 1e-5:
        return "well_converged"
    elif median_final <= 1e-3:
        return "converging"
    else:
        return "not_converged"


def compute_residual_trend(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Return residual convergence series for the requested field(s).

    X-axis is simulation time (sim_time) when available — matching the
    /api/runs/{id}/timesteps format — with iteration as fallback.

    Convergence assessment differs by solver type:
    - Transient (pimpleFoam, icoFoam…): checks inner-loop FINAL residuals.
      Initial residuals oscillate with physics and are NOT a divergence signal.
    - Steady-state (simpleFoam…): checks monotonic drop of INITIAL residuals.
    """
    fields_arg = args.get("fields")
    if isinstance(fields_arg, str):
        fields_arg = [f.strip() for f in fields_arg.split(",")]
    if not fields_arg:
        fields_arg = ["Ux", "Uy", "Uz", "p", "k", "omega", "epsilon"]

    if not snap.sim_progress:
        return {"error": "No simulation progress data available"}

    transient = _is_transient(snap)

    # Check whether sim_time is available (transient sims always have it)
    has_sim_time = any(
        step.get("sim_time") is not None for step in snap.sim_progress[:5]
    )
    x_key = "sim_time" if has_sim_time else "iteration"
    x_label = "Simulation Time (s)" if has_sim_time else "Iteration"

    series: dict[str, list[dict[str, Any]]] = {f: [] for f in fields_arg}
    for step in snap.sim_progress:
        x_val = step.get("sim_time") if has_sim_time else step.get("iteration", 0)
        it = step.get("iteration", 0)
        residuals = step.get("residuals", {})
        for f in fields_arg:
            if f in residuals:
                r = residuals[f]
                series[f].append({
                    x_key: x_val,
                    "iteration": it,
                    "initial": r.get("initial"),
                    "final": r.get("final"),
                })
    non_empty = {k: v for k, v in series.items() if v}
    if not non_empty:
        return {"error": "Requested fields not found in simulation progress"}

    # Convergence assessment — method depends on solver type
    convergence_assessment: dict[str, str] = {}
    for fname, pts in non_empty.items():
        if transient:
            convergence_assessment[fname] = _assess_transient_inner_convergence(pts)
        else:
            convergence_assessment[fname] = _assess_steady_convergence(pts)

    # Build recharts-ready rows keyed by x_key.
    #
    # Transient (pimpleFoam…): use FINAL residuals — these are the inner-loop
    # (PISO/PIMPLE) residuals after each time step, matching exactly what the
    # frontend residuals chart plots (values in 1e-6 to 1e-7 range).
    # Initial residuals oscillate with the physics (0.1–0.3) and are NOT useful
    # for the chart — they would show noise, not solver health.
    #
    # Steady-state (simpleFoam…): use INITIAL residuals — these show the
    # monotonic drop over iterations that engineers look for (1 → 1e-5).
    residual_key = "final" if transient else "initial"

    x_to_row: dict[Any, dict[str, Any]] = {}
    for fname, pts in non_empty.items():
        for pt in pts:
            xv = pt[x_key]
            if xv not in x_to_row:
                x_to_row[xv] = {x_key: xv}
            val = pt.get(residual_key)
            # Fall back to the other residual type if the preferred one is absent
            if val is None:
                val = pt.get("initial" if transient else "final")
            if val is not None and val > 0:
                x_to_row[xv][fname] = val

    all_rows = [x_to_row[xv] for xv in sorted(x_to_row.keys())]
    # Drop rows that have no field values (can happen if both initial/final are null)
    all_rows = [r for r in all_rows if len(r) > 1]

    # Downsample to ≤ 300 points (keep first, last, evenly spaced)
    MAX_CHART_POINTS = 300
    if len(all_rows) > MAX_CHART_POINTS:
        step_size = len(all_rows) / MAX_CHART_POINTS
        indices = {0, len(all_rows) - 1}
        indices |= {int(i * step_size) for i in range(1, MAX_CHART_POINTS - 1)}
        recharts_rows = [all_rows[i] for i in sorted(indices)]
    else:
        recharts_rows = all_rows

    # Determine last sim_time and completion fraction for context
    last_sim_time = snap.sim_progress[-1].get("sim_time") if snap.sim_progress else None
    end_time = (
        snap.solver.get("endTime") or snap.solver.get("end_time")
        or snap.simulation_config.get("solver", {}).get("endTime")
    )

    return {
        "total_iterations": len(snap.sim_progress),
        "last_sim_time": last_sim_time,
        "end_time": end_time,
        "solver_type": "transient" if transient else "steady_state",
        "residual_type_shown": residual_key,  # "final" or "initial"
        "convergence": convergence_assessment,
        "convergence_note": (
            "Transient simulation: chart shows inner-loop FINAL residuals per time step "
            "(PISO/PIMPLE quality). Initial residuals oscillate with physics and are normal. "
            "Convergence status = inner loop solved each time step well."
            if transient else
            "Steady-state simulation: chart shows initial residuals dropping over iterations. "
            "Convergence = residuals fell 3+ orders or reached ≤ 1e-5."
        ),
        # recharts-ready block — matches the frontend residuals chart exactly
        "chart": {
            "type": "line",
            "title": "Residual History",
            "xKey": x_key,
            "xLabel": x_label,
            "yLabel": f"{'Final' if transient else 'Initial'} Residual (log scale)",
            "yScale": "log",
            "lines": list(non_empty.keys()),
            "data": recharts_rows,
            "convergence": convergence_assessment,
            "solver_type": "transient" if transient else "steady_state",
        },
    }


def extract_velocity_profile(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Placeholder for line-probe extraction; returns what we can from boundary data."""
    patch = args.get("patch")
    axis = args.get("axis", "x")

    if patch and patch in snap.patches:
        bc = snap.patches[patch]
        u_val = bc.get("U", {}).get("value")
        return {
            "patch": patch,
            "axis": axis,
            "boundary_velocity": u_val,
            "note": "Full profile extraction requires VTK post-processing. "
                    "Boundary value shown.",
        }

    return {
        "patch": patch,
        "axis": axis,
        "note": "Patch not found or velocity data not available.",
    }


def run_python_analysis(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Execute arbitrary Python code to compute derived scientific quantities.

    The code runs in a restricted namespace with access to ``math``, basic
    builtins, and the full simulation snapshot (``snap``).  The code must
    assign its final answer to a variable called ``result``.
    """
    code: str = args.get("code", "")
    description: str = args.get("description", "Custom analysis")
    if not code.strip():
        return {"error": "No code provided"}

    allowed_builtins = {
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
        "len": len, "range": range, "enumerate": enumerate, "zip": zip,
        "sorted": sorted, "reversed": reversed, "list": list, "dict": dict,
        "tuple": tuple, "set": set, "float": float, "int": int, "str": str,
        "bool": bool, "True": True, "False": False, "None": None,
        "isinstance": isinstance, "type": type, "print": lambda *a, **kw: None,
        "map": map, "filter": filter, "any": any, "all": all,
        "pow": pow, "divmod": divmod,
    }

    namespace: dict[str, Any] = {
        "__builtins__": allowed_builtins,
        "math": math,
        "snap": snap,
        "physics": snap.physics,
        "solver": snap.solver,
        "fluid": snap.fluid,
        "turbulence": snap.turbulence,
        "patches": snap.patches,
        "final_result": snap.final_result,
        "vtk_result": snap.vtk_result,
        "lint_result": snap.lint_result,
        "sim_progress": snap.sim_progress,
        "mesh_info": snap.mesh_info,
        "generated_files": snap.generated_files,
    }

    try:
        exec(code, namespace)  # noqa: S102
    except Exception:
        return {
            "error": f"Code execution failed:\n{traceback.format_exc()}",
            "code": code,
        }

    result = namespace.get("result", "__NOT_SET__")
    if result == "__NOT_SET__":
        return {"warning": "Code ran but did not set a `result` variable", "code": code}

    if isinstance(result, (dict, list, str, int, float, bool, type(None))):
        return {"description": description, "result": result}
    return {"description": description, "result": str(result)}


def generate_report(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Produce a structured report (markdown + typed data block) for the simulation.

    The LLM passes ``sections`` to control what's included, and ``focus`` for a
    user-requested emphasis (e.g. "convergence", "boundary conditions", "results").
    The caller (service.py) emits the ``report_markdown`` as an artifact so the
    frontend can render/export it as a PDF via react-pdf.
    """
    focus: str = args.get("focus", "")
    sections: list[str] = args.get("sections") or []

    # Auto-select sections based on focus keyword and available data
    if not sections:
        if "convergence" in focus or "residual" in focus:
            sections = ["summary", "residuals"]
        elif "bound" in focus or "bc" in focus:
            sections = ["summary", "boundary_conditions"]
        elif "mesh" in focus:
            sections = ["summary", "mesh"]
        elif "result" in focus or "output" in focus:
            sections = ["summary", "results", "residuals"]
        else:
            # Full report — include everything that has data
            sections = ["summary", "physics", "mesh", "boundary_conditions",
                        "residuals", "results"]

    parts: list[str] = []

    # ── Summary ──────────────────────────────────────────────────────────────
    if "summary" in sections:
        solver_name = snap.solver.get("solver") or snap.physics.get("solver") or "N/A"
        regime = (snap.physics.get("flowType") or snap.physics.get("flowRegime") or "N/A")
        re_num = snap.physics.get("Re") or snap.physics.get("reynoldsNumber") or "N/A"
        turb_model = snap.turbulence.get("model") or "N/A"
        status = snap.agent_run.get("status", "N/A")
        started = snap.agent_run.get("started_at", "N/A")
        completed = snap.agent_run.get("completed_at", "N/A")
        parts.append(
            "## Simulation Summary\n\n"
            "| Property | Value |\n|---|---|\n"
            f"| Solver | `{solver_name}` |\n"
            f"| Flow regime | {regime} |\n"
            f"| Reynolds number | {re_num} |\n"
            f"| Turbulence model | {turb_model} |\n"
            f"| Run status | **{status}** |\n"
            f"| Started | {started} |\n"
            f"| Completed | {completed} |"
        )

    # ── Physics ───────────────────────────────────────────────────────────────
    if "physics" in sections and snap.physics:
        rows = "\n".join(f"| {k} | {v} |" for k, v in snap.physics.items())
        if snap.fluid:
            fluid_rows = "\n".join(f"| {k} | {v} |" for k, v in snap.fluid.items())
            rows += "\n" + fluid_rows
        parts.append(f"## Physics & Fluid Properties\n\n| Setting | Value |\n|---|---|\n{rows}")

    # ── Mesh ──────────────────────────────────────────────────────────────────
    if "mesh" in sections and snap.mesh_info:
        mi = snap.mesh_info
        has_errors = mi.get("hasErrors", False)
        quality_badge = "⚠️ Issues detected" if has_errors else "✅ No issues"
        parts.append(
            "## Mesh Quality\n\n"
            f"- **Cells:** {mi.get('cells', 'N/A')}\n"
            f"- **Faces:** {mi.get('faces', 'N/A')}\n"
            f"- **Points:** {mi.get('points', 'N/A')}\n"
            f"- **Max aspect ratio:** {mi.get('maxAspectRatio', 'N/A')}\n"
            f"- **Max skewness:** {mi.get('maxSkewness', 'N/A')}\n"
            f"- **Quality:** {quality_badge}"
        )
        if mi.get("messages"):
            msgs = "\n".join(f"  - {m}" for m in mi["messages"])
            parts[-1] += f"\n\n**checkMesh messages:**\n{msgs}"

    # ── Boundary Conditions ───────────────────────────────────────────────────
    if "boundary_conditions" in sections and snap.patches:
        bc_lines = []
        for pname, pdata in snap.patches.items():
            if isinstance(pdata, dict):
                # Flatten key fields for readability
                u_type = (pdata.get("U") or {}).get("type", "—")
                p_type = (pdata.get("p") or {}).get("type", "—")
                bc_lines.append(
                    f"| **{pname}** | U: `{u_type}` | p: `{p_type}` |"
                )
        if bc_lines:
            parts.append(
                "## Boundary Conditions\n\n"
                "| Patch | Velocity (U) | Pressure (p) |\n|---|---|---|\n"
                + "\n".join(bc_lines)
            )

    # ── Residuals / Convergence ───────────────────────────────────────────────
    convergence_chart: dict[str, Any] | None = None
    if "residuals" in sections and snap.sim_progress:
        # Always use the FINAL step for the residuals table — that is the most
        # meaningful snapshot: for steady-state it reflects how far the solution
        # has converged; for transient it shows the inner-loop quality at the
        # last time step.
        last = snap.sim_progress[-1]
        residuals = last.get("residuals", {})
        total_iters = len(snap.sim_progress)
        last_sim_time = last.get("sim_time")

        if residuals:
            transient = _is_transient(snap)
            field_series: dict[str, list[dict[str, Any]]] = {}
            for step in snap.sim_progress:
                for f, r in step.get("residuals", {}).items():
                    field_series.setdefault(f, []).append({
                        "initial": r.get("initial"),
                        "final": r.get("final"),
                    })

            rows_list = []
            for f, r in residuals.items():
                pts = field_series.get(f, [{"initial": r.get("initial"), "final": r.get("final")}])
                if transient:
                    status = _assess_transient_inner_convergence(pts)
                    # For transient runs, the FINAL residual (inner-loop) is the
                    # key quality metric; the initial residual naturally oscillates
                    # with the physics and is NOT a convergence indicator.
                    key_residual = r.get("final", "N/A")
                    key_label = f"**{key_residual}** _(inner-loop final)_"
                else:
                    status = _assess_steady_convergence(pts)
                    key_residual = r.get("initial", "N/A")
                    key_label = str(key_residual)
                badge = {"well_converged": "✅", "converging": "🔶", "not_converged": "❌"}.get(status, "—")
                rows_list.append(
                    f"| {f} | {r.get('initial', 'N/A')} | {key_label} | {badge} |"
                )
            rows = "\n".join(rows_list)

            # Build section heading that makes it clear these are FINAL-step values
            if transient and last_sim_time is not None:
                residual_heading = (
                    f"## Residuals — Final Time Step (t = {last_sim_time:.4g} s) "
                    f"— {total_iters} steps total\n\n"
                    "> **Transient simulation**: convergence status is based on the "
                    "inner-loop **FINAL** residual (PISO/PIMPLE quality per time step). "
                    "Initial residuals oscillate with the physics and are shown for reference only.\n\n"
                )
            else:
                residual_heading = (
                    f"## Residuals — Final Iteration ({last.get('iteration', total_iters)}"
                    f" / {total_iters})\n\n"
                )

            parts.append(
                residual_heading
                + "| Field | Initial | Final (key metric) | Status |\n|---|---|---|---|\n"
                + rows
            )

        courant = last.get("courant", {})
        if courant:
            parts.append(
                f"**Courant number** — mean: {courant.get('mean', 'N/A')}, "
                f"max: {courant.get('max', 'N/A')}"
            )

        # Build convergence chart data for frontend PDF embedding
        trend = compute_residual_trend({"fields": list(residuals.keys())}, snap)
        if "chart" in trend:
            convergence_chart = trend["chart"]

    # ── Final Results ──────────────────────────────────────────────────────────
    if "results" in sections and snap.final_result:
        fr = snap.final_result
        if isinstance(fr, dict):
            rows = "\n".join(f"| {k} | {v} |" for k, v in fr.items())
            parts.append(f"## Final Results\n\n| Field | Value |\n|---|---|\n{rows}")
        else:
            parts.append(f"## Final Results\n\n```json\n{json.dumps(fr, indent=2, default=str)}\n```")

    report_md = "\n\n---\n\n".join(parts) if parts else "No simulation data available yet."

    result: dict[str, Any] = {
        "report_markdown": report_md,
        "sections_included": sections,
        # Typed data block for the frontend PDF renderer — all raw values
        "report_data": {
            "solver": snap.solver.get("solver") or snap.physics.get("solver"),
            "status": snap.agent_run.get("status"),
            "re": snap.physics.get("Re") or snap.physics.get("reynoldsNumber"),
            "flow_type": snap.physics.get("flowType") or snap.physics.get("flowRegime"),
            "turbulence_model": snap.turbulence.get("model"),
            "mesh": snap.mesh_info,
            "final_result": snap.final_result,
            "vtk_result": snap.vtk_result,
            "last_residuals": snap.sim_progress[-1].get("residuals") if snap.sim_progress else None,
            "total_iterations": len(snap.sim_progress),
        },
    }
    if convergence_chart:
        result["chart"] = convergence_chart  # service.py will emit this as a "chart" artifact too

    return result


def analyze_chart(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Analyze a chart / VTK field and provide textual interpretation."""
    chart_type = args.get("chart_type", "residuals")
    field = args.get("field")

    if chart_type == "residuals":
        trend_result = compute_residual_trend({"fields": field}, snap)
        return {
            "chart_type": "residuals",
            "analysis": trend_result,
            "note": "Residual data extracted from simulation progress.",
        }

    if chart_type == "field" and field:
        stats = compute_field_stats({"field": field}, snap)
        return {"chart_type": "field", "field": field, "stats": stats}

    return {"chart_type": chart_type, "note": "Insufficient data for analysis."}


def read_generated_file(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Return the contents of a generated OpenFOAM file fetched from the database."""
    path = args.get("path", "").strip()
    if not snap.generated_files:
        return {
            "error": "No generated files found. The simulation code has not been generated yet, "
                     "or the run ID is missing."
        }

    available = list(snap.generated_files.keys())

    # Exact match first
    content = snap.generated_files.get(path)

    # Case-insensitive / suffix match (e.g. user says "U", "fvSchemes", "controlDict")
    if content is None and path:
        path_lower = path.lower()
        for key in available:
            key_lower = key.lower()
            if key_lower == path_lower or key_lower.endswith("/" + path_lower):
                content = snap.generated_files[key]
                path = key
                break

    if content is None:
        return {
            "error": f"File '{path}' not found in generated files.",
            "available_files": available,
            "hint": "Use the exact path as listed in available_files (e.g. '0/U', 'system/fvSchemes').",
        }

    return {
        "path": path,
        "content": content,
        "char_count": len(content),
        "source": "database (file_generation_map)",
    }


# ---------------------------------------------------------------------------
# Tool registry  (name → callable)
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "compute_field_stats": compute_field_stats,
    "compute_residual_trend": compute_residual_trend,
    "extract_velocity_profile": extract_velocity_profile,
    "run_python_analysis": run_python_analysis,
    "generate_report": generate_report,
    "analyze_chart": analyze_chart,
    "read_generated_file": read_generated_file,
}


# ---------------------------------------------------------------------------
# Gemini tool schemas (types.Tool)
# ---------------------------------------------------------------------------

CHAT_TOOLS_SCHEMA = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="compute_field_stats",
            description=(
                "Compute min/mean/max statistics of a simulation field "
                "(e.g. U, p, k, omega, T) from VTK results or residual data. "
                "Use this whenever the user asks about values of a field."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "field": types.Schema(type="STRING", description="Field name, e.g. U, p, k, omega, T"),
                    "patch": types.Schema(type="STRING", description="Patch name (optional, e.g. inlet, outlet)"),
                    "time_step": types.Schema(type="STRING", description="Time step (optional)"),
                },
                required=["field"],
            ),
        ),
        types.FunctionDeclaration(
            name="compute_residual_trend",
            description=(
                "Return residual convergence history for one or more fields. "
                "Use this to assess whether the simulation converged, to explain "
                "residual plots, or when the user asks about convergence."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(
                        type="STRING",
                        description="Comma-separated field names, e.g. 'Ux,p,k'. Leave empty for all.",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="extract_velocity_profile",
            description=(
                "Extract velocity profile data at a given boundary patch. "
                "Use when the user asks for velocity distribution at inlet/outlet/wall."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "patch": types.Schema(type="STRING", description="Patch name, e.g. inlet, outlet"),
                    "axis": types.Schema(type="STRING", description="Axis for profile extraction: x, y, or z"),
                },
                required=["patch"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_python_analysis",
            description=(
                "Execute Python code to compute derived scientific/numerical values. "
                "Use this for any calculation the user requests that isn't covered by other tools: "
                "Reynolds number from scratch, pressure drop, flow rate, Mach number, drag coefficient, "
                "unit conversions, dimensional analysis, etc. "
                "The code has access to `math` and all simulation data via variables: "
                "`physics`, `solver`, `fluid`, `turbulence`, `patches`, `final_result`, "
                "`vtk_result`, `sim_progress`, `mesh_info`, `generated_files`. "
                "The code MUST assign its answer to a variable called `result`."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "code": types.Schema(
                        type="STRING",
                        description="Python code to execute. Must set `result = ...` as final output.",
                    ),
                    "description": types.Schema(
                        type="STRING",
                        description="Brief description of what the code computes.",
                    ),
                },
                required=["code", "description"],
            ),
        ),
        types.FunctionDeclaration(
            name="generate_report",
            description=(
                "Generate a full simulation report (markdown + structured data) that the "
                "frontend will render as a PDF. Call this whenever the user asks to generate "
                "a report, export results, download a PDF, or get a complete simulation summary. "
                "Use the 'focus' parameter to tailor the report to what the user asked for "
                "(e.g. 'convergence', 'boundary conditions', 'results'). "
                "Omit 'sections' to auto-select based on focus or include everything."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "focus": types.Schema(
                        type="STRING",
                        description=(
                            "What the user specifically wants to know, in a few words. "
                            "Examples: 'convergence', 'boundary conditions', 'results', "
                            "'mesh quality', 'full summary'. Used to auto-select sections "
                            "when 'sections' is not provided."
                        ),
                    ),
                    "sections": types.Schema(
                        type="ARRAY",
                        items=types.Schema(type="STRING"),
                        description=(
                            "Explicit list of sections to include. Options: summary, physics, "
                            "mesh, boundary_conditions, residuals, results. "
                            "Leave empty to auto-select based on 'focus'."
                        ),
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="analyze_chart",
            description=(
                "Analyze a simulation chart (residual plot, field contour, etc.) "
                "and provide a textual interpretation. Use when the user asks you "
                "to explain what a chart shows."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "chart_type": types.Schema(
                        type="STRING",
                        description="Type of chart: 'residuals' or 'field'",
                    ),
                    "field": types.Schema(
                        type="STRING",
                        description="Field name if chart_type is 'field'",
                    ),
                },
                required=["chart_type"],
            ),
        ),
        types.FunctionDeclaration(
            name="read_generated_file",
            description=(
                "Fetch the full content of a generated OpenFOAM case file from the database. "
                "ALWAYS call this tool first whenever the user asks to see, review, explain, "
                "or understand any simulation file (e.g. 'show me the U file', 'what does "
                "fvSolution look like', 'explain the boundary conditions in 0/p'). "
                "Do NOT guess or reproduce file contents from memory — always read from DB."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "path": types.Schema(
                        type="STRING",
                        description=(
                            "Relative path of the file inside the OpenFOAM case. "
                            "Examples: 'system/controlDict', 'system/fvSchemes', "
                            "'system/fvSolution', '0/U', '0/p', '0/k', '0/omega', "
                            "'constant/transportProperties', "
                            "'constant/turbulenceProperties'. "
                            "Partial names like 'U', 'fvSchemes' also work."
                        ),
                    ),
                },
                required=["path"],
            ),
        ),
    ]
)
