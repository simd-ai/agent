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

            # ── Global statistics over the FULL sim_progress ─────────────────
            # The sample above is truncated for context size; these aggregates
            # ensure the LLM always sees accurate min/max/mean computed over
            # every step — not just the 8 shown in sim_progress_sample.
            field_finals: dict[str, list[float]] = {}
            field_initials: dict[str, list[float]] = {}
            max_courant: float | None = None
            for step in prog:
                for f, r in step.get("residuals", {}).items():
                    for bucket, key in ((field_finals, "final"), (field_initials, "initial")):
                        raw = r.get(key)
                        if raw is not None:
                            try:
                                v = float(raw)
                                if v > 0:
                                    bucket.setdefault(f, []).append(v)
                            except (TypeError, ValueError):
                                pass
                co = step.get("courant", {})
                if co:
                    try:
                        cv = float(co.get("max") or 0)
                        if cv > 0 and (max_courant is None or cv > max_courant):
                            max_courant = cv
                    except (TypeError, ValueError):
                        pass

            residual_global: dict[str, Any] = {}
            for f, vals in field_finals.items():
                residual_global[f] = {
                    "final_min":  min(vals),
                    "final_max":  max(vals),
                    "final_mean": sum(vals) / len(vals),
                    "final_last": vals[-1],
                    "steps_with_data": len(vals),
                }
            for f, vals in field_initials.items():
                entry = residual_global.setdefault(f, {})
                entry["initial_min"]  = min(vals)
                entry["initial_max"]  = max(vals)
                entry["initial_last"] = vals[-1]

            global_stats: dict[str, Any] = {
                "total_steps": total,
                "sim_time_range": {
                    "start": prog[0].get("sim_time"),
                    "end":   prog[-1].get("sim_time"),
                },
            }
            if residual_global:
                global_stats["residuals"] = residual_global
            if max_courant is not None:
                global_stats["max_courant_number"] = max_courant
            d["sim_progress_global_stats"] = global_stats
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
                stat: dict[str, Any] = {"time_steps_sampled": len(finals)}
                errors: dict[str, str] = {}
                for key, fn in (
                    ("final_residual_last", lambda v: v[-1]),
                    ("final_residual_min",  min),
                    ("final_residual_max",  max),
                    ("final_residual_mean", lambda v: sum(v) / len(v)),
                ):
                    try:
                        stat[key] = fn(finals)
                    except Exception as exc:
                        errors[key] = str(exc)
                if errors:
                    stat["errors"] = errors
                results[comp] = stat

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


def compute_residual_trend(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Return residual history series for the requested field(s).

    X-axis is simulation time (sim_time) when available — matching the
    /api/runs/{id}/timesteps format — with iteration as fallback.

    For transient solvers (pimpleFoam, icoFoam…): shows inner-loop FINAL
    residuals. For steady-state (simpleFoam…): shows INITIAL residuals.
    No convergence assessment is performed.
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

    # Build recharts-ready rows keyed by x_key.
    #
    # Transient (pimpleFoam…): use FINAL residuals — these are the inner-loop
    # (PISO/PIMPLE) residuals after each time step.
    # Steady-state (simpleFoam…): use INITIAL residuals — these show the drop
    # over iterations that engineers look for.
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
        "residual_type_shown": residual_key,
        "chart": {
            "type": "line",
            "title": "Residual History",
            "xKey": x_key,
            "xLabel": x_label,
            "yLabel": f"{'Final' if transient else 'Initial'} Residual (log scale)",
            "yScale": "log",
            "lines": list(non_empty.keys()),
            "data": recharts_rows,
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


def _plain_solver_explanation(solver_name: str, transient: bool) -> str:
    """Return a plain-language explanation of what the chosen solver does."""
    s = (solver_name or "").lower()
    if "simplefoam" in s or s == "simple":
        return (
            "The solver chosen for this simulation is **simpleFoam**. "
            "This solver is designed for steady-state, incompressible flow — meaning it "
            "computes the final, settled flow field rather than tracking how the fluid moves "
            "step by step over time. It is well suited for situations where the flow "
            "conditions do not change significantly once the system reaches equilibrium, "
            "such as flow around a stationary object or through a duct at constant speed."
        )
    if "rhosimplefoam" in s:
        return (
            "The solver chosen is **rhoSimpleFoam**. "
            "This is a steady-state solver for compressible flow, meaning it accounts for "
            "the fact that the fluid's density can change — something that becomes important "
            "at high speeds (typically above Mach 0.3). It finds the long-term average "
            "flow field, not the time evolution."
        )
    if "pimplefoam" in s:
        return (
            "The solver chosen is **pimpleFoam**. "
            "This is a time-stepping solver — it advances the simulation forward in small "
            "time increments, capturing how the flow evolves over time. It is used when the "
            "flow is expected to change dynamically, for example due to vortex shedding, "
            "oscillating boundaries, or unsteady inlet conditions."
        )
    if "rhopimplefoam" in s:
        return (
            "The solver chosen is **rhoPimpleFoam**. "
            "This is a time-stepping solver for compressible flow, combining time accuracy "
            "with the ability to model density changes in the fluid. It is used when the "
            "flow is both fast (compressible effects matter) and time-dependent."
        )
    if "interfoam" in s or "interisofoam" in s:
        return (
            "The solver chosen is **interFoam**. "
            "This solver is designed for two-phase flows — situations where two different "
            "fluids (such as water and air) coexist and interact. It tracks the boundary "
            "(called the interface) between the two fluids as it moves and deforms over time."
        )
    if "pisofoam" in s or "icofoam" in s:
        return (
            f"The solver chosen is **{solver_name}**. "
            "This is a time-stepping solver for incompressible flow, advancing the "
            "simulation in small time steps to capture how the flow changes over time."
        )
    if transient:
        return (
            f"The solver **{solver_name}** is a time-stepping (transient) solver. "
            "It advances the simulation forward in time, step by step, capturing the "
            "dynamic evolution of the flow field."
        )
    return (
        f"The solver **{solver_name}** is a steady-state solver. "
        "It computes the long-term average flow field without tracking time evolution."
    )


def _plain_turbulence_explanation(model: str) -> str:
    """Return a plain-language explanation of the turbulence model."""
    m = (model or "").lower()
    if "komegasst" in m or "k-omega" in m or "komega" in m:
        return (
            "The turbulence model selected is **k-omega SST** (Shear Stress Transport). "
            "Turbulence refers to the chaotic, swirling motion that develops in most "
            "real-world flows. Simulating every tiny turbulent eddy directly would require "
            "enormous computing power, so engineers use turbulence models — mathematical "
            "approximations that capture the average effect of turbulence on the flow. "
            "k-omega SST is one of the most trusted models in engineering because it "
            "performs well both close to solid surfaces (like walls of a pipe or body of a "
            "car) and in the open flow further away."
        )
    if "kepsilon" in m or "k-epsilon" in m or "keps" in m:
        return (
            "The turbulence model selected is **k-epsilon**. "
            "This is a widely used two-equation model that characterises turbulence using "
            "two quantities: the turbulent kinetic energy (k) and its rate of dissipation "
            "(epsilon). It performs well in free-stream regions and jets but is less "
            "accurate very close to walls."
        )
    if "spalart" in m or "sa" == m:
        return (
            "The turbulence model selected is **Spalart-Allmaras**. "
            "This is a simpler, one-equation model developed for aerodynamic applications "
            "such as flow over aircraft wings and surfaces. It is computationally "
            "efficient and works well for attached boundary layer flows."
        )
    if "laminar" in m or not m:
        return (
            "The flow is treated as **laminar** — meaning it moves in smooth, ordered "
            "layers without turbulent mixing. This is valid when the Reynolds number is "
            "low, typically in slow flows or with very viscous fluids. No turbulence "
            "model is needed in this case."
        )
    if "realizablek" in m or "realizable" in m:
        return (
            "The turbulence model selected is **Realizable k-epsilon**. "
            "This is an improved version of the standard k-epsilon model that satisfies "
            "certain mathematical constraints (realizability conditions), making it more "
            "accurate for flows with strong streamline curvature and rotation."
        )
    return (
        f"The turbulence model selected is **{model}**. "
        "Turbulence models are mathematical approximations that capture the average "
        "effect of chaotic flow fluctuations without having to simulate every detail, "
        "making the computation practical on standard hardware."
    )


def _plain_reynolds_explanation(re_num: Any, regime: str) -> str:
    """Return a plain-language explanation of the Reynolds number and flow regime."""
    regime_lower = (regime or "").lower()
    try:
        re_val = float(re_num)
        re_str = f"{re_val:,.0f}"
    except (TypeError, ValueError):
        re_str = str(re_num)
        re_val = None

    regime_desc = ""
    if "turbulent" in regime_lower:
        regime_desc = (
            "At this Reynolds number the flow is **turbulent** — the fluid moves in a "
            "chaotic, mixing fashion with eddies and swirls at many scales. "
            "This is the most common regime in engineering applications."
        )
    elif "laminar" in regime_lower:
        regime_desc = (
            "At this Reynolds number the flow is **laminar** — the fluid moves in "
            "smooth, parallel layers with little mixing between them. "
            "This regime occurs at low speeds or with very viscous fluids."
        )
    elif "transitional" in regime_lower:
        regime_desc = (
            "The flow is in the **transitional** regime — somewhere between laminar and "
            "fully turbulent. Predicting this regime accurately is challenging and "
            "typically requires specialised models."
        )

    re_explanation = (
        f"The **Reynolds number** for this simulation is **{re_str}**. "
        "The Reynolds number is a dimensionless quantity that describes whether a flow "
        "will be smooth and orderly (laminar) or chaotic and mixing (turbulent). "
        "It combines the fluid speed, the size of the geometry, and the fluid's "
        "viscosity (resistance to flow) into a single number. "
        "As a rough guide: below about 2,300 the flow is typically laminar; "
        "above about 4,000 it is typically turbulent."
    )
    if regime_desc:
        re_explanation += f" {regime_desc}"
    return re_explanation


def _plain_bc_explanation(patches: dict[str, Any]) -> str:
    """Return a short plain-language note about boundary condition types found."""
    seen_types: set[str] = set()
    for pdata in patches.values():
        if isinstance(pdata, dict):
            for field_bc in pdata.values():
                if isinstance(field_bc, dict):
                    t = field_bc.get("type", "")
                    if t:
                        seen_types.add(t.lower())

    explanations: list[str] = []
    if "fixedvalue" in seen_types:
        explanations.append(
            "**Fixed value** boundaries set an exact quantity (e.g. a specific velocity or "
            "temperature) at that surface — like specifying that fluid enters at 10 m/s."
        )
    if "zerogradient" in seen_types:
        explanations.append(
            "**Zero gradient** boundaries allow the quantity to flow freely out of the domain "
            "without any artificial reflection — commonly used at outlets."
        )
    if "noslip" in seen_types or "noSlip" in seen_types:
        explanations.append(
            "**No-slip** walls enforce zero velocity at solid surfaces, matching the "
            "physical behaviour that fluid sticks to a wall."
        )
    if "symmetry" in seen_types or "symmetryplane" in seen_types:
        explanations.append(
            "**Symmetry** boundaries allow the simulation to model only half (or a fraction) "
            "of the geometry, reducing computation while capturing the full physics."
        )
    if "inletoutlet" in seen_types:
        explanations.append(
            "**Inlet/outlet** boundaries switch automatically between inlet and outlet "
            "behaviour depending on flow direction — useful when backflow is possible."
        )
    if not explanations:
        return ""
    return (
        "Boundary conditions define what happens at every surface of the simulation domain. "
        "Think of them as the rules that tell the solver what the fluid is doing at each "
        "wall, inlet, and outlet. In this simulation:\n\n"
        + "\n".join(f"- {e}" for e in explanations)
    )


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

    solver_name = snap.solver.get("solver") or snap.physics.get("solver") or "N/A"
    regime = snap.physics.get("flowType") or snap.physics.get("flowRegime") or "N/A"
    re_num = snap.physics.get("Re") or snap.physics.get("reynoldsNumber") or "N/A"
    turb_model = snap.turbulence.get("model") or "N/A"
    transient = _is_transient(snap)

    # ── Summary ──────────────────────────────────────────────────────────────
    if "summary" in sections:
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

    # ── Plain-language explanation ─────────────────────────────────────────────
    explanation_parts: list[str] = []

    solver_expl = _plain_solver_explanation(solver_name, transient)
    if solver_expl:
        explanation_parts.append(solver_expl)

    re_expl = _plain_reynolds_explanation(re_num, regime)
    if re_expl:
        explanation_parts.append(re_expl)

    turb_expl = _plain_turbulence_explanation(turb_model)
    if turb_expl:
        explanation_parts.append(turb_expl)

    if explanation_parts:
        parts.append(
            "## About This Simulation\n\n"
            + "\n\n".join(explanation_parts)
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
        quality_note = "Issues detected — see messages below" if has_errors else "No issues detected"
        mesh_expl = (
            "\n\nThe mesh is the grid that divides the simulation domain into small cells. "
            "The solver computes the flow equations separately in each cell, then assembles "
            "the full picture. A finer mesh (more cells) generally gives more accurate "
            "results but requires more computation time. "
            "Aspect ratio measures how stretched a cell is — very elongated cells can "
            "reduce accuracy. Skewness measures how distorted a cell is from an ideal shape."
        )
        parts.append(
            "## Mesh Quality\n\n"
            f"- **Cells:** {mi.get('cells', 'N/A')}\n"
            f"- **Faces:** {mi.get('faces', 'N/A')}\n"
            f"- **Points:** {mi.get('points', 'N/A')}\n"
            f"- **Max aspect ratio:** {mi.get('maxAspectRatio', 'N/A')}\n"
            f"- **Max skewness:** {mi.get('maxSkewness', 'N/A')}\n"
            f"- **Quality:** {quality_note}"
            + mesh_expl
        )
        if mi.get("messages"):
            msgs = "\n".join(f"  - {m}" for m in mi["messages"])
            parts[-1] += f"\n\n**checkMesh messages:**\n{msgs}"

    # ── Boundary Conditions ───────────────────────────────────────────────────
    if "boundary_conditions" in sections and snap.patches:
        bc_lines = []
        for pname, pdata in snap.patches.items():
            if isinstance(pdata, dict):
                u_type = (pdata.get("U") or {}).get("type", "—")
                p_type = (pdata.get("p") or {}).get("type", "—")
                bc_lines.append(
                    f"| **{pname}** | U: `{u_type}` | p: `{p_type}` |"
                )
        if bc_lines:
            bc_expl = _plain_bc_explanation(snap.patches)
            bc_section = (
                "## Boundary Conditions\n\n"
                "| Patch | Velocity (U) | Pressure (p) |\n|---|---|---|\n"
                + "\n".join(bc_lines)
            )
            if bc_expl:
                bc_section += f"\n\n{bc_expl}"
            parts.append(bc_section)

    # ── Residuals ─────────────────────────────────────────────────────────────
    if "residuals" in sections and snap.sim_progress:
        last = snap.sim_progress[-1]
        residuals = last.get("residuals", {})
        total_iters = len(snap.sim_progress)
        last_sim_time = last.get("sim_time")

        if residuals:
            rows_list = []
            for f, r in residuals.items():
                if transient:
                    key_residual = r.get("final", "N/A")
                    key_label = f"**{key_residual}** (inner-loop final)"
                else:
                    key_residual = r.get("initial", "N/A")
                    key_label = str(key_residual)
                rows_list.append(
                    f"| {f} | {r.get('initial', 'N/A')} | {key_label} |"
                )
            rows = "\n".join(rows_list)

            if transient and last_sim_time is not None:
                residual_heading = (
                    f"## Solver Residuals — Final Time Step (t = {last_sim_time:.4g} s) "
                    f"— {total_iters} steps total\n\n"
                    "> **Transient simulation**: the inner-loop final residual (shown in bold) "
                    "reflects how accurately the solver resolved each time step. "
                    "The initial residual naturally varies with the physics and is shown for reference.\n\n"
                )
            else:
                residual_heading = (
                    f"## Solver Residuals — Final Iteration ({last.get('iteration', total_iters)}"
                    f" / {total_iters})\n\n"
                )

            residual_expl = (
                "\n\nResiduals are a measure of how well the solver has satisfied the "
                "governing equations in each cell of the mesh. Think of them as the "
                "remaining 'error' in the solution at each step — smaller residuals mean "
                "the equations are being satisfied more precisely. "
                "They are reported on a logarithmic scale, so a value of 1e-6 is far "
                "smaller (better) than 1e-3."
            )

            parts.append(
                residual_heading
                + "| Field | Initial | Final (key metric) |\n|---|---|---|\n"
                + rows
                + residual_expl
            )

        courant = last.get("courant", {})
        if courant:
            parts.append(
                f"**Courant number** — mean: {courant.get('mean', 'N/A')}, "
                f"max: {courant.get('max', 'N/A')}\n\n"
                "The Courant number (also called the CFL number) relates the fluid speed "
                "to the time step size and cell size. Keeping it below 1 ensures the "
                "simulation remains numerically stable in time-stepping solvers."
            )

    # ── Final Results ──────────────────────────────────────────────────────────
    if "results" in sections and snap.final_result:
        fr = snap.final_result
        if isinstance(fr, dict):
            rows = "\n".join(f"| {k} | {v} |" for k, v in fr.items())
            parts.append(f"## Final Results\n\n| Field | Value |\n|---|---|\n{rows}")
        else:
            parts.append(f"## Final Results\n\n```json\n{json.dumps(fr, indent=2, default=str)}\n```")

    report_md = "\n\n---\n\n".join(parts) if parts else "No simulation data available yet."

    # ── Build report_request_payload ─────────────────────────────────────────
    # Tells the frontend which 3-D field views to render + capture as images,
    # which convergence charts to embed, and which key metrics to show.
    # The frontend uses VTK.js renderWindow.captureNextImage() for each field.

    # Colourmap defaults per field (frontend falls back to "turbo" if absent)
    _COLORMAPS: dict[str, str] = {
        "U": "turbo", "p": "turbo", "p_rgh": "turbo",
        "T": "coolwarm", "h": "coolwarm",
        "k": "viridis", "omega": "plasma", "epsilon": "plasma",
        "nut": "plasma", "mut": "plasma", "alphat": "plasma",
        "alpha": "blues",
    }

    vtk_fields: list[dict[str, Any]] = []
    if isinstance(snap.vtk_result, dict):
        vtk_fields = snap.vtk_result.get("fields", []) or []

    # Determine which fields to show based on scope/focus
    scope = "specific" if focus else "full"
    field_screenshots: list[dict[str, Any]] = []
    if vtk_fields:
        # For a specific-field report, filter to the focused field if possible
        display_fields = vtk_fields
        if focus and scope == "specific":
            focused = [f for f in vtk_fields if f.get("name", "").lower() in focus.lower()]
            if focused:
                display_fields = focused

        for fdata in display_fields:
            fname = fdata.get("name") or fdata.get("field") or ""
            if not fname:
                continue
            colormap = _COLORMAPS.get(fname, "turbo")
            field_range = (
                [fdata["min"], fdata["max"]]
                if fdata.get("min") is not None and fdata.get("max") is not None
                else None
            )
            if transient:
                # Show BOTH initial state (first timestep) and final state
                field_screenshots.append({
                    "field_name": fname,
                    "label": f"{fname} — Initial State",
                    "timestep": "first",
                    "colormap": colormap,
                    "range": field_range,
                    "caption": f"{fname} at the start of the simulation (t = first timestep)",
                })
                field_screenshots.append({
                    "field_name": fname,
                    "label": f"{fname} — Final State",
                    "timestep": None,   # null = latest frame
                    "colormap": colormap,
                    "range": field_range,
                    "caption": f"{fname} at the end of the simulation",
                })
            else:
                # Steady-state: only one state (the converged solution)
                field_screenshots.append({
                    "field_name": fname,
                    "label": f"{fname} — Converged Solution",
                    "timestep": None,
                    "colormap": colormap,
                    "range": field_range,
                    "caption": f"Steady-state {fname} distribution",
                })

    # Convergence charts to embed
    convergence_charts: list[dict[str, Any]] = []
    if snap.sim_progress:
        # Always include residuals
        residual_fields = sorted({
            f
            for step in snap.sim_progress
            for f in step.get("residuals", {}).keys()
        })
        chart_fields = residual_fields if not (focus and scope == "specific") else [
            f for f in residual_fields if f.lower().startswith(focus[:2].lower())
        ] or residual_fields
        convergence_charts.append({
            "type": "residuals",
            "fields": chart_fields or None,
            "label": "Residual Convergence",
            "caption": None,
        })
        # Courant number for transient runs
        if transient and any(s.get("courant") for s in snap.sim_progress):
            convergence_charts.append({
                "type": "courant",
                "fields": None,
                "label": "Courant Number",
                "caption": None,
            })
        # Continuity error if present
        if any(s.get("continuity") for s in snap.sim_progress):
            convergence_charts.append({
                "type": "continuity",
                "fields": None,
                "label": "Continuity Error",
                "caption": None,
            })

    # Key scalar metrics for the report header
    metrics: list[dict[str, Any]] = []
    solver_name = snap.solver.get("solver") or snap.physics.get("solver")
    if solver_name:
        metrics.append({"label": "Solver", "value": solver_name, "unit": None})
    if snap.sim_progress:
        last_step = snap.sim_progress[-1]
        if transient:
            last_t = last_step.get("sim_time")
            if last_t is not None:
                metrics.append({"label": "Simulation Time", "value": str(last_t), "unit": "s"})
        metrics.append({"label": "Iterations", "value": str(len(snap.sim_progress)), "unit": None})
        courant = last_step.get("courant", {})
        if courant.get("max") is not None:
            metrics.append({"label": "Final Max Courant", "value": str(courant["max"]), "unit": None})
        last_res = last_step.get("residuals", {})
        for fname, r in last_res.items():
            rv = r.get("final") if transient else r.get("initial")
            if rv is not None:
                metrics.append({"label": f"Final {fname} Residual", "value": f"{rv:.3e}", "unit": None})

    report_request_payload: dict[str, Any] = {
        "title": f"{'Full' if scope == 'full' else focus.title() if focus else 'Full'} Simulation Report",
        "summary": "",   # LLM fills this in its text response; frontend renders from markdown
        "field_screenshots": field_screenshots,
        "convergence_charts": convergence_charts,
        "metrics": metrics,
        "scope": scope,
    }

    return {
        "report_markdown": report_md,
        "sections_included": sections,
        "report_data": {
            "solver": solver_name,
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
        "report_request_payload": report_request_payload,
    }


def analyze_chart(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Analyze a chart / VTK field and provide textual interpretation."""
    chart_type = args.get("chart_type", "residuals")
    field = args.get("field")

    if chart_type == "residuals":
        trend_result = compute_residual_trend({"fields": field}, snap)
        return {
            "chart_type": "residuals",
            "analysis": {k: v for k, v in trend_result.items() if k != "convergence"},
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
                "Return residual history for one or more fields as a chart. "
                "Use this when the user explicitly asks to see residuals or the "
                "residual plot. Do NOT call this proactively."
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
                "Omit 'sections' to auto-select based on focus or include everything. "
                "IMPORTANT: the report must use plain language accessible to users without a "
                "CFD background — explain every choice (solver, turbulence model, BCs) in "
                "plain terms. Do NOT use any emoji. Do NOT make convergence/divergence "
                "conclusions; if residual data is shown, describe the trend only."
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
