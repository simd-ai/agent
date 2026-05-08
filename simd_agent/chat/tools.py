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

from simd_agent.llm.gemini.provider import genai_types as types

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
        all_runs: list[dict[str, Any]] | None = None,
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
        self.all_runs = all_runs or []

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

            # ── Backend convergence assessment ─────────────────────────────
            # Computed by simd_agent/convergence.py and persisted in runs.result
            convergence = None
            if isinstance(self.agent_run, dict):
                result_col = self.agent_run.get("result")
                if isinstance(result_col, dict):
                    convergence = result_col.get("convergence")
            if convergence:
                d["convergence_assessment"] = convergence
        if self.generated_files:
            d["generated_file_paths"] = list(self.generated_files.keys())
        if len(self.all_runs) > 1:
            d["all_runs"] = [
                {
                    "id": r.get("id"),
                    "status": r.get("status"),
                    "solver": r.get("solver"),
                    "label": r.get("label"),
                    "started_at": r.get("started_at"),
                }
                for r in self.all_runs
            ]
            d["total_runs"] = len(self.all_runs)
        # Flag whether field_ranges data is available for field value plotting
        if self.sim_progress and any(
            s.get("field_ranges") or s.get("fieldRanges") for s in self.sim_progress[:20]
        ):
            d["has_field_value_data"] = True
        # Flag whether volume integral data is available
        if self.sim_progress and any(
            s.get("volume_integrals") or s.get("volumeIntegrals") for s in self.sim_progress[:20]
        ):
            d["has_volume_integral_data"] = True
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
                # Handle both formats:
                #   Backend/DB: {"min": 0, "max": 1.5, "mean": 0.7}
                #   Frontend:   {"range": [0, 1.5]}
                mn   = fdata.get("min")
                mx   = fdata.get("max")
                mean = fdata.get("mean")
                # Parse range array if separate min/max not available
                raw_range = fdata.get("range")
                if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
                    if mn is None:
                        mn = raw_range[0]
                    if mx is None:
                        mx = raw_range[1]
                if any(v is not None for v in (mn, mx, mean)):
                    num_components = fdata.get("num_components", 1)
                    suffix = " (magnitude)" if num_components > 1 else ""
                    chart_data = []
                    if mn is not None:
                        chart_data.append({"stat": "min", field: mn})
                    if mean is not None:
                        chart_data.append({"stat": "mean", field: mean})
                    if mx is not None:
                        chart_data.append({"stat": "max", field: mx})
                    return {
                        "field": field,
                        "patch": patch,
                        "sim_time": vtk.get("time"),
                        "min": mn, "max": mx, "mean": mean,
                        "num_components": num_components,
                        "note": f"Spatial min/max over the entire mesh{suffix}.",
                        "source": "vtk_result",
                        "chart": {
                            "type": "bar",
                            "title": f"{field} Statistics{suffix} (t={vtk.get('time')} s)",
                            "xKey": "stat",
                            "yLabel": field,
                            "lines": [field],
                            "data": chart_data,
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
    "pimplefoam", "pisofoam",
    "rhopimplefoam", "buoyantpimplefoam",
}

_STEADY_SOLVERS = {
    "simplefoam", "rhosimplefoam", "buoyantsimplefoam",
    "laplacianfoam", "potentialfoam", "scalarTransportFoam",
}


def _is_transient(snap: SimulationSnapshot) -> bool:
    """Return True if this is a time-marching (transient) simulation."""
    # Try to get the solver name from multiple sources
    solver = (
        snap.solver.get("solver")
        or snap.physics.get("solver")
        or snap.simulation_config.get("solver", {}).get("solver")
        or ""
    ).lower()

    # Also check authoritative sources: final_result, agent_run
    if not solver:
        fr = snap.final_result if isinstance(snap.final_result, dict) else {}
        solver = (fr.get("solver") or "").lower()
    if not solver and snap.agent_run:
        solver = (snap.agent_run.get("solver") or "").lower()

    # Known transient solvers
    if solver in _TRANSIENT_SOLVERS:
        return True

    # Known steady-state solvers — return False definitively
    if solver in _STEADY_SOLVERS:
        return False

    time_scheme = (snap.physics.get("timeScheme") or "").lower()
    if time_scheme in ("transient", "unsteady"):
        return True
    if time_scheme in ("steady", "steadystate", "steady-state"):
        return False

    # Fallback for unknown solver: check sim_progress for non-integer time
    # progression (steady solvers use integer pseudo-time = iteration count,
    # transient solvers use fractional real time like 0.001, 0.002, ...).
    if snap.sim_progress and len(snap.sim_progress) > 1:
        times = [
            s.get("sim_time") for s in snap.sim_progress[:5]
            if s.get("sim_time") is not None
        ]
        if len(times) >= 2 and times[-1] != times[0]:
            # Check if times are integer-like (steady pseudo-time)
            all_integer = all(
                isinstance(t, int) or (isinstance(t, float) and t == int(t))
                for t in times
            )
            if not all_integer:
                return True
    return False


def compute_residual_trend(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Return residual history series for the requested field(s).

    X-axis is simulation time (sim_time) when available — matching the
    /api/runs/{id}/timesteps format — with iteration as fallback.

    For transient solvers (pimpleFoam, rhoPimpleFoam…): shows inner-loop FINAL
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
        out: dict[str, Any] = {"description": description, "result": result}
        # If the code built a chart spec, surface it so service.py emits an artifact
        if isinstance(result, dict) and "chart" in result:
            out["chart"] = result["chart"]
        return out
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
    if "buoyantsimplefoam" in s:
        return (
            "The solver chosen is **buoyantSimpleFoam**. "
            "This is a steady-state solver for buoyancy-driven (natural convection) flow. "
            "It accounts for density variations due to temperature, enabling it to model "
            "heat-driven circulation, HVAC systems, and gravity-influenced thermal flows."
        )
    if "buoyantpimplefoam" in s:
        return (
            "The solver chosen is **buoyantPimpleFoam**. "
            "This is a time-stepping solver for buoyancy-driven flow. It captures how "
            "temperature-driven density differences create and evolve flow over time — "
            "suitable for transient natural convection, thermal stratification, and "
            "time-varying heat source scenarios."
        )
    if "pisofoam" in s:
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


# ---------------------------------------------------------------------------
# Helpers for extracting validated_config fields
# ---------------------------------------------------------------------------

def _extract_validated_config(snap: SimulationSnapshot) -> dict[str, Any]:
    """Return the validated_config dict from final_result (authoritative source).
    Falls back to empty dict if not present."""
    fr = snap.final_result
    if isinstance(fr, dict):
        vc = fr.get("validated_config")
        if isinstance(vc, dict):
            return vc
    return {}


def _vc_solver_name(vc: dict[str, Any], snap: SimulationSnapshot) -> str:
    """Get OpenFOAM solver name (e.g. 'simpleFoam') from the most authoritative source."""
    # 1. final_result.solver — set by Orchestrator at run completion (most reliable)
    fr = snap.final_result
    if isinstance(fr, dict):
        name = fr.get("solver")
        if name:
            return str(name)
    # 2. agent_run.solver — DB column on the runs table
    if snap.agent_run:
        name = snap.agent_run.get("solver")
        if name:
            return str(name)
        # 2b. agent_run.result.solver — stored by finalize_run
        run_result = snap.agent_run.get("result")
        if isinstance(run_result, dict):
            name = run_result.get("solver")
            if name:
                return str(name)
    # 3. lint_result.selected_solver — set by SolverSelector during linting
    if snap.lint_result:
        name = snap.lint_result.get("selected_solver") or snap.lint_result.get("solver")
        if name:
            return str(name)
    # 4. validated_config.solver.type (SolverV1 serialized)
    vc_solver = vc.get("solver") or {}
    if isinstance(vc_solver, dict):
        name = vc_solver.get("type") or vc_solver.get("solver_type")
        if name:
            return str(name)
    # 5. fallback to frontend context
    return snap.solver.get("solver") or snap.physics.get("solver") or "N/A"


def _vc_turbulence_model(vc: dict[str, Any], snap: SimulationSnapshot) -> str:
    """Get turbulence model from validated_config (authoritative)."""
    # validated_config.turbulence.model
    vc_turb = vc.get("turbulence") or {}
    if isinstance(vc_turb, dict):
        m = vc_turb.get("model")
        if m:
            return str(m)
    # validated_config.physics.turbulence_model
    vc_phys = vc.get("physics") or {}
    if isinstance(vc_phys, dict):
        m = vc_phys.get("turbulence_model")
        if m:
            return str(m)
    # fallback to frontend context
    return snap.turbulence.get("model") or "N/A"


def _vc_flow_regime(vc: dict[str, Any], snap: SimulationSnapshot) -> str:
    vc_phys = vc.get("physics") or {}
    if isinstance(vc_phys, dict):
        r = vc_phys.get("flow_regime") or vc_phys.get("flowRegime") or vc_phys.get("flowType")
        if r:
            return str(r)
    return snap.physics.get("flowType") or snap.physics.get("flowRegime") or "N/A"


def _governing_equations_section(solver_name: str, transient: bool,
                                  compressible: bool, energy: bool,
                                  turb_model: str) -> str:
    """Return markdown for the Governing Equations section.

    Uses LaTeX notation wrapped in $...$ (inline) and $$...$$ (display)
    for proper mathematical rendering via @react-pdf/math in the PDF.
    """
    lines: list[str] = ["## Governing Equations\n"]

    # --- Continuity ---
    if compressible:
        lines.append("**Continuity (mass conservation):**\n")
        if transient:
            lines.append("$$\\frac{\\partial \\rho}{\\partial t} + \\nabla \\cdot (\\rho \\mathbf{U}) = 0$$\n")
        else:
            lines.append("$$\\nabla \\cdot (\\rho \\mathbf{U}) = 0$$\n")
    else:
        lines.append("**Continuity (mass conservation):**\n")
        lines.append("$$\\nabla \\cdot \\mathbf{U} = 0$$\n")

    # --- Momentum ---
    lines.append("**Momentum (Reynolds-Averaged Navier-Stokes):**\n")
    if compressible:
        if transient:
            lines.append(
                "$$\\frac{\\partial (\\rho \\mathbf{U})}{\\partial t} + \\nabla \\cdot (\\rho \\mathbf{U} \\otimes \\mathbf{U}) "
                "= -\\nabla p + \\nabla \\cdot (\\tau + \\tau_t) + \\rho \\mathbf{g}$$\n"
            )
        else:
            lines.append(
                "$$\\nabla \\cdot (\\rho \\mathbf{U} \\otimes \\mathbf{U}) "
                "= -\\nabla p + \\nabla \\cdot (\\tau + \\tau_t) + \\rho \\mathbf{g}$$\n"
            )
    else:
        if transient:
            lines.append(
                "$$\\frac{\\partial \\mathbf{U}}{\\partial t} + \\nabla \\cdot (\\mathbf{U} \\otimes \\mathbf{U}) "
                "= -\\frac{\\nabla p}{\\rho} + \\nabla \\cdot (\\nu_{eff} \\nabla \\mathbf{U})$$\n"
            )
        else:
            lines.append(
                "$$\\nabla \\cdot (\\mathbf{U} \\otimes \\mathbf{U}) "
                "= -\\frac{\\nabla p}{\\rho} + \\nabla \\cdot (\\nu_{eff} \\nabla \\mathbf{U})$$\n"
            )
    lines.append(
        "where ν_eff = ν + ν_t (molecular + turbulent viscosity) "
        "and τ is the viscous stress tensor.\n"
    )

    # --- Energy ---
    if energy:
        lines.append("**Energy (enthalpy transport):**\n")
        if compressible:
            if transient:
                lines.append(
                    "$$\\frac{\\partial (\\rho h)}{\\partial t} + \\nabla \\cdot (\\rho \\mathbf{U} h) "
                    "= \\nabla \\cdot (\\alpha_{eff} \\nabla h) + \\frac{\\partial p}{\\partial t}$$\n"
                )
            else:
                lines.append(
                    "$$\\nabla \\cdot (\\rho \\mathbf{U} h) = \\nabla \\cdot (\\alpha_{eff} \\nabla h)$$\n"
                )
        else:
            if transient:
                lines.append(
                    "$$\\frac{\\partial T}{\\partial t} + \\nabla \\cdot (\\mathbf{U} T) "
                    "= \\nabla \\cdot (\\alpha_{eff} \\nabla T)$$\n"
                )
            else:
                lines.append(
                    "$$\\nabla \\cdot (\\mathbf{U} T) = \\nabla \\cdot (\\alpha_{eff} \\nabla T)$$\n"
                )
        lines.append(
            "where α_eff = α + α_t (molecular + turbulent thermal diffusivity).\n"
        )

    # --- Turbulence transport ---
    m = (turb_model or "").lower()
    if "komegasst" in m or "k-omega" in m or "komega" in m:
        lines.append("**Turbulence — k-ω SST:**\n")
        lines.append("Turbulent kinetic energy (k):\n")
        lines.append(
            "$$\\frac{\\partial (\\rho k)}{\\partial t} + \\nabla \\cdot (\\rho \\mathbf{U} k) "
            "= P_k - \\beta^* \\rho k \\omega + \\nabla \\cdot \\left[(\\mu + \\sigma_k \\mu_t) \\nabla k\\right]$$\n"
        )
        lines.append("Specific dissipation rate (ω):\n")
        lines.append(
            "$$\\frac{\\partial (\\rho \\omega)}{\\partial t} + \\nabla \\cdot (\\rho \\mathbf{U} \\omega) "
            "= \\gamma P_\\omega - \\beta \\rho \\omega^2 + \\nabla \\cdot \\left[(\\mu + \\sigma_\\omega \\mu_t) "
            "\\nabla \\omega\\right] + CD_{k\\omega}$$\n"
        )
        lines.append(
            "where P_k is the production of k, CD_kω is the cross-diffusion term "
            "that blends between k-ω (near wall) and k-ε (free stream), and "
            "μ_t = ρa₁k / max(a₁ω, SF₂).\n"
        )
    elif "kepsilon" in m or "k-epsilon" in m:
        lines.append("**Turbulence — k-ε:**\n")
        lines.append("Turbulent kinetic energy (k):\n")
        lines.append(
            "$$\\frac{\\partial (\\rho k)}{\\partial t} + \\nabla \\cdot (\\rho \\mathbf{U} k) "
            "= P_k - \\rho \\varepsilon + \\nabla \\cdot \\left[(\\mu + \\mu_t / \\sigma_k) \\nabla k\\right]$$\n"
        )
        lines.append("Dissipation rate (ε):\n")
        lines.append(
            "$$\\frac{\\partial (\\rho \\varepsilon)}{\\partial t} + \\nabla \\cdot (\\rho \\mathbf{U} \\varepsilon) "
            "= C_{1\\varepsilon} \\frac{\\varepsilon}{k} P_k - C_{2\\varepsilon} \\rho \\frac{\\varepsilon^2}{k} "
            "+ \\nabla \\cdot \\left[(\\mu + \\mu_t / \\sigma_\\varepsilon) \\nabla \\varepsilon\\right]$$\n"
        )
        lines.append("where μ_t = ρC_μk²/ε.\n")
    elif "spalart" in m:
        lines.append("**Turbulence — Spalart-Allmaras:**\n")
        lines.append(
            "$$\\frac{\\partial \\tilde{\\nu}}{\\partial t} + \\mathbf{U} \\cdot \\nabla \\tilde{\\nu} "
            "= C_{b1} \\tilde{S} \\tilde{\\nu} - C_{w1} f_w \\left(\\frac{\\tilde{\\nu}}{d}\\right)^2 "
            "+ \\frac{1}{\\sigma} \\nabla \\cdot \\left[(\\nu + \\tilde{\\nu}) \\nabla \\tilde{\\nu}\\right] "
            "+ C_{b2} |\\nabla \\tilde{\\nu}|^2$$\n"
        )
        lines.append("where ν_t = ν̃·f_v1 and d is the distance to the nearest wall.\n")

    return "\n".join(lines)


def _turbulence_details_section(turb_model: str, snap: SimulationSnapshot) -> str:
    """Return markdown for the Turbulence Model Details section."""
    m = (turb_model or "").lower()
    lines: list[str] = ["## Turbulence Model\n"]

    if "komegasst" in m or "k-omega" in m or "komega" in m:
        lines.append(f"**Model:** k-ω SST (Shear Stress Transport)\n")
        lines.append("### Model Constants\n")
        lines.append("| Constant | Value | Description |")
        lines.append("|---|---|---|")
        lines.append("| α₁ (a1) | 0.31 | Shear-stress limiter coefficient |")
        lines.append("| β* | 0.09 | Turbulent viscosity coefficient (C_μ) |")
        lines.append("| β₁ | 0.075 | k-ω destruction coefficient (inner) |")
        lines.append("| β₂ | 0.0828 | k-ε destruction coefficient (outer) |")
        lines.append("| σ_k1 | 0.85 | k diffusion coefficient (inner) |")
        lines.append("| σ_k2 | 1.0 | k diffusion coefficient (outer) |")
        lines.append("| σ_ω1 | 0.5 | ω diffusion coefficient (inner) |")
        lines.append("| σ_ω2 | 0.856 | ω diffusion coefficient (outer) |")
        lines.append("")
        lines.append("### Wall Treatment\n")
        lines.append(
            "- **omegaWallFunction** — blended wall function for ω, automatically "
            "switches between viscous sublayer and log-law formulation\n"
            "- **kqRWallFunction** — zero-gradient for k at walls (production = dissipation)\n"
            "- **nutUSpaldingWallFunction** — continuous wall function for ν_t based on "
            "Spalding's law, valid across all y+ values"
        )
    elif "kepsilon" in m or "k-epsilon" in m:
        label = "Realizable k-ε" if "realizable" in m else "Standard k-ε"
        lines.append(f"**Model:** {label}\n")
        lines.append("### Model Constants\n")
        lines.append("| Constant | Value | Description |")
        lines.append("|---|---|---|")
        lines.append("| C_μ | 0.09 | Turbulent viscosity coefficient |")
        lines.append("| C₁ε | 1.44 | Production coefficient |")
        lines.append("| C₂ε | 1.92 | Destruction coefficient |")
        lines.append("| σ_k | 1.0 | k diffusion Prandtl number |")
        lines.append("| σ_ε | 1.3 | ε diffusion Prandtl number |")
        lines.append("")
        lines.append("### Wall Treatment\n")
        lines.append(
            "- **epsilonWallFunction** — equilibrium wall function for ε\n"
            "- **kqRWallFunction** — zero-gradient for k at walls\n"
            "- **nutkWallFunction** — standard log-law wall function for ν_t"
        )
    elif "spalart" in m:
        lines.append("**Model:** Spalart-Allmaras (one-equation)\n")
        lines.append("### Model Constants\n")
        lines.append("| Constant | Value |")
        lines.append("|---|---|")
        lines.append("| σ | 2/3 |")
        lines.append("| C_b1 | 0.1355 |")
        lines.append("| C_b2 | 0.622 |")
        lines.append("| C_w2 | 0.3 |")
        lines.append("| C_w3 | 2.0 |")
        lines.append("| C_v1 | 7.1 |")
        lines.append("")
        lines.append("### Wall Treatment\n")
        lines.append("- **nutUSpaldingWallFunction** — continuous wall function for ν_t")
    elif "laminar" in m or not m:
        lines.append(
            "**Model:** Laminar (no turbulence model)\n\n"
            "The flow is treated as laminar — no additional transport equations for "
            "turbulence quantities are solved."
        )
        return "\n".join(lines)
    else:
        lines.append(f"**Model:** {turb_model}\n")
        return "\n".join(lines)

    # Add turbulence IC values if available
    turb_data = snap.turbulence
    if turb_data:
        ic_rows = []
        for key, label, unit in [
            ("k", "k (turbulent kinetic energy)", "m²/s²"),
            ("omega", "ω (specific dissipation rate)", "1/s"),
            ("epsilon", "ε (dissipation rate)", "m²/s³"),
            ("nut", "ν_t (turbulent viscosity)", "m²/s"),
            ("nuTilda", "ν_tilda (modified viscosity)", "m²/s"),
        ]:
            val = turb_data.get(key)
            if val is not None:
                ic_rows.append(f"| {label} | {val} | {unit} |")
        if ic_rows:
            lines.append("\n\n### Turbulence Inlet Values\n")
            lines.append("| Quantity | Value | Unit |")
            lines.append("|---|---|---|")
            lines.extend(ic_rows)

    return "\n".join(lines)


_OF_TERM_LABELS: dict[str, str] = {
    # Divergence terms
    "div(phi,U)":       "Convection of velocity (U)",
    "div(phi,k)":       "Convection of k",
    "div(phi,omega)":   "Convection of ω",
    "div(phi,epsilon)": "Convection of ε",
    "div(phi,nuTilda)": "Convection of ν_tilda",
    "div(phi,h)":       "Convection of enthalpy (h)",
    "div(phi,e)":       "Convection of internal energy (e)",
    "div(phi,T)":       "Convection of temperature (T)",
    "div(phi,K)":       "Convection of kinetic energy (K)",
    "div(phi,Ekp)":     "Convection of Ekp (energy)",
    "div(phid,p)":      "Pressure-density convection",
    "div(phi,alpha)":   "Convection of volume fraction (α)",
    "div(((rho*nuEff)*dev2(T(grad(U)))))": "Viscous stress divergence",
    "div((nuEff*dev2(T(grad(U)))))":       "Viscous stress divergence",
    "div(phi,p)":       "Pressure transport",
    # Gradient terms
    "grad(U)":   "Gradient of velocity",
    "grad(p)":   "Gradient of pressure",
    "grad(k)":   "Gradient of k",
    "grad(omega)": "Gradient of ω",
    "grad(epsilon)": "Gradient of ε",
    "grad(T)":   "Gradient of temperature",
    # Time terms
    "default":   "Default",
    # Laplacian terms
    "laplacian(nuEff,U)":    "Viscous diffusion of velocity",
    "laplacian(DkEff,k)":    "Diffusion of k",
    "laplacian(DomegaEff,omega)": "Diffusion of ω",
    "laplacian(DepsilonEff,epsilon)": "Diffusion of ε",
    "laplacian(DnuTildaEff,nuTilda)": "Diffusion of ν_tilda",
    "laplacian(alphaEff,h)": "Thermal diffusion of enthalpy",
    "laplacian(alphaEff,e)": "Thermal diffusion of internal energy",
    "laplacian(rAUf,p)":     "Pressure correction diffusion",
    "laplacian((1|A(U)),p)": "Pressure correction diffusion",
}


def _humanize_of_term(term: str) -> str:
    """Map an OpenFOAM fvSchemes term to a human-readable description."""
    if term in _OF_TERM_LABELS:
        return _OF_TERM_LABELS[term]
    # Try a relaxed match: strip outer whitespace, check for common patterns
    t = term.strip()
    if t.startswith("div(phi,"):
        field = t.removeprefix("div(phi,").rstrip(")")
        return f"Convection of {field}"
    if t.startswith("div("):
        return f"Divergence term: {t}"
    if t.startswith("grad("):
        field = t.removeprefix("grad(").rstrip(")")
        return f"Gradient of {field}"
    if t.startswith("laplacian("):
        return f"Diffusion term: {t}"
    if t.startswith("snGrad("):
        return f"Surface-normal gradient: {t}"
    return term


def _discretization_section(snap: SimulationSnapshot) -> str:
    """Extract fvSchemes from generated files and present discretization schemes."""
    import re as _re

    fv_schemes_content = snap.generated_files.get("system/fvSchemes", "")
    if not fv_schemes_content:
        return ""

    lines: list[str] = ["## Numerical Discretization (fvSchemes)\n"]

    # Parse major blocks from fvSchemes
    _BLOCKS = [
        ("ddtSchemes", "Time Discretization"),
        ("gradSchemes", "Gradient Schemes"),
        ("divSchemes", "Divergence Schemes"),
        ("laplacianSchemes", "Laplacian Schemes"),
        ("interpolationSchemes", "Interpolation Schemes"),
        ("snGradSchemes", "Surface-Normal Gradient Schemes"),
    ]

    for block_name, heading in _BLOCKS:
        # Find the block content between braces
        pattern = _re.compile(
            rf'{block_name}\s*\{{([^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*)\}}',
            _re.DOTALL
        )
        match = pattern.search(fv_schemes_content)
        if not match:
            continue
        block_content = match.group(1).strip()
        if not block_content:
            continue

        # Parse entries: "key  value;" lines
        entries = []
        for line in block_content.splitlines():
            line = line.strip()
            if not line or line.startswith("//") or line == "default":
                continue
            # Remove trailing semicolons and comments
            line = _re.sub(r'//.*$', '', line).strip().rstrip(';').strip()
            if not line:
                continue
            # Split into field and scheme
            parts = line.split(None, 1)
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))
            elif len(parts) == 1:
                entries.append(("default", parts[0]))

        if entries:
            lines.append(f"\n**{heading}:**\n")
            lines.append("| Term | Description | Scheme |")
            lines.append("|---|---|---|")
            for term, scheme in entries:
                desc = _humanize_of_term(term)
                lines.append(f"| {term} | {desc} | {scheme} |")

    if len(lines) <= 1:
        return ""

    lines.append(
        "\n\nThe divergence scheme controls numerical diffusion — **upwind** is robust "
        "but diffusive, **linearUpwind** is second-order accurate, and **bounded** variants "
        "prevent overshoots. The time scheme (Euler = first-order, backward = second-order) "
        "affects temporal accuracy in transient simulations."
    )

    return "\n".join(lines)


def _initial_conditions_section(snap: SimulationSnapshot) -> str:
    """Extract internalField values from generated 0/ files."""
    import re as _re

    ic_rows: list[str] = []
    _FIELD_UNITS: dict[str, str] = {
        "U": "m/s", "p": "Pa", "p_rgh": "Pa", "T": "K",
        "k": "m²/s²", "omega": "1/s", "epsilon": "m²/s³",
        "nut": "m²/s", "alphat": "kg/m·s", "nuTilda": "m²/s",
    }

    for filepath, content in sorted(snap.generated_files.items()):
        if not filepath.startswith("0/"):
            continue
        field_name = filepath.split("/", 1)[1]
        if field_name.startswith("."):
            continue

        # Extract internalField value
        match = _re.search(
            r'internalField\s+(uniform\s+)?(\([^)]+\)|[^\s;]+)',
            content
        )
        if not match:
            continue
        value = match.group(0).replace("internalField", "").strip().rstrip(";").strip()
        unit = _FIELD_UNITS.get(field_name, "")
        ic_rows.append(f"| {field_name} | {value} | {unit} |")

    if not ic_rows:
        return ""

    lines = [
        "## Initial Conditions\n",
        "The following initial field values (internalField) are applied throughout "
        "the computational domain at the start of the simulation:\n",
        "| Field | Value | Unit |",
        "|---|---|---|",
    ]
    lines.extend(ic_rows)
    return "\n".join(lines)


def _transport_properties_section(snap: SimulationSnapshot,
                                   vc: dict[str, Any]) -> str:
    """Build a transport / thermophysical properties section."""
    lines: list[str] = []

    # Get fluid data from validated config or frontend context
    vc_fluid = vc.get("fluid") or {}
    fluid = vc_fluid if vc_fluid else snap.fluid

    if not fluid:
        return ""

    _FLUID_LABELS: dict[str, str] = {
        "name": "Fluid name",
        "density": "Density ρ",
        "kinematic_viscosity": "Kinematic viscosity ν",
        "dynamic_viscosity": "Dynamic viscosity μ",
        "specific_heat": "Specific heat C_p",
        "thermal_conductivity": "Thermal conductivity λ",
        "prandtl_number": "Prandtl number Pr",
        "temperature": "Reference temperature T_ref",
    }
    _FLUID_UNITS: dict[str, str] = {
        "density": "kg/m³",
        "kinematic_viscosity": "m²/s",
        "dynamic_viscosity": "Pa·s",
        "specific_heat": "J/(kg·K)",
        "thermal_conductivity": "W/(m·K)",
        "prandtl_number": "—",
        "temperature": "K",
    }

    rows = []
    for key, label in _FLUID_LABELS.items():
        val = fluid.get(key)
        if val is None or val == "" or val == []:
            continue
        unit = _FLUID_UNITS.get(key, "")
        rows.append(f"| {label} | {val} | {unit} |")

    if not rows:
        return ""

    lines = [
        "## Transport Properties\n",
        "| Property | Value | Unit |",
        "|---|---|---|",
    ]
    lines.extend(rows)

    # Check if we have transportProperties or thermophysicalProperties file
    tp_content = snap.generated_files.get("constant/transportProperties", "")
    thermo_content = snap.generated_files.get("constant/thermophysicalProperties", "")

    if tp_content:
        lines.append(
            "\nFluid transport is modelled using **transportProperties** "
            "(incompressible Newtonian model with constant kinematic viscosity ν)."
        )
    elif thermo_content:
        lines.append(
            "\nFluid properties are defined in **thermophysicalProperties** "
            "(compressible model with equation of state and thermodynamic properties)."
        )

    return "\n".join(lines)


def _enhanced_boundary_conditions_section(snap: SimulationSnapshot,
                                           vc: dict[str, Any]) -> str:
    """Build an enhanced BC section with actual field values."""
    vc_bcs = vc.get("boundary_conditions") or {}
    patches = vc_bcs if vc_bcs else snap.patches
    if not patches:
        return ""

    lines: list[str] = ["## Boundary Conditions\n"]

    for pname, pdata in patches.items():
        if not isinstance(pdata, dict):
            continue
        patch_type = pdata.get("patch_type") or pdata.get("patchType") or "patch"
        # Infer actual OpenFOAM type from BC content when the raw type is just "patch"
        if patch_type == "patch":
            vel_bc = pdata.get("velocity") or pdata.get("U") or {}
            if isinstance(vel_bc, dict):
                bc_t = (vel_bc.get("type") or "").lower()
                if bc_t in ("noslip", "fixedvalue") and vel_bc.get("value") in (None, "0", "(0 0 0)", [0,0,0]):
                    patch_type = "wall"
                elif "wall" in pname.lower():
                    patch_type = "wall"
            elif "wall" in pname.lower():
                patch_type = "wall"
        lines.append(f"### {pname} ({patch_type})\n")

        bc_rows: list[str] = []
        # Iterate over all field BCs for this patch
        _FIELD_ORDER = ["velocity", "U", "pressure", "p", "temperature", "T",
                        "k", "omega", "epsilon", "nut", "alphat", "nuTilda"]
        _FIELD_LABELS: dict[str, str] = {
            "velocity": "Velocity (U)", "U": "Velocity (U)",
            "pressure": "Pressure (p)", "p": "Pressure (p)",
            "temperature": "Temperature (T)", "T": "Temperature (T)",
            "k": "Turb. kinetic energy (k)",
            "omega": "Specific dissipation (ω)",
            "epsilon": "Dissipation rate (ε)",
            "nut": "Turb. viscosity (ν_t)",
            "alphat": "Turb. thermal diff. (α_t)",
            "nuTilda": "Modified viscosity (ν̃)",
        }

        seen_fields: set[str] = set()
        for field_key in _FIELD_ORDER:
            if field_key in seen_fields:
                continue
            bc = pdata.get(field_key)
            if not isinstance(bc, dict):
                continue
            seen_fields.add(field_key)

            bc_type = bc.get("type", "—")
            value = bc.get("value")
            label = _FIELD_LABELS.get(field_key, field_key)

            if value is not None:
                bc_rows.append(f"| {label} | {bc_type} | {value} |")
            else:
                bc_rows.append(f"| {label} | {bc_type} | — |")

        # Also check for any fields not in the standard order
        for field_key, bc in pdata.items():
            if field_key in seen_fields or field_key in ("patch_type", "patchType"):
                continue
            if not isinstance(bc, dict):
                continue
            bc_type = bc.get("type", "—")
            value = bc.get("value")
            label = _FIELD_LABELS.get(field_key, field_key)
            if value is not None:
                bc_rows.append(f"| {label} | {bc_type} | {value} |")
            else:
                bc_rows.append(f"| {label} | {bc_type} | — |")

        if bc_rows:
            lines.append("| Field | BC Type | Value |")
            lines.append("|---|---|---|")
            lines.extend(bc_rows)
            lines.append("")

    # Add plain-language explanation
    bc_expl = _plain_bc_explanation(patches)
    if bc_expl:
        lines.append(bc_expl)

    return "\n".join(lines)


def _problem_definition_section(snap: SimulationSnapshot, vc: dict[str, Any],
                                 solver_name: str, regime: str,
                                 re_num: Any, turb_model: str,
                                 transient: bool) -> str:
    """Build a Problem Definition section describing the simulation setup."""
    lines: list[str] = ["## Problem Definition\n"]

    # Geometry info from mesh
    mi = vc.get("mesh") or snap.mesh_info or {}
    cells = mi.get("cells") or mi.get("numCells")

    # Infer geometry type from patch names
    patches = vc.get("boundary_conditions") or snap.patches or {}
    patch_names = [n.lower() for n in patches.keys()]

    geom_hints: list[str] = []
    if any("inlet" in n for n in patch_names):
        geom_hints.append("inlet")
    if any("outlet" in n for n in patch_names):
        geom_hints.append("outlet")
    if any("wall" in n for n in patch_names):
        geom_hints.append("wall")
    if any("symmetry" in n for n in patch_names):
        geom_hints.append("symmetry plane")

    geom_desc = ""
    if "inlet" in geom_hints and "outlet" in geom_hints:
        geom_desc = "internal flow domain (inlet → outlet)"
    elif "inlet" in geom_hints:
        geom_desc = "flow domain with defined inlet"
    elif "wall" in geom_hints:
        geom_desc = "enclosed domain with wall boundaries"

    # Physics flags
    vc_phys = vc.get("physics") or snap.physics or {}
    heat_transfer = vc_phys.get("heat_transfer", False)
    compressible = vc_phys.get("compressibility", "").lower() in ("compressible", "true") \
        if isinstance(vc_phys.get("compressibility"), str) \
        else bool(vc_phys.get("compressibility"))
    gravity = vc_phys.get("gravity", False)

    # Build description
    desc_parts: list[str] = []
    if geom_desc:
        desc_parts.append(f"This simulation models flow through a **{geom_desc}**")
    else:
        desc_parts.append("This simulation models fluid flow in the given domain")

    if cells:
        desc_parts.append(f"discretized into **{cells:,}** computational cells" if isinstance(cells, int) else f"discretized into **{cells}** computational cells")

    lines.append(". ".join(desc_parts) + ".\n")

    # Flow characteristics
    char_parts: list[str] = []
    char_parts.append(f"- **Flow regime:** {regime}")
    try:
        re_val = float(re_num)
        char_parts.append(f"- **Reynolds number:** {re_val:,.0f}")
    except (TypeError, ValueError):
        if re_num and re_num != "N/A":
            char_parts.append(f"- **Reynolds number:** {re_num}")

    char_parts.append(f"- **Time treatment:** {'Transient (time-dependent)' if transient else 'Steady-state'}")
    if compressible:
        char_parts.append("- **Compressibility:** Compressible (density varies with pressure/temperature)")
    else:
        char_parts.append("- **Compressibility:** Incompressible (constant density)")
    if heat_transfer:
        char_parts.append("- **Heat transfer:** Enabled (energy equation solved)")
    if gravity:
        char_parts.append("- **Buoyancy:** Enabled (gravity-driven density effects)")

    char_parts.append(f"- **Solver:** **{solver_name}**")
    char_parts.append(f"- **Turbulence model:** {turb_model}")

    lines.append("\n".join(char_parts))

    return "\n".join(lines)


def _build_standard_report(
    *,
    snap: "SimulationSnapshot",
    vc: dict[str, Any],
    solver_name: str,
    turb_model: str,
    regime: str,
    re_num: Any,
    transient: bool,
    compressible: bool,
    energy: bool,
    conv: dict[str, Any] | None,
    bg_paragraph: str,
    run_status: str,
    total_iters: int,
    duration_s: float | None,
) -> str:
    """Build a simplified simulation report for non-CFD engineers.

    Focuses on: what was simulated, did it work, key results, actionable insights.
    Avoids: governing equations, numerical schemes, residual tables, mesh quality metrics.
    """
    parts: list[str] = []
    iter_label = "time steps" if transient else "iterations"
    time_str = ""
    if duration_s:
        time_str = (
            f" over {duration_s / 60:.1f} minutes"
            if duration_s > 60
            else f" in {duration_s:.0f} seconds"
        )

    # ── 1. SIMULATION OVERVIEW ──────────────────────────────────────────────
    overview: list[str] = ["## Simulation Overview\n"]
    if bg_paragraph:
        overview.append(f"{bg_paragraph}\n")
    else:
        flow_desc = "time-varying" if transient else "steady-state"
        comp_desc = "compressible" if compressible else "incompressible"
        overview.append(
            f"A {flow_desc} {comp_desc} fluid flow simulation was performed "
            f"using the **{solver_name}** solver"
            f"{' with heat transfer analysis' if energy else ''}.\n"
        )
    parts.append("\n".join(overview))

    # ── 2. SETUP AT A GLANCE ───────────────────────────────────────────────
    setup: list[str] = ["## Setup at a Glance\n"]
    setup.append("| Parameter | Value |")
    setup.append("|---|---|")
    setup.append(f"| Solver | {solver_name} |")
    flow_type = (
        f"{'Time-varying' if transient else 'Steady-state'}, "
        f"{'compressible' if compressible else 'incompressible'}"
    )
    setup.append(f"| Flow Type | {flow_type} |")
    setup.append(f"| Turbulence | {turb_model or 'Laminar (no turbulence model)'} |")
    if re_num and re_num != "N/A":
        try:
            setup.append(f"| Reynolds Number | {float(re_num):,.0f} |")
        except (TypeError, ValueError):
            setup.append(f"| Reynolds Number | {re_num} |")

    vc_mesh = vc.get("mesh") or {}
    cm = vc_mesh.get("check_mesh") or {}
    cells = (
        vc_mesh.get("cells")
        or cm.get("cells")
        or (snap.mesh_info or {}).get("cells")
        or (snap.mesh_info or {}).get("numCells")
    )
    if cells:
        try:
            setup.append(f"| Mesh Size | {int(cells):,} cells |")
        except (TypeError, ValueError):
            setup.append(f"| Mesh Size | {cells} cells |")

    if total_iters:
        setup.append(
            f"| {'Time Steps' if transient else 'Iterations'} | {total_iters:,} |"
        )
    if duration_s:
        if duration_s < 60:
            setup.append(f"| Compute Time | {duration_s:.0f} seconds |")
        else:
            setup.append(f"| Compute Time | {duration_s / 60:.1f} minutes |")

    parts.append("\n".join(setup))

    # ── 3. HOW THE SIMULATION WENT ─────────────────────────────────────────
    outcome: list[str] = ["## How the Simulation Went\n"]
    if conv:
        conv_st = conv.get("status", "unknown")
        if conv_st == "converged":
            outcome.append(
                f"**Result: Converged**\n\n"
                f"The simulation ran for {total_iters:,} {iter_label}{time_str} and "
                f"reached a stable solution. All field equations are well-balanced, "
                f"meaning the results are reliable for engineering decisions.\n"
            )
        elif conv_st == "converging":
            outcome.append(
                f"**Result: Still Converging**\n\n"
                f"The simulation ran for {total_iters:,} {iter_label}{time_str} and is "
                f"trending toward a stable solution, but hasn't fully settled. The results "
                f"are directionally correct but may shift with more iterations. Consider "
                f"extending the run.\n"
            )
        elif conv_st == "oscillating":
            if transient:
                outcome.append(
                    f"**Result: Oscillating**\n\n"
                    f"The simulation ran for {total_iters:,} {iter_label}{time_str}. The "
                    f"solution is fluctuating, which can be expected in time-varying flows "
                    f"(e.g. vortex shedding). Check if these oscillations match the expected "
                    f"physical behavior.\n"
                )
            else:
                outcome.append(
                    f"**Result: Oscillating**\n\n"
                    f"The simulation ran for {total_iters:,} {iter_label}{time_str} but the "
                    f"solution is fluctuating instead of settling. This often means the flow "
                    f"is inherently unsteady or the solver settings need adjustment.\n"
                )
        elif conv_st == "diverging":
            outcome.append(
                "**Result: Diverging**\n\n"
                "The simulation is not producing stable results. This usually indicates "
                "an issue with the setup (boundary values, mesh quality, or time step). "
                "The current results should not be used for decisions.\n"
            )
        elif conv_st == "stalling":
            outcome.append(
                f"**Result: Stalling**\n\n"
                f"The simulation stopped making progress after {total_iters:,} {iter_label}. "
                f"The solution has stabilized but hasn't reached the accuracy targets. "
                f"Mesh refinement or solver adjustments may help.\n"
            )
        else:
            outcome.append(f"Simulation completed with status: **{run_status}**.\n")
    else:
        outcome.append(f"Simulation completed with status: **{run_status}**.\n")
    parts.append("\n".join(outcome))

    # ── 4. KEY NUMBERS ─────────────────────────────────────────────────────
    key_rows: list[str] = []
    if snap.sim_progress:
        last = snap.sim_progress[-1]
        courant = last.get("courant", {})
        if courant.get("max") is not None:
            try:
                co = float(courant["max"])
                note = " (good)" if co < 1.0 else " (high, consider smaller time step)" if co > 5.0 else ""
                key_rows.append(f"| Max Courant Number | {co:.2f}{note} |")
            except (TypeError, ValueError):
                key_rows.append(f"| Max Courant Number | {courant['max']} |")

    if isinstance(snap.vtk_result, dict):
        vtk_fields = snap.vtk_result.get("fields", []) or []
        _STD_FIELDS: dict[str, tuple[str, str]] = {
            "U": ("Max Velocity", "m/s"),
            "p": ("Pressure Range", "Pa"),
            "p_rgh": ("Pressure Range", "Pa"),
            "T": ("Temperature Range", "K"),
        }
        seen_labels: set[str] = set()
        for fdata in vtk_fields:
            if not isinstance(fdata, dict):
                continue
            fname = fdata.get("name", "")
            if fname not in _STD_FIELDS:
                continue
            label, unit = _STD_FIELDS[fname]
            if label in seen_labels:
                continue
            seen_labels.add(label)
            raw_range = fdata.get("range")
            mn, mx = fdata.get("min"), fdata.get("max")
            if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
                mn = mn if mn is not None else raw_range[0]
                mx = mx if mx is not None else raw_range[1]
            if mn is not None and mx is not None:
                if fname == "U":
                    key_rows.append(f"| {label} | {float(mx):.2f} {unit} |")
                else:
                    key_rows.append(
                        f"| {label} | {float(mn):.1f} to {float(mx):.1f} {unit} |"
                    )

    if key_rows:
        parts.append(
            "## Key Numbers\n\n"
            "| Metric | Value |\n"
            "|---|---|\n" + "\n".join(key_rows)
        )

    # ── 5. INSIGHTS & NEXT STEPS ───────────────────────────────────────────
    insights: list[str] = ["## Insights & Next Steps\n"]
    if conv:
        conv_st = conv.get("status", "unknown")
        if conv_st == "converged":
            insights.append(
                "- The simulation produced a stable result. These values are suitable "
                "for engineering analysis and design decisions."
            )
            insights.append(
                "- For higher accuracy, consider refining the mesh in areas with steep "
                "flow gradients (near walls, obstacles, or sudden geometry changes)."
            )
            if energy:
                insights.append(
                    "- Temperature distribution is resolved. Check that wall temperatures "
                    "and heat fluxes match your physical expectations."
                )
        elif conv_st == "converging":
            insights.append(
                "- Extend the simulation (increase iterations or end time) to let the "
                "solution fully stabilize."
            )
            insights.append(
                "- The current trend is positive. Partial results can be used for rough "
                "estimates."
            )
        elif conv_st == "oscillating":
            if transient:
                insights.append(
                    "- Check if the oscillations correspond to physical flow features "
                    "(vortex shedding, flow instability)."
                )
                insights.append(
                    "- If oscillations seem too large, try reducing the time step or "
                    "increasing mesh resolution."
                )
            else:
                insights.append(
                    "- Consider switching to a transient solver to capture unsteady flow."
                )
                insights.append(
                    "- Alternatively, reduce relaxation factors to help the steady solver "
                    "converge."
                )
        elif conv_st == "diverging":
            insights.append(
                "- Double-check boundary conditions, especially pressure and velocity values."
            )
            insights.append(
                "- Verify mesh quality, particularly near walls and complex geometry."
            )
            insights.append(
                "- Try running with more conservative settings (smaller time step, "
                "higher relaxation)."
            )
        elif conv_st == "stalling":
            insights.append(
                "- Refine the mesh in regions with high gradients."
            )
            insights.append(
                "- Try adjusting solver relaxation factors or switching turbulence model."
            )
    else:
        insights.append(
            "- Review the 3D field visualizations to verify flow patterns match "
            "expectations."
        )
        insights.append(
            "- If results look reasonable, proceed with analysis. Otherwise, "
            "revisit boundary conditions."
        )
    parts.append("\n".join(insights))

    return "\n\n".join(parts)


async def generate_report(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Produce a simulation report (markdown + typed data block).

    Supports two report types:
      - ``"standard"`` (default): simplified, plain-language summary aimed at
        engineers who are not CFD specialists.
      - ``"expert"``: full engineering report with governing equations,
        discretization schemes, mesh quality, residual tables, etc.

    The ``focus`` parameter controls which 3-D field screenshots are prioritised
    in the report_request_payload — it does not omit sections.
    """
    report_type: str = args.get("report_type", "standard")
    if report_type not in ("standard", "expert"):
        report_type = "standard"
    focus: str = args.get("focus", "")
    # Structure is always the full standard report — ignore sections param
    sections = [
        "executive_summary", "introduction", "methodology",
        "mesh", "governing_equations", "turbulence_details",
        "transport_properties", "boundary_conditions",
        "initial_conditions", "solver_settings", "discretization",
        "residuals", "convergence", "results", "conclusions",
    ]

    parts: list[str] = []
    transient = _is_transient(snap)

    # ── Authoritative config from validated_config ────────────────────────────
    vc = _extract_validated_config(snap)
    solver_name = _vc_solver_name(vc, snap)
    turb_model  = _vc_turbulence_model(vc, snap)
    regime      = _vc_flow_regime(vc, snap)
    # Reynolds number — check multiple sources: frontend context, lint_result
    # (DB column on runs table), or nested inside agent_run.result.
    re_num = (
        snap.physics.get("Re")
        or snap.physics.get("reynoldsNumber")
        or snap.lint_result.get("reynolds_number")
        or snap.lint_result.get("reynoldsNumber")
    )
    if not re_num and isinstance(snap.agent_run, dict):
        _ar_result = snap.agent_run.get("result") or {}
        if isinstance(_ar_result, dict):
            _ar_lint = _ar_result.get("lint_result") or {}
            if isinstance(_ar_lint, dict):
                re_num = _ar_lint.get("reynolds_number") or _ar_lint.get("reynoldsNumber")
    if not re_num:
        re_num = "N/A"

    # Derived flags
    s_lower = (solver_name or "").lower()
    compressible = ("rho" in s_lower or "buoyant" in s_lower
                    or "compressible" in s_lower)
    energy = ("rho" in s_lower or "buoyant" in s_lower)
    vc_phys = vc.get("physics") or snap.physics or {}
    if vc_phys.get("heat_transfer"):
        energy = True

    # ── Common data used by both report types ─────────────────────────────
    fr = snap.final_result if isinstance(snap.final_result, dict) else {}
    run_status = fr.get("status") or (snap.agent_run.get("status") if snap.agent_run else "unknown")
    total_iters = len(snap.sim_progress) if snap.sim_progress else 0
    duration_s = fr.get("duration_seconds")

    conv = None
    if isinstance(snap.agent_run, dict):
        result_col = snap.agent_run.get("result")
        if isinstance(result_col, dict):
            conv = result_col.get("convergence")

    algo_name = ("PIMPLE" if transient and "pimple" in s_lower
                 else "SIMPLE" if "simple" in s_lower
                 else "PISO" if "piso" in s_lower else "PIMPLE")

    # ── Fetch LLM background paragraph (shared by both report types) ───────
    _bg_paragraph = ""
    if snap.simulation_id:
        try:
            from simd_agent.chat.db import fetch_chat_history
            from simd_agent.llm import get_provider
            _chat_msgs = await fetch_chat_history(snap.simulation_id, limit=30)
            _conv_lines: list[str] = []
            for m in _chat_msgs:
                role = m.get("role", "")
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role == "assistant" and len(content) > 300:
                    content = content[:300] + "…"
                _conv_lines.append(f"{role}: {content}")

            if _conv_lines:
                _conv_text = "\n".join(_conv_lines[-15:])
                _provider = get_provider()
                _summary_resp = await _provider.client.aio.models.generate_content(
                    model=_provider.models["default"],
                    contents=(
                        f"Summarize the following conversation between a user and a CFD assistant "
                        f"into 2–3 sentences suitable for the 'Background' section of an engineering "
                        f"simulation report. Write in third person, professional tone. "
                        f"Focus on: what the user wanted to simulate, the key physical setup "
                        f"(geometry, fluid, conditions), and the objective. "
                        f"Do NOT mention the chat or conversation itself.\n\n"
                        f"{_conv_text}"
                    ),
                    config=_provider.types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=256,
                    ),
                )
                _bg_paragraph = (_summary_resp.text or "").strip()
        except Exception as exc:
            logger.debug("[generate_report] background summary failed: %s", exc)

    # ── Branch: standard vs expert report ───────────────────────────────────
    if report_type == "standard":
        report_md = _build_standard_report(
            snap=snap, vc=vc,
            solver_name=solver_name, turb_model=turb_model,
            regime=regime, re_num=re_num,
            transient=transient, compressible=compressible, energy=energy,
            conv=conv, bg_paragraph=_bg_paragraph,
            run_status=run_status, total_iters=total_iters,
            duration_s=duration_s,
        )
        sections = ["overview", "setup", "outcome", "key_numbers", "insights"]

    # ══════════════════════════════════════════════════════════════════════════
    # Expert report — full engineering sections (skipped when standard)
    # ══════════════════════════════════════════════════════════════════════════
    if report_type == "expert":

        # 1. Executive Summary
        exec_lines: list[str] = ["## 1. Executive Summary\n"]
        _exec_kv: list[str] = []
        _exec_kv.append("**Generated by:** SIMD Agent")
        _exec_kv.append(f"**Solver:** **{solver_name}** ({'transient' if transient else 'steady-state'}, "
                        f"{'compressible' if compressible else 'incompressible'})")
        _regime_label = regime or "N/A"
        if re_num and re_num != "N/A":
            try:
                _exec_kv.append(f"**Flow Regime:** {_regime_label} (Re = {float(re_num):,.0f})")
            except (TypeError, ValueError):
                _exec_kv.append(f"**Flow Regime:** {_regime_label} (Re = {re_num})")
        else:
            _exec_kv.append(f"**Flow Regime:** {_regime_label}")
        _exec_kv.append(f"**Turbulence Model:** {turb_model or 'laminar'}")
        if energy:
            _exec_kv.append("**Energy Equation:** Active (heat transfer enabled)")
        _exec_kv.append(f"**Status:** {run_status}")
        if total_iters:
            _exec_kv.append(f"**Total {'Time Steps' if transient else 'Iterations'}:** {total_iters}")
        if duration_s:
            _exec_kv.append(f"**Wall-Clock Time:** {duration_s:.1f} s")
        _exec_kv.append("**Parallel Computation:** 12 MPI processes (domain decomposition: scotch)")
        if conv:
            _STATUS_MAP = {"converged": "Converged", "converging": "Converging",
                           "oscillating": "Oscillating", "diverging": "Diverging",
                           "stalling": "Stalling"}
            _exec_kv.append(f"**Convergence:** {_STATUS_MAP.get(conv.get('status', ''), conv.get('status', ''))}")
        exec_lines.append("  \n".join(_exec_kv))
        parts.append("\n".join(exec_lines))

        # 2. Introduction and Background
        intro_lines: list[str] = ["## 2. Introduction and Background\n"]
        if _bg_paragraph:
            intro_lines.append(f"{_bg_paragraph}\n")

        vc_mesh_data = vc.get("mesh") or {}
        mesh_patches = vc_mesh_data.get("patches") or (snap.mesh_info.get("patches") if snap.mesh_info else []) or []
        patch_names = [p.get("name", "") if isinstance(p, dict) else str(p) for p in mesh_patches]
        patch_lower = [n.lower() for n in patch_names]
        geo_type = "internal flow domain"
        if any("freestream" in n or "farfield" in n for n in patch_lower):
            geo_type = "external aerodynamic domain"
        elif any("inlet" in n for n in patch_lower) and any("outlet" in n for n in patch_lower):
            geo_type = "internal flow domain (enclosed passage with inlet/outlet)"

        intro_lines.append(f"**Problem Type:** CFD analysis of {geo_type}\n")
        intro_lines.append(
            f"**Objective:** Solve the Reynolds-Averaged Navier-Stokes (RANS) equations "
            f"using the {algo_name} pressure-velocity coupling algorithm to obtain the "
            f"{'time-dependent' if transient else 'steady-state'} flow field"
            f"{', temperature distribution,' if energy else ''} and turbulence quantities.\n"
        )
        parts.append("\n".join(intro_lines))

    # ── Expert sections 3–12 (kept at original indent, guarded) ─────────
    if report_type == "expert":

        # ══════════════════════════════════════════════════════════════════
        # 3. SIMULATION METHODOLOGY
        # ══════════════════════════════════════════════════════════════════
        method_lines: list[str] = ["## 3. Simulation Methodology\n"]
        method_lines.append("### 3.1 Software\n")
        method_lines.append("| Property | Value |")
        method_lines.append("|---|---|")
        method_lines.append("| Software | SIMD Agent (OpenFOAM v2406 ESI) |")
        method_lines.append(f"| Solver | **{solver_name}** |")
        method_lines.append(f"| Algorithm | {algo_name} |")
        method_lines.append(f"| Time scheme | {'transient' if transient else 'steady-state (pseudo-time)'} |")
        method_lines.append("| Parallelisation | MPI (Open MPI) |")
        method_lines.append("| Processes | 12 |")
        method_lines.append(f"| Domain decomposition | scotch (automatic graph partitioning) |")
        method_lines.append("")

        method_lines.append("\n### 3.2 Solution Strategy\n")
        method_lines.append(
            f"The simulation employs the {algo_name} algorithm, which iteratively solves the "
            f"pressure-velocity coupling. "
        )
        if transient:
            method_lines.append(
                "For each time step, multiple outer correctors (PIMPLE loops) ensure "
                "tight coupling between the equations before advancing in time.\n"
            )
        else:
            method_lines.append(
                "Under-relaxation factors are applied to stabilize the iterative process "
                "and ensure monotone convergence toward the steady-state solution.\n"
            )
        parts.append("\n".join(method_lines))

        # ══════════════════════════════════════════════════════════════════════════
        # 4. COMPUTATIONAL MESH
        # ══════════════════════════════════════════════════════════════════════════
        if "mesh" in sections:
            # Gather mesh data from all sources:
            # 1. validated_config.mesh (from linting) — has nested check_mesh sub-dict
            # 2. snap.mesh_info (from DB mesh_info.check_mesh) — has camelCase keys
            mi = vc_mesh_data if vc_mesh_data else {}
            cm = mi.get("check_mesh") or {}
            db_cm = snap.mesh_info or {}  # DB check_mesh column (may be the check_mesh dict itself)

            def _pick(*candidates: Any) -> Any:
                """Return first truthy non-zero value from candidates."""
                for c in candidates:
                    if c is not None and c != 0 and c != "":
                        return c
                return "N/A"

            cells = _pick(mi.get("cells"), cm.get("cells"), db_cm.get("cells"),
                           db_cm.get("numCells"), mi.get("numCells"))
            faces = _pick(mi.get("faces"), cm.get("faces"), db_cm.get("faces"),
                           db_cm.get("numFaces"), mi.get("numFaces"))
            points = _pick(mi.get("points"), cm.get("points"), db_cm.get("points"),
                            db_cm.get("numPoints"), mi.get("numPoints"))
            max_aspect = _pick(cm.get("max_aspect_ratio"), cm.get("maxAspectRatio"),
                               db_cm.get("max_aspect_ratio"), db_cm.get("maxAspectRatio"),
                               mi.get("max_aspect_ratio"), mi.get("maxAspectRatio"))
            max_skew = _pick(cm.get("max_skewness"), cm.get("maxSkewness"),
                             db_cm.get("max_skewness"), db_cm.get("maxSkewness"),
                             mi.get("max_skewness"), mi.get("maxSkewness"))
            def _pick_or_none(*candidates: Any) -> Any:
                """Like _pick but returns None instead of N/A for missing values."""
                for c in candidates:
                    if c is not None and c != 0 and c != "":
                        return c
                return None

            max_non_ortho = _pick_or_none(cm.get("max_non_orthogonality"), cm.get("maxNonOrthogonality"),
                                           db_cm.get("max_non_orthogonality"), db_cm.get("maxNonOrthogonality"),
                                           mi.get("max_non_orthogonality"), mi.get("maxNonOrthogonality"))
            avg_non_ortho = _pick_or_none(cm.get("avg_non_orthogonality"), cm.get("avgNonOrthogonality"),
                                           db_cm.get("avg_non_orthogonality"), db_cm.get("avgNonOrthogonality"),
                                           mi.get("avg_non_orthogonality"), mi.get("avgNonOrthogonality"))
            mesh_ok = cm.get("mesh_ok") if cm.get("mesh_ok") is not None else (
                db_cm.get("mesh_ok") if db_cm.get("mesh_ok") is not None else (
                    db_cm.get("meshOk") if db_cm.get("meshOk") is not None else mi.get("mesh_ok")))

            mesh_lines: list[str] = ["## 4. Computational Mesh\n"]
            mesh_lines.append("### 4.1 Mesh Statistics\n")
            mesh_lines.append("| Property | Value |")
            mesh_lines.append("|---|---|")
            mesh_lines.append(f"| Total cells | {cells} |")
            mesh_lines.append(f"| Total faces | {faces} |")
            mesh_lines.append(f"| Total points | {points} |")
            mesh_lines.append("| Element type | Polyhedral (converted from Gmsh) |")
            mesh_lines.append("")

            # Format numeric quality metrics for display
            def _fmt_metric(v: Any) -> str:
                if v == "N/A" or v is None:
                    return "N/A"
                try:
                    fv = float(v)
                    return f"{fv:.2f}" if fv < 1000 else f"{fv:.0f}"
                except (TypeError, ValueError):
                    return str(v)

            mesh_lines.append("### 4.2 Mesh Quality Metrics\n")
            mesh_lines.append("| Metric | Value | Acceptable Range |")
            mesh_lines.append("|---|---|---|")
            mesh_lines.append(f"| Max aspect ratio | {_fmt_metric(max_aspect)} | < 100 |")
            mesh_lines.append(f"| Max skewness | {_fmt_metric(max_skew)} | < 4.0 |")
            if max_non_ortho is not None:
                mesh_lines.append(f"| Max non-orthogonality | {_fmt_metric(max_non_ortho)} deg | < 70 deg |")
            if avg_non_ortho is not None:
                mesh_lines.append(f"| Avg non-orthogonality | {_fmt_metric(avg_non_ortho)} deg | < 40 deg |")
            quality_str = "PASS (no errors)" if mesh_ok else ("FAIL" if mesh_ok is False else "N/A")
            mesh_lines.append(f"| checkMesh result | {quality_str} | PASS |")
            mesh_lines.append("")

            mesh_lines.append(
                "\nMesh quality directly affects numerical accuracy and solver stability. "
                "Aspect ratio measures cell elongation (1.0 = perfect cube). "
                "Skewness measures deviation from ideal cell shape (0 = perfect). "
                "Non-orthogonality measures the angle between face normal and cell-to-cell vector "
                "(0 = perfectly orthogonal).\n"
            )

            # Patch table — infer wall type from BC data or patch name
            vc_bcs = vc.get("boundary_conditions") or snap.patches or {}
            if mesh_patches:
                mesh_lines.append("\n### 4.3 Boundary Patches\n")
                mesh_lines.append("| Patch | Type | Faces |")
                mesh_lines.append("|---|---|---|")
                for p in mesh_patches:
                    if isinstance(p, dict):
                        pn = p.get("name", "--")
                        pt = p.get("type", "--")
                        nf = p.get("nFaces") or p.get("n_faces", "--")
                        # Infer wall type from BC data or patch name
                        if pt == "patch" and pn in vc_bcs:
                            pbc = vc_bcs[pn]
                            if isinstance(pbc, dict):
                                vel_bc = pbc.get("velocity") or pbc.get("U") or {}
                                if isinstance(vel_bc, dict) and (vel_bc.get("type") or "").lower() == "noslip":
                                    pt = "wall"
                        if pt == "patch" and "wall" in pn.lower():
                            pt = "wall"
                        mesh_lines.append(f"| {pn} | {pt} | {nf} |")
                mesh_lines.append("")

            parts.append("\n".join(mesh_lines))

        # ══════════════════════════════════════════════════════════════════════════
        # 5. GOVERNING EQUATIONS
        # ══════════════════════════════════════════════════════════════════════════
        section = _governing_equations_section(
            solver_name, transient, compressible, energy, turb_model)
        if section:
            # Renumber to section 5
            section = section.replace("## Governing Equations", "## 5. Governing Equations", 1)
            parts.append(section)

        # ══════════════════════════════════════════════════════════════════════════
        # 6. TURBULENCE MODEL DETAILS
        # ══════════════════════════════════════════════════════════════════════════
        section = _turbulence_details_section(turb_model, snap)
        if section:
            section = section.replace("## Turbulence", "## 6. Turbulence", 1)
            parts.append(section)

        # ══════════════════════════════════════════════════════════════════════════
        # 7. TRANSPORT PROPERTIES
        # ══════════════════════════════════════════════════════════════════════════
        section = _transport_properties_section(snap, vc)
        if section:
            section = section.replace("## Transport", "## 7. Transport", 1)
            parts.append(section)

        # ══════════════════════════════════════════════════════════════════════════
        # 8. BOUNDARY CONDITIONS
        # ══════════════════════════════════════════════════════════════════════════
        section = _enhanced_boundary_conditions_section(snap, vc)
        if section:
            section = section.replace("## Boundary", "## 8. Boundary", 1)
            parts.append(section)

        # ══════════════════════════════════════════════════════════════════════════
        # 9. INITIAL CONDITIONS
        # ══════════════════════════════════════════════════════════════════════════
        section = _initial_conditions_section(snap)
        if section:
            section = section.replace("## Initial", "## 9. Initial", 1)
            parts.append(section)

        # ══════════════════════════════════════════════════════════════════════════
        # 10. SOLVER SETTINGS & NUMERICAL DISCRETIZATION
        # ══════════════════════════════════════════════════════════════════════════
        solver_section_lines: list[str] = ["## 10. Solver Settings and Numerical Discretization\n"]

        vc_solver_cfg = vc.get("solver") or {}
        if isinstance(vc_solver_cfg, dict):
            solver_section_lines.append("### 10.1 Run Parameters\n")
            solver_section_lines.append("| Parameter | Value |")
            solver_section_lines.append("|---|---|")
            solver_section_lines.append(f"| Solver | **{solver_name}** |")
            solver_section_lines.append(f"| Pressure-velocity coupling | {algo_name} |")
            _SOLVER_LABELS: dict[str, str] = {
                "max_iterations": "Max iterations",
                "end_time": "End time (s)", "delta_t": "Time step (s)",
                "write_interval": "Write interval",
                "convergence_criteria": "Convergence criteria",
            }
            for k, v in vc_solver_cfg.items():
                if v is not None and k not in ("type",) and k in _SOLVER_LABELS:
                    solver_section_lines.append(f"| {_SOLVER_LABELS[k]} | {v} |")
            solver_section_lines.append("")

        # Extract PIMPLE/SIMPLE loop parameters from fvSolution
        fv_solution = snap.generated_files.get("system/fvSolution", "")
        if fv_solution:
            import re as _re
            algo_param_rows: list[str] = []
            for param, label in [
                ("nOuterCorrectors", f"Outer correctors ({'PIMPLE loops' if transient else 'iterations'})"),
                ("nCorrectors", "Pressure correctors"),
                ("nNonOrthogonalCorrectors", "Non-orthogonal correctors"),
            ]:
                match = _re.search(rf'{param}\s+(\d+)', fv_solution)
                if match:
                    algo_param_rows.append(f"| {label} | {match.group(1)} |")
            if algo_param_rows:
                solver_section_lines.append("\n### 10.2 Algorithm Parameters\n")
                solver_section_lines.append("| Parameter | Value |")
                solver_section_lines.append("|---|---|")
                solver_section_lines.extend(algo_param_rows)
                solver_section_lines.append("")

        # Discretization schemes from fvSchemes
        section = _discretization_section(snap)
        if section:
            section = section.replace("## Numerical Discretization", "### 10.3 Discretization Schemes", 1)
            solver_section_lines.append("\n" + section)

        parts.append("\n".join(solver_section_lines))

        # ══════════════════════════════════════════════════════════════════════════
        # 11. RESULTS AND CONVERGENCE
        # ══════════════════════════════════════════════════════════════════════════
        results_lines: list[str] = ["## 11. Results and Convergence\n"]

        if snap.sim_progress:
            last = snap.sim_progress[-1]
            residuals = last.get("residuals", {})
            last_sim_time = last.get("sim_time")

            if residuals:
                results_lines.append("### 11.1 Final Residuals\n")
                if transient and last_sim_time is not None:
                    results_lines.append(
                        f"Final time step: t = {last_sim_time:.4g} s "
                        f"({total_iters} steps total)\n"
                    )
                else:
                    results_lines.append(
                        f"Final iteration: {last.get('iteration', total_iters)} / {total_iters}\n"
                    )
                results_lines.append("| Field | Initial Residual | Final Residual |")
                results_lines.append("|---|---|---|")
                for f, r in residuals.items():
                    init_r = r.get("initial", "N/A")
                    final_r = r.get("final", "N/A") if transient else r.get("initial", "N/A")
                    results_lines.append(f"| {f} | {init_r} | {final_r} |")
                results_lines.append(
                    "\nResiduals measure how well each conservation equation is satisfied. "
                    "Smaller values indicate better convergence (target: 10⁻⁴ to 10⁻⁶).\n"
                )

            courant = last.get("courant", {})
            if courant:
                results_lines.append(
                    f"\n**Courant number:** mean = {courant.get('mean', 'N/A')}, "
                    f"max = {courant.get('max', 'N/A')}\n"
                )

        # Convergence assessment
        if conv:
            results_lines.append("\n### 11.2 Convergence Assessment\n")
            _STATUS_MAP2 = {"converged": "Converged", "converging": "Converging",
                            "oscillating": "Oscillating", "stalling": "Stalling",
                            "diverging": "Diverging"}
            results_lines.append(f"**Overall status: {_STATUS_MAP2.get(conv.get('status', ''), conv.get('status', ''))}**\n")
            fields_data = conv.get("fields", [])
            if fields_data:
                results_lines.append("")
                results_lines.append("| Field | First | Current | Drop | Target | Status |")
                results_lines.append("|---|---|---|---|---|---|")
                for fm in fields_data:
                    drop = fm.get("ordersDrop", 0)
                    results_lines.append(
                        f"| {fm['field']} | {fm.get('firstResidual', 0):.1e} | "
                        f"{fm.get('lastResidual', 0):.1e} | {drop:.1f} OoM | "
                        f"{fm.get('threshold', 0):.0e} | "
                        f"{_STATUS_MAP2.get(fm.get('status', ''), fm.get('status', ''))} |"
                    )
                results_lines.append("")
            cont = conv.get("continuity")
            if cont:
                results_lines.append(
                    f"\n**Continuity error:** {cont['lastValue']:.2e} "
                    f"(slope: {cont['recentSlope']:.4f})\n"
                )
            courant_a = conv.get("courant")
            if courant_a:
                results_lines.append(
                    f"**Courant number:** max {courant_a['recentMax']:.2f} — "
                    f"{courant_a.get('level', 'acceptable')}\n"
                )

        parts.append("\n".join(results_lines))

        # ══════════════════════════════════════════════════════════════════════════
        # 12. CONCLUSIONS AND RECOMMENDATIONS
        # ══════════════════════════════════════════════════════════════════════════
        concl_lines: list[str] = ["## 12. Conclusions and Recommendations\n"]
        concl_lines.append("### Findings\n")
        if conv:
            conv_st = conv.get("status", "unknown")
            if conv_st == "converged":
                concl_lines.append(
                    "The simulation has converged successfully. All field residuals have "
                    "dropped below their target thresholds, indicating a well-resolved solution.\n"
                )
            elif conv_st == "converging":
                concl_lines.append(
                    "The simulation is converging but has not yet reached the target residual "
                    "thresholds. Additional iterations may be required.\n"
                )
            elif conv_st == "oscillating":
                concl_lines.append(
                    "Residuals are oscillating, which may indicate insufficient under-relaxation, "
                    "mesh quality issues, or physical instability in the flow.\n"
                )
            elif conv_st == "diverging":
                concl_lines.append(
                    "The simulation is diverging. This typically indicates issues with boundary "
                    "conditions, mesh quality, or time step size.\n"
                )
            else:
                concl_lines.append(f"Simulation completed with status: {run_status}.\n")
        else:
            concl_lines.append(f"Simulation completed with status: {run_status}.\n")

        concl_lines.append("\n### Recommendations\n")
        concl_lines.append(
            "- Verify boundary condition values against physical expectations\n"
            "- Consider mesh refinement in regions of high gradients for improved accuracy\n"
        )
        if conv and conv.get("status") == "oscillating":
            concl_lines.append("- Reduce under-relaxation factors to improve stability\n")
            concl_lines.append("- Check mesh quality in regions where oscillations originate\n")
        if transient:
            concl_lines.append("- Verify time step satisfies CFL condition (Co < 1) throughout the domain\n")

        parts.append("\n".join(concl_lines))

        report_md = "\n\n".join(parts) if parts else "No simulation data available yet."

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

        # Sort by physical importance: primary flow fields first, then
        # turbulence, then derived/auxiliary.  Unknown fields go last.
        _FIELD_PRIORITY: dict[str, int] = {
            "U": 0, "p": 1, "p_rgh": 2, "T": 3,
            "k": 4, "omega": 5, "epsilon": 6,
            "nut": 7, "alphat": 8, "mut": 9,
        }
        display_fields = sorted(
            display_fields,
            key=lambda f: _FIELD_PRIORITY.get(f.get("name", ""), 100),
        )

        _FIELD_UNITS: dict[str, str] = {
            "U": "m/s", "p": "Pa", "p_rgh": "Pa", "T": "K",
            "k": "m²/s²", "omega": "1/s", "epsilon": "m²/s³",
            "nut": "m²/s", "alphat": "kg/m·s",
        }

        _SKIP_FIELDS = {
            "vtkOriginalPointIds", "vtkOriginalCellIds", "Normals",
            "TCoords", "vtkGhostType",
            # Backend-computed magnitudes — the frontend renders _mag from
            # the vector source automatically, so skip standalone copies.
            "U_magnitude", "wallShearStress_magnitude",
        }

        # Derived/internal fields — only shown in expert reports.
        _DERIVED_FIELDS = {"nut", "alphat", "p_rgh", "phi", "meshPhi", "nuTilda", "mut"}
        if report_type == "standard":
            _SKIP_FIELDS = _SKIP_FIELDS | _DERIVED_FIELDS

        for fdata in display_fields:
            fname = fdata.get("name") or fdata.get("field") or ""
            if not fname or fname in _SKIP_FIELDS:
                continue
            colormap = _COLORMAPS.get(fname, "turbo")

            # Handle both field formats:
            #   Frontend (SimVtkResult.fields): {"name":"p", "range":[0,1.5], "num_components":1}
            #   Server/DB format:               {"name":"p", "min":0, "max":1.5}
            raw_range = fdata.get("range")
            if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
                field_range = [float(raw_range[0]), float(raw_range[1])]
            elif fdata.get("min") is not None and fdata.get("max") is not None:
                field_range = [float(fdata["min"]), float(fdata["max"])]
            else:
                field_range = None

            unit = _FIELD_UNITS.get(fname, "")
            unit_str = f" ({unit})" if unit else ""

            # Check if multiple timesteps exist (always show first+last comparison)
            n_timesteps = snap.vtk_result.get("total_timesteps", 1) if isinstance(snap.vtk_result, dict) else 1
            has_multiple = n_timesteps > 1

            if has_multiple:
                if transient:
                    last_step = snap.sim_progress[-1] if snap.sim_progress else {}
                    t_final = last_step.get("sim_time", "")
                    t_label = f" (t = {t_final} s)" if t_final != "" else ""
                    first_caption = f"{fname} at the start of the simulation (first timestep)"
                    last_caption = f"{fname} at the end of the simulation{t_label}"
                    last_label = f"{fname}{unit_str} — Final State{t_label}"
                else:
                    first_caption = f"{fname}{unit_str} distribution (first output iteration)"
                    last_caption = f"Converged {fname}{unit_str} distribution (final iteration)"
                    last_label = f"{fname}{unit_str} — Final Solution"

                field_screenshots.append({
                    "field_name": fname,
                    "label": f"{fname}{unit_str} — Initial State",
                    "timestep": "first",
                    "colormap": colormap,
                    "range": field_range,
                    "caption": first_caption,
                })
                field_screenshots.append({
                    "field_name": fname,
                    "label": last_label,
                    "timestep": None,   # null = latest frame
                    "colormap": colormap,
                    "range": field_range,
                    "caption": last_caption,
                })
            else:
                field_screenshots.append({
                    "field_name": fname,
                    "label": f"{fname}{unit_str} — Final Solution",
                    "timestep": None,
                    "colormap": colormap,
                    "range": field_range,
                    "caption": f"Steady-state {fname}{unit_str} distribution (final iteration)",
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
    metrics.append({"label": "Solver", "value": solver_name, "unit": None})
    metrics.append({"label": "Turbulence model", "value": turb_model, "unit": None})
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

    # For standard reports, trim metrics to the essentials
    if report_type == "standard":
        metrics = [m for m in metrics if m["label"] in (
            "Solver", "Iterations", "Simulation Time",
        )]

    report_request_payload: dict[str, Any] = {
        "title": "Simulation Summary" if report_type == "standard" else "Simulation Report",
        "summary": "",
        "report_type": report_type,
        "mesh_screenshot": "mesh" in sections,
        "field_screenshots": field_screenshots,
        "convergence_charts": convergence_charts,
        "metrics": metrics,
        "scope": scope,
    }

    # ── Telemetry: report generated ──
    from simd_agent.telemetry import get_telemetry, ReportGenerated
    get_telemetry().capture(ReportGenerated(
        solver=solver_name,
        flow_regime=regime,
        has_results=bool(snap.sim_progress),
    ))

    return {
        "report_markdown": report_md,
        "report_type": report_type,
        "sections_included": sections,
        "report_data": {
            "solver": solver_name,
            "turbulence_model": turb_model,
            "flow_regime": regime,
            "status": fr.get("status") or (snap.agent_run.get("status") if snap.agent_run else None),
            "iterations": fr.get("iterations"),
            "duration_seconds": fr.get("duration_seconds"),
            "last_residuals": snap.sim_progress[-1].get("residuals") if snap.sim_progress else None,
            "total_iterations": len(snap.sim_progress),
        },
        "report_request_payload": report_request_payload,
    }


def query_simulation_results(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Return a comprehensive overview of all simulation results in one call.

    This is the primary tool for answering user questions about simulation
    outcomes: "what is the max pressure?", "what is the pressure drop?",
    "is this result good?", "what do you recommend?", etc.

    It assembles:
    - All VTK field ranges (spatial min/max over the full mesh)
    - Boundary condition set-points (inlet/outlet velocities, pressures, temps)
    - Derived quantities (pressure drop, flow rate if computable)
    - Convergence summary
    - Solver/physics context
    """
    question = args.get("question", "")

    result: dict[str, Any] = {}

    # ── 1. VTK field ranges (spatial min/max) ─────────────────────────────
    vtk = snap.vtk_result or {}
    vtk_fields = vtk.get("fields", [])
    _FIELD_UNITS: dict[str, str] = {
        "U": "m/s", "p": "Pa", "p_rgh": "Pa", "T": "K",
        "k": "m²/s²", "omega": "1/s", "epsilon": "m²/s³",
        "nut": "m²/s", "alphat": "kg/m·s", "nuTilda": "m²/s",
    }
    if isinstance(vtk_fields, list) and vtk_fields:
        field_summary = []
        for fdata in vtk_fields:
            if not isinstance(fdata, dict):
                continue
            fname = fdata.get("name", "")
            raw_range = fdata.get("range")
            mn = fdata.get("min")
            mx = fdata.get("max")
            mean = fdata.get("mean")
            if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
                if mn is None:
                    mn = raw_range[0]
                if mx is None:
                    mx = raw_range[1]
            nc = fdata.get("num_components", 1)
            entry: dict[str, Any] = {
                "field": fname,
                "unit": _FIELD_UNITS.get(fname, ""),
                "min": mn, "max": mx,
            }
            if mean is not None:
                entry["mean"] = mean
            if nc > 1:
                entry["note"] = "magnitude of vector field"
            field_summary.append(entry)
        result["field_ranges"] = field_summary
        result["vtk_time"] = vtk.get("time")

    # ── 2. Boundary condition values ──────────────────────────────────────
    patches = snap.patches or {}
    bc_summary: dict[str, Any] = {}
    for pname, pdata in patches.items():
        if not isinstance(pdata, dict):
            continue
        patch_bc: dict[str, Any] = {}
        for field_key, bc in pdata.items():
            if not isinstance(bc, dict) or field_key in ("patch_type", "patchType"):
                continue
            entry = {"type": bc.get("type", "—")}
            val = bc.get("value")
            if val is not None:
                entry["value"] = val
            patch_bc[field_key] = entry
        if patch_bc:
            bc_summary[pname] = patch_bc
    if bc_summary:
        result["boundary_conditions"] = bc_summary

    # ── 3. Derived quantities ─────────────────────────────────────────────
    derived: dict[str, Any] = {}

    # Pressure drop (outlet p - inlet p from field ranges + BCs)
    p_field = None
    for fdata in vtk_fields if isinstance(vtk_fields, list) else []:
        if isinstance(fdata, dict) and fdata.get("name") == "p":
            raw_range = fdata.get("range")
            if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
                p_field = {"min": raw_range[0], "max": raw_range[1]}
            elif fdata.get("min") is not None and fdata.get("max") is not None:
                p_field = {"min": fdata["min"], "max": fdata["max"]}

    if p_field:
        derived["pressure_range"] = {
            "min_Pa": p_field["min"],
            "max_Pa": p_field["max"],
            "delta_Pa": p_field["max"] - p_field["min"],
            "note": "Total pressure range across the domain. "
                    "For internal flows, this approximates the pressure drop.",
        }

    # Velocity range
    u_field = None
    for fdata in vtk_fields if isinstance(vtk_fields, list) else []:
        if isinstance(fdata, dict) and fdata.get("name") == "U":
            raw_range = fdata.get("range")
            if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
                u_field = {"min": raw_range[0], "max": raw_range[1]}
            elif fdata.get("min") is not None and fdata.get("max") is not None:
                u_field = {"min": fdata["min"], "max": fdata["max"]}

    if u_field:
        derived["velocity_range"] = {
            "min_ms": u_field["min"],
            "max_ms": u_field["max"],
            "note": "Magnitude range. Max > inlet velocity suggests acceleration "
                    "due to geometry or boundary layer effects.",
        }

    # Reynolds number
    re_num = snap.physics.get("Re") or snap.physics.get("reynoldsNumber")
    if re_num:
        derived["reynolds_number"] = re_num

    if derived:
        result["derived_quantities"] = derived

    # ── 4. Convergence summary ────────────────────────────────────────────
    conv = None
    if isinstance(snap.agent_run, dict):
        result_col = snap.agent_run.get("result")
        if isinstance(result_col, dict):
            conv = result_col.get("convergence")
    if conv:
        conv_summary: dict[str, Any] = {
            "status": conv.get("status", "unknown"),
        }
        fields_data = conv.get("fields", [])
        if fields_data:
            conv_summary["fields"] = [
                {
                    "field": fm["field"],
                    "status": fm.get("status", ""),
                    "orders_drop": fm.get("ordersDrop", 0),
                    "last_residual": fm.get("lastResidual"),
                    "threshold": fm.get("threshold"),
                }
                for fm in fields_data
            ]
        cont = conv.get("continuity")
        if cont:
            conv_summary["continuity_error"] = cont.get("lastValue")
        courant_a = conv.get("courant")
        if courant_a:
            conv_summary["max_courant"] = courant_a.get("recentMax")
        result["convergence"] = conv_summary

    # ── 5. Solver / physics context ───────────────────────────────────────
    vc = _extract_validated_config(snap)
    solver_name = _vc_solver_name(vc, snap)
    turb_model = _vc_turbulence_model(vc, snap)
    regime = _vc_flow_regime(vc, snap)
    transient = _is_transient(snap)

    result["simulation_info"] = {
        "solver": solver_name,
        "turbulence_model": turb_model,
        "flow_regime": regime,
        "time_treatment": "transient" if transient else "steady_state",
        "total_iterations": len(snap.sim_progress),
    }

    # Last residuals
    if snap.sim_progress:
        last = snap.sim_progress[-1]
        last_res = last.get("residuals", {})
        if last_res:
            result["last_residuals"] = {
                f: {
                    "initial": r.get("initial"),
                    "final": r.get("final"),
                }
                for f, r in last_res.items()
            }
        if last.get("sim_time") is not None:
            result["final_sim_time"] = last["sim_time"]

    # ── 6. Fluid properties ───────────────────────────────────────────────
    vc_fluid = vc.get("fluid") or {}
    fluid = vc_fluid if vc_fluid else snap.fluid
    if fluid:
        result["fluid"] = {
            k: v for k, v in fluid.items()
            if v is not None and v != "" and v != []
        }

    # ── 7. Mesh summary ──────────────────────────────────────────────────
    mi = vc.get("mesh") or snap.mesh_info or {}
    if mi:
        cm = mi.get("check_mesh") or {}
        result["mesh"] = {
            "cells": mi.get("cells") or cm.get("cells") or mi.get("numCells"),
            "points": mi.get("points") or cm.get("points") or mi.get("numPoints"),
            "max_aspect_ratio": (mi.get("max_aspect_ratio") or mi.get("maxAspectRatio")
                                 or cm.get("max_aspect_ratio") or cm.get("maxAspectRatio")),
            "max_skewness": (mi.get("max_skewness") or mi.get("maxSkewness")
                             or cm.get("max_skewness") or cm.get("maxSkewness")),
            "max_non_orthogonality": (mi.get("max_non_orthogonality") or mi.get("maxNonOrthogonality")
                                      or cm.get("max_non_orthogonality") or cm.get("maxNonOrthogonality")),
        }

    if not result:
        return {"error": "No simulation results available yet."}

    result["question"] = question
    return result


def plot_field_over_iterations(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Plot a quantity extracted from sim_progress across all iterations.

    Supports:
      - Residual fields (Ux, Uy, Uz, p, k, omega, epsilon, …)
      - "courant" → Courant number over time
      - "continuity" → continuity error over time
      - Multiple fields comma-separated ("Ux,p,k")

    Returns a recharts-compatible chart artifact that the frontend renders
    inline in the chat.
    """
    fields_arg = args.get("fields", "")
    chart_title: str = args.get("title", "")

    if not snap.sim_progress:
        return {
            "error": "No iteration-by-iteration progress data was recorded for this run. "
                     "Try 'compute_field_stats' for VTK-based spatial statistics, or "
                     "'query_simulation_results' for a results overview."
        }

    # Parse field list (may be a list from the analyzer or comma-separated string)
    if isinstance(fields_arg, list):
        requested = [f.strip() for f in fields_arg if isinstance(f, str) and f.strip()]
    else:
        requested = [f.strip() for f in str(fields_arg).split(",") if f.strip()]
    if not requested:
        return {"error": "No fields specified. Use e.g. fields='p,Ux' or fields='courant'."}

    transient = _is_transient(snap)
    has_sim_time = any(
        step.get("sim_time") is not None for step in snap.sim_progress[:5]
    )
    x_key = "sim_time" if has_sim_time else "iteration"
    x_label = "Simulation Time (s)" if has_sim_time else "Iteration"

    # Determine which data to extract per field
    special_fields = {"courant", "continuity"}
    residual_fields = [f for f in requested if f.lower() not in special_fields]
    include_courant = any(f.lower() == "courant" for f in requested)
    include_continuity = any(f.lower() == "continuity" for f in requested)

    residual_key = "final" if transient else "initial"

    # Build rows keyed by x value
    x_to_row: dict[Any, dict[str, Any]] = {}

    for step in snap.sim_progress:
        xv = step.get("sim_time") if has_sim_time else step.get("iteration", 0)
        if xv not in x_to_row:
            x_to_row[xv] = {x_key: xv}
        row = x_to_row[xv]

        # Residual fields
        residuals = step.get("residuals", {})
        for fname in residual_fields:
            if fname in residuals:
                r = residuals[fname]
                val = r.get(residual_key) or r.get("initial") or r.get("final")
                if val is not None and val > 0:
                    row[fname] = val

        # Courant number (may be a dict {mean, max} or a scalar)
        if include_courant:
            co = step.get("courant")
            if isinstance(co, dict):
                max_co = co.get("max")
                if max_co is not None:
                    try:
                        row["Courant"] = float(max_co)
                    except (TypeError, ValueError):
                        pass
            elif isinstance(co, (int, float)) and co > 0:
                row["Courant"] = co

        # Continuity error
        if include_continuity:
            cont = step.get("continuity")
            if cont is not None:
                row["Continuity"] = abs(cont) if cont != 0 else None

    all_rows = [x_to_row[xv] for xv in sorted(x_to_row.keys())]
    all_rows = [r for r in all_rows if len(r) > 1]

    if not all_rows:
        return {"error": f"No data found for fields: {', '.join(requested)}"}

    # Downsample to ≤ 300 points
    MAX_PTS = 300
    if len(all_rows) > MAX_PTS:
        step_size = len(all_rows) / MAX_PTS
        indices = {0, len(all_rows) - 1}
        indices |= {int(i * step_size) for i in range(1, MAX_PTS - 1)}
        all_rows = [all_rows[i] for i in sorted(indices)]

    # Build line names from actual data
    line_names: list[str] = []
    for fname in residual_fields:
        if any(fname in r for r in all_rows):
            line_names.append(fname)
    if include_courant and any("Courant" in r for r in all_rows):
        line_names.append("Courant")
    if include_continuity and any("Continuity" in r for r in all_rows):
        line_names.append("Continuity")

    # Determine y-axis scale: log for residuals, auto for Courant/continuity
    use_log = any(f not in ("Courant", "Continuity") for f in line_names)
    y_label = (
        f"{'Final' if transient else 'Initial'} Residual"
        if use_log else "Value"
    )

    title = chart_title or f"{', '.join(line_names)} vs {x_label}"

    return {
        "total_iterations": len(snap.sim_progress),
        "fields_plotted": line_names,
        "chart": {
            "type": "line",
            "title": title,
            "xKey": x_key,
            "xLabel": x_label,
            "yLabel": y_label,
            "yScale": "log" if use_log else "auto",
            "lines": line_names,
            "data": all_rows,
        },
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
# Field value plotting (actual min/max from fieldMinMax, not residuals)
# ---------------------------------------------------------------------------

_FIELD_VALUE_UNITS: dict[str, str] = {
    "p": "Pa", "p_rgh": "Pa", "T": "K", "k": "m²/s²",
    "omega": "1/s", "epsilon": "m²/s³", "nut": "m²/s",
    "alphat": "kg/m·s", "alpha.water": "—", "rho": "kg/m³",
}


def plot_field_values(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Plot actual field values (min/max) over simulation time from fieldMinMax data.

    Unlike residual plots (which show solver convergence), this shows the
    physical field values themselves — e.g. how pressure or temperature evolve.
    """
    fields_arg = args.get("fields", "p")
    metric: str = args.get("metric", "both")  # min, max, both, range
    chart_title: str = args.get("title", "")

    if isinstance(fields_arg, list):
        requested = [f.strip() for f in fields_arg if isinstance(f, str) and f.strip()]
    else:
        requested = [f.strip() for f in str(fields_arg).split(",") if f.strip()]
    if not requested:
        return {"error": "No fields specified. Use e.g. fields='p' or fields='p,T'."}

    has_sim_time = any(
        step.get("sim_time") is not None for step in snap.sim_progress[:5]
    )
    x_key = "sim_time" if has_sim_time else "iteration"
    x_label = "Simulation Time (s)" if has_sim_time else "Iteration"

    # Build rows from field_ranges data
    x_to_row: dict[Any, dict[str, Any]] = {}
    found_fields: set[str] = set()

    for step in snap.sim_progress:
        fr = step.get("field_ranges") or step.get("fieldRanges")
        if not fr or not isinstance(fr, dict):
            continue

        xv = step.get("sim_time") if has_sim_time else step.get("iteration", 0)
        if xv not in x_to_row:
            x_to_row[xv] = {x_key: xv}
        row = x_to_row[xv]

        for fname in requested:
            fdata = fr.get(fname)
            if not fdata or not isinstance(fdata, dict):
                continue

            found_fields.add(fname)
            fmin = fdata.get("min")
            fmax = fdata.get("max")

            if metric in ("min", "both") and fmin is not None:
                row[f"{fname}_min"] = fmin
            if metric in ("max", "both") and fmax is not None:
                row[f"{fname}_max"] = fmax
            if metric == "range" and fmin is not None and fmax is not None:
                row[f"{fname}_range"] = fmax - fmin

    if not found_fields:
        # No field_ranges (fieldMinMax) data recorded for this run.
        # Do NOT fall back to residuals here — that duplicates
        # plot_field_over_iterations / compute_residual_trend and produces
        # two nearly-identical charts when both tools are planned.
        # Instead, fall back to VTK spatial stats or return an error.
        vtk = snap.vtk_result or {}
        vtk_fields = vtk.get("fields", [])
        if isinstance(vtk_fields, list) and vtk_fields:
            vtk_data = []
            for fdata in vtk_fields:
                if not isinstance(fdata, dict):
                    continue
                fname = fdata.get("name", "")
                if fname not in requested:
                    continue
                mn = fdata.get("min")
                mx = fdata.get("max")
                raw_range = fdata.get("range")
                if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
                    mn = mn if mn is not None else raw_range[0]
                    mx = mx if mx is not None else raw_range[1]
                if mn is not None:
                    vtk_data.append({"stat": "min", fname: mn})
                if mx is not None:
                    vtk_data.append({"stat": "max", fname: mx})

            if vtk_data:
                return {
                    "fields": list(requested),
                    "source": "vtk_result (final timestep only)",
                    "note": "Field evolution data (fieldMinMax) is not available. "
                            "Showing spatial min/max from the final VTK result instead.",
                    "chart": {
                        "type": "bar",
                        "title": f"{', '.join(requested)} — Final Timestep Statistics",
                        "xKey": "stat",
                        "yLabel": ", ".join(
                            _FIELD_VALUE_UNITS.get(f, "") for f in requested if _FIELD_VALUE_UNITS.get(f)
                        ) or "Value",
                        "lines": list(requested),
                        "yKey": requested[0],
                        "data": vtk_data,
                    },
                }

        return {
            "error": (
                f"No field value data found for '{fields_arg}'. "
                "Field evolution data (fieldMinMax) was not recorded during this run. "
                "Try 'plot_field_over_iterations' for residual convergence plots, or "
                "'compute_field_stats' for VTK-based spatial statistics."
            ),
        }

    all_rows = [x_to_row[xv] for xv in sorted(x_to_row.keys())]
    all_rows = [r for r in all_rows if len(r) > 1]

    # Downsample to ≤ 300 points
    MAX_PTS = 300
    if len(all_rows) > MAX_PTS:
        step_size = len(all_rows) / MAX_PTS
        indices = {0, len(all_rows) - 1}
        indices |= {int(i * step_size) for i in range(1, MAX_PTS - 1)}
        all_rows = [all_rows[i] for i in sorted(indices)]

    # Build line names
    line_names: list[str] = []
    for fname in requested:
        if fname not in found_fields:
            continue
        if metric in ("min", "both"):
            line_names.append(f"{fname}_min")
        if metric in ("max", "both"):
            line_names.append(f"{fname}_max")
        if metric == "range":
            line_names.append(f"{fname}_range")

    unit_parts = [_FIELD_VALUE_UNITS.get(f, "") for f in found_fields if _FIELD_VALUE_UNITS.get(f)]
    y_label = ", ".join(dict.fromkeys(unit_parts)) if unit_parts else "Value"
    title = chart_title or f"{', '.join(found_fields)} over {x_label}"

    return {
        "fields_plotted": list(found_fields),
        "metric": metric,
        "total_data_points": len(all_rows),
        "chart": {
            "type": "line",
            "title": title,
            "xKey": x_key,
            "xLabel": x_label,
            "yLabel": y_label,
            "yScale": "auto",
            "lines": line_names,
            "data": all_rows,
        },
    }


def plot_patch_values(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Plot patch-averaged values over iterations and compute derived quantities.

    Uses surfaceFieldValue data (patchValues / patch_values) recorded each
    iteration to show how patch-averaged pressure, temperature, etc. evolve.
    Can also compute pressure drop (inlet − outlet), temperature drop,
    or other cross-patch differences.
    """
    fields_arg = args.get("fields", "p")
    patches_arg = args.get("patches", "")
    quantity: str = args.get("quantity", "values")  # values, drop, difference
    chart_title: str = args.get("title", "")

    if isinstance(fields_arg, list):
        requested_fields = [f.strip() for f in fields_arg if isinstance(f, str) and f.strip()]
    else:
        requested_fields = [f.strip() for f in str(fields_arg).split(",") if f.strip()]
    if not requested_fields:
        return {"error": "No fields specified. Use e.g. fields='p' or fields='p,T'."}

    if isinstance(patches_arg, list):
        requested_patches = [p.strip() for p in patches_arg if isinstance(p, str) and p.strip()]
    else:
        requested_patches = [p.strip() for p in str(patches_arg).split(",") if p.strip()] if patches_arg else []

    # Collect all available patch_values data
    has_sim_time = any(
        step.get("sim_time") is not None for step in snap.sim_progress[:5]
    )
    x_key = "sim_time" if has_sim_time else "iteration"
    x_label = "Simulation Time (s)" if has_sim_time else "Iteration"

    # Discover which patches have data
    available_patches: set[str] = set()
    for step in snap.sim_progress:
        pv = step.get("patch_values") or step.get("patchValues")
        if pv and isinstance(pv, dict):
            available_patches.update(pv.keys())

    if not available_patches:
        return {
            "error": (
                "No patch-averaged data available for this simulation. "
                "Patch-averaged values (surfaceFieldValue) were not recorded during this run. "
                "This feature requires a simulation run with surfaceFieldValue function objects "
                "in controlDict."
            ),
        }

    # If no patches specified, use all available
    target_patches = requested_patches if requested_patches else sorted(available_patches)
    # Validate requested patches
    missing = [p for p in target_patches if p not in available_patches]
    if missing and not any(p in available_patches for p in target_patches):
        return {
            "error": f"Patches {missing} not found in patch-averaged data. "
                     f"Available patches: {sorted(available_patches)}",
        }
    target_patches = [p for p in target_patches if p in available_patches]

    # For "drop"/"difference" mode, we need exactly 2 patches (inlet + outlet)
    if quantity in ("drop", "difference"):
        if len(target_patches) < 2:
            # Try to auto-detect inlet and outlet from patch configs
            inlet_patch = None
            outlet_patch = None
            for pname in available_patches:
                pname_lower = pname.lower()
                if "inlet" in pname_lower or "in" == pname_lower:
                    inlet_patch = pname
                elif "outlet" in pname_lower or "out" == pname_lower:
                    outlet_patch = pname
            if inlet_patch and outlet_patch:
                target_patches = [inlet_patch, outlet_patch]
            else:
                return {
                    "error": (
                        f"Need at least 2 patches for {quantity} computation. "
                        f"Available patches: {sorted(available_patches)}. "
                        f"Specify patches='inlet,outlet' explicitly."
                    ),
                }

    # Build chart data
    rows: list[dict[str, Any]] = []
    line_names: list[str] = []

    if quantity in ("drop", "difference"):
        # Compute difference: first patch − second patch for each field
        p1, p2 = target_patches[0], target_patches[1]
        for fname in requested_fields:
            line_names.append(f"{fname}_drop ({p1}−{p2})")

        for step in snap.sim_progress:
            pv = step.get("patch_values") or step.get("patchValues")
            if not pv or not isinstance(pv, dict):
                continue
            xv = step.get("sim_time") if has_sim_time else step.get("iteration", 0)
            row: dict[str, Any] = {x_key: xv}
            p1_data = pv.get(p1, {})
            p2_data = pv.get(p2, {})
            for fname in requested_fields:
                v1 = p1_data.get(fname)
                v2 = p2_data.get(fname)
                if v1 is not None and v2 is not None:
                    line_key = f"{fname}_drop ({p1}−{p2})"
                    row[line_key] = v1 - v2
            if len(row) > 1:
                rows.append(row)
    else:
        # Plot raw patch-averaged values per patch per field
        for fname in requested_fields:
            for pname in target_patches:
                line_names.append(f"{fname}_{pname}")

        for step in snap.sim_progress:
            pv = step.get("patch_values") or step.get("patchValues")
            if not pv or not isinstance(pv, dict):
                continue
            xv = step.get("sim_time") if has_sim_time else step.get("iteration", 0)
            row = {x_key: xv}
            for fname in requested_fields:
                for pname in target_patches:
                    val = (pv.get(pname) or {}).get(fname)
                    if val is not None:
                        row[f"{fname}_{pname}"] = val
            if len(row) > 1:
                rows.append(row)

    if not rows:
        return {
            "error": (
                f"No patch-averaged data found for fields '{fields_arg}' on patches "
                f"{target_patches}. Available patches: {sorted(available_patches)}."
            ),
        }

    # Downsample to ≤ 300 points
    MAX_PTS = 300
    if len(rows) > MAX_PTS:
        step_size = len(rows) / MAX_PTS
        indices = {0, len(rows) - 1}
        indices |= {int(i * step_size) for i in range(1, MAX_PTS - 1)}
        rows = [rows[i] for i in sorted(indices)]

    # Filter out lines that have no data
    line_names = [ln for ln in line_names if any(ln in r for r in rows)]

    unit_parts = [_FIELD_VALUE_UNITS.get(f, "") for f in requested_fields if _FIELD_VALUE_UNITS.get(f)]
    y_label = ", ".join(dict.fromkeys(unit_parts)) if unit_parts else "Value"

    if not chart_title:
        if quantity in ("drop", "difference"):
            chart_title = f"{', '.join(requested_fields)} drop ({target_patches[0]} − {target_patches[1]})"
        else:
            chart_title = f"Patch-averaged {', '.join(requested_fields)} over {x_label}"

    # Compute summary stats for the last iteration
    summary: dict[str, Any] = {}
    if rows:
        last = rows[-1]
        for ln in line_names:
            if ln in last:
                summary[ln] = last[ln]

    result: dict[str, Any] = {
        "patches": target_patches,
        "fields_plotted": requested_fields,
        "quantity": quantity,
        "total_data_points": len(rows),
        "last_values": summary,
        "chart": {
            "type": "line",
            "title": chart_title,
            "xKey": x_key,
            "xLabel": x_label,
            "yLabel": y_label,
            "yScale": "auto",
            "lines": line_names,
            "data": rows,
        },
    }

    return result


def plot_volume_values(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Plot domain-wide volume-averaged or volume-integrated field values over time.

    Uses volFieldValue data (volumeIntegrals / volume_integrals) recorded each
    iteration to show how volume-averaged pressure, temperature, or liquid
    volume evolve throughout the run.
    """
    fields_arg = args.get("fields", "p")
    chart_title: str = args.get("title", "")

    if isinstance(fields_arg, list):
        requested = [f.strip() for f in fields_arg if isinstance(f, str) and f.strip()]
    else:
        requested = [f.strip() for f in str(fields_arg).split(",") if f.strip()]
    if not requested:
        return {"error": "No fields specified. Use e.g. fields='p' or fields='p,T'."}

    has_sim_time = any(
        step.get("sim_time") is not None for step in snap.sim_progress[:5]
    )
    x_key = "sim_time" if has_sim_time else "iteration"
    x_label = "Simulation Time (s)" if has_sim_time else "Iteration"

    # Build rows from volume_integrals data
    x_to_row: dict[Any, dict[str, Any]] = {}
    found_fields: set[str] = set()
    operations: dict[str, str] = {}  # field → operation name

    for step in snap.sim_progress:
        vi = step.get("volume_integrals") or step.get("volumeIntegrals")
        if not vi or not isinstance(vi, dict):
            continue

        xv = step.get("sim_time") if has_sim_time else step.get("iteration", 0)
        if xv not in x_to_row:
            x_to_row[xv] = {x_key: xv}
        row = x_to_row[xv]

        for fname in requested:
            fdata = vi.get(fname)
            if not fdata or not isinstance(fdata, dict):
                continue

            found_fields.add(fname)
            val = fdata.get("value")
            op = fdata.get("operation", "volAverage")
            operations[fname] = op

            if val is not None:
                row[fname] = val

    if not found_fields:
        return {
            "error": (
                f"No volume-integrated data found for '{fields_arg}'. "
                "Volume field data (volFieldValue) was not recorded during this run. "
                "Try 'plot_patch_values' for patch-averaged values, or "
                "'plot_field_values' for global min/max."
            ),
        }

    all_rows = [x_to_row[xv] for xv in sorted(x_to_row.keys())]
    all_rows = [r for r in all_rows if len(r) > 1]

    # Downsample to <= 300 points
    MAX_PTS = 300
    if len(all_rows) > MAX_PTS:
        step_size = len(all_rows) / MAX_PTS
        indices = {0, len(all_rows) - 1}
        indices |= {int(i * step_size) for i in range(1, MAX_PTS - 1)}
        all_rows = [all_rows[i] for i in sorted(indices)]

    line_names = [f for f in requested if f in found_fields]

    unit_parts = [_FIELD_VALUE_UNITS.get(f, "") for f in found_fields if _FIELD_VALUE_UNITS.get(f)]
    y_label = ", ".join(dict.fromkeys(unit_parts)) if unit_parts else "Value"

    # Build descriptive title from operations
    if not chart_title:
        op_labels = []
        for f in line_names:
            op = operations.get(f, "volAverage")
            label = "Volume-averaged" if op == "volAverage" else "Volume-integrated"
            op_labels.append(f"{label} {f}")
        chart_title = f"{', '.join(op_labels)} over {x_label}"

    # Summary stats for the last iteration
    summary: dict[str, Any] = {}
    if all_rows:
        last = all_rows[-1]
        for ln in line_names:
            if ln in last:
                summary[ln] = last[ln]

    return {
        "fields_plotted": list(found_fields),
        "operations": operations,
        "total_data_points": len(all_rows),
        "last_values": summary,
        "chart": {
            "type": "line",
            "title": chart_title,
            "xKey": x_key,
            "xLabel": x_label,
            "yLabel": y_label,
            "yScale": "auto",
            "lines": line_names,
            "data": all_rows,
        },
    }


async def compare_runs(args: dict[str, Any], snap: SimulationSnapshot) -> dict[str, Any]:
    """Compare simulation data across multiple runs as an interactive chart.

    Fetches progress for each run on-demand (not preloaded) to avoid
    slowing down normal chat turns.
    """
    from simd_agent.chat.db import fetch_sim_progress_full

    fields_arg = args.get("fields", "p")
    data_type: str = args.get("data_type", "residuals")  # residuals or field_values
    metric: str = args.get("metric", "max")  # for field_values: min, max, both

    if len(snap.all_runs) < 2:
        return {
            "error": "Only one run available for this simulation. "
                     "Run the simulation again to compare across runs.",
            "total_runs": len(snap.all_runs),
        }

    if isinstance(fields_arg, list):
        requested = [f.strip() for f in fields_arg if isinstance(f, str) and f.strip()]
    else:
        requested = [f.strip() for f in str(fields_arg).split(",") if f.strip()]
    if not requested:
        return {"error": "No fields specified."}

    # Limit to most recent 5 runs for performance
    runs_to_compare = snap.all_runs[-5:]

    # Determine x-axis and residual key
    transient = _is_transient(snap)
    residual_key = "final" if transient else "initial"

    # Fetch progress for each run and build per-run line data
    all_rows: list[dict[str, Any]] = []
    line_names: list[str] = []

    for run_idx, run_meta in enumerate(runs_to_compare, start=1):
        run_id = str(run_meta.get("id", ""))
        run_label = run_meta.get("label") or f"Run {run_idx}"
        solver = run_meta.get("solver") or ""
        if solver:
            run_label = f"{run_label} ({solver})"

        try:
            progress = await fetch_sim_progress_full(run_id, limit=2000)
        except Exception as exc:
            logger.warning(f"[compare_runs] Failed to fetch progress for run {run_id}: {exc}")
            continue

        if not progress:
            continue

        has_sim_time = any(
            s.get("sim_time") is not None for s in progress[:5]
        )
        x_key = "sim_time" if has_sim_time else "iteration"

        for fname in requested:
            line_key = f"{run_label} — {fname}"
            has_data = False

            for step in progress:
                xv = step.get("sim_time") if has_sim_time else step.get("iteration", 0)

                if data_type == "field_values":
                    fr = step.get("field_ranges") or step.get("fieldRanges")
                    if not fr or not isinstance(fr, dict):
                        continue
                    fdata = fr.get(fname)
                    if not fdata or not isinstance(fdata, dict):
                        continue
                    val = fdata.get(metric, fdata.get("max"))
                    if val is not None:
                        all_rows.append({x_key: xv, line_key: val})
                        has_data = True
                else:  # residuals
                    res = step.get("residuals", {}).get(fname)
                    if not res:
                        continue
                    val = res.get(residual_key) or res.get("initial") or res.get("final")
                    if val is not None and val > 0:
                        all_rows.append({x_key: xv, line_key: val})
                        has_data = True

            if has_data:
                line_names.append(line_key)

    if not line_names:
        return {
            "error": f"No {data_type} data found for fields '{fields_arg}' across runs.",
            "runs_checked": len(runs_to_compare),
        }

    # Merge rows by x-value
    x_key = "sim_time" if any("sim_time" in r for r in all_rows) else "iteration"
    x_label = "Simulation Time (s)" if x_key == "sim_time" else "Iteration"

    merged: dict[Any, dict[str, Any]] = {}
    for row in all_rows:
        xv = row.get(x_key)
        if xv not in merged:
            merged[xv] = {x_key: xv}
        for k, v in row.items():
            if k != x_key:
                merged[xv][k] = v

    chart_rows = [merged[xv] for xv in sorted(merged.keys())]

    # Downsample to ≤ 300 points
    MAX_PTS = 300
    if len(chart_rows) > MAX_PTS:
        step_size = len(chart_rows) / MAX_PTS
        indices = {0, len(chart_rows) - 1}
        indices |= {int(i * step_size) for i in range(1, MAX_PTS - 1)}
        chart_rows = [chart_rows[i] for i in sorted(indices)]

    use_log = data_type == "residuals"
    y_label = (
        f"{'Final' if transient else 'Initial'} Residual (log scale)"
        if use_log
        else _FIELD_VALUE_UNITS.get(requested[0], "Value")
    )

    return {
        "runs_compared": len(runs_to_compare),
        "fields": requested,
        "data_type": data_type,
        "chart": {
            "type": "line",
            "title": f"{', '.join(requested)} — Cross-Run Comparison",
            "xKey": x_key,
            "xLabel": x_label,
            "yLabel": y_label,
            "yScale": "log" if use_log else "auto",
            "lines": line_names,
            "data": chart_rows,
        },
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
    "query_simulation_results": query_simulation_results,
    "plot_field_over_iterations": plot_field_over_iterations,
    "plot_field_values": plot_field_values,
    "plot_patch_values": plot_patch_values,
    "plot_volume_values": plot_volume_values,
    "compare_runs": compare_runs,
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
                "Generate a simulation report (markdown + structured data) that the "
                "frontend will render as a PDF. Call this whenever the user asks to generate "
                "a report, export results, download a PDF, or get a complete simulation summary. "
                "Two report types are available: 'standard' (simplified plain-language summary "
                "for non-CFD engineers) and 'expert' (full engineering report with equations, "
                "schemes, mesh quality, and residual tables). Default is 'standard'. "
                "Use 'focus' only to control which 3D field screenshots are prioritized. "
                "IMPORTANT: Do NOT use any emoji."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "report_type": types.Schema(
                        type="STRING",
                        description=(
                            "Type of report to generate. 'standard' = simplified plain-language "
                            "summary with insights (default). 'expert' = full engineering report "
                            "with governing equations, discretization, mesh quality, residual tables."
                        ),
                        enum=["standard", "expert"],
                    ),
                    "focus": types.Schema(
                        type="STRING",
                        description=(
                            "Optional focus for prioritizing field screenshots in the report. "
                            "Examples: 'pressure', 'velocity', 'temperature'. "
                            "The report structure is always complete regardless of focus."
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
            name="query_simulation_results",
            description=(
                "Return a comprehensive overview of the simulation results: all VTK field "
                "ranges, boundary condition values, derived quantities (pressure drop, "
                "velocity range, Reynolds number), convergence summary, solver/physics "
                "context, fluid properties, and mesh summary. Use this as the FIRST tool "
                "call whenever the user asks a broad question about results, wants an "
                "overview, asks 'how did the simulation go', or asks about multiple fields "
                "at once. Also use it when the user asks about recommendations, because "
                "the convergence and derived data help you reason about what to suggest."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "question": types.Schema(
                        type="STRING",
                        description=(
                            "The user's question or intent, in a few words. "
                            "This is passed through to the result so you can "
                            "tailor your response."
                        ),
                    ),
                },
                required=["question"],
            ),
        ),
        types.FunctionDeclaration(
            name="plot_field_over_iterations",
            description=(
                "Plot residual fields, Courant number, or continuity error across all "
                "iterations as an interactive chart. Use this whenever the user asks to "
                "'plot', 'chart', or 'graph' any quantity over time/iterations. "
                "Supports residual field names (Ux, Uy, Uz, p, k, omega, epsilon, T, h…), "
                "'courant' for Courant number, 'continuity' for continuity error. "
                "Multiple fields can be comma-separated."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(
                        type="STRING",
                        description=(
                            "Comma-separated field names to plot. "
                            "Examples: 'p', 'Ux,Uy,Uz', 'k,omega', 'courant', "
                            "'p,continuity'. Use residual field names from sim_progress."
                        ),
                    ),
                    "title": types.Schema(
                        type="STRING",
                        description="Optional chart title.",
                    ),
                },
                required=["fields"],
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
        types.FunctionDeclaration(
            name="plot_field_values",
            description=(
                "Plot actual physical field values (pressure, temperature, turbulence "
                "quantities, etc.) over simulation time as an INTERACTIVE CHART rendered "
                "directly in the chat. Uses fieldMinMax data recorded each iteration to "
                "show how field min/max evolve throughout the run. "
                "This is DIFFERENT from residual plots — residuals show solver convergence "
                "(how well equations are solved), while this shows the physical values "
                "themselves (e.g. actual pressure in Pa, temperature in K). "
                "Use this when the user asks: 'plot pressure', 'pressure profile', "
                "'temperature over time', 'how does pressure change', 'field evolution', "
                "'pressure drop over time', 'show me the pressure trend', etc. "
                "Also use when has_field_value_data is true in the context."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(
                        type="STRING",
                        description=(
                            "Comma-separated field names to plot. "
                            "Examples: 'p', 'T', 'k,omega', 'p,T'. "
                            "Field names must match those in fieldMinMax output."
                        ),
                    ),
                    "metric": types.Schema(
                        type="STRING",
                        description=(
                            "What to plot: 'min' (minimum), 'max' (maximum), "
                            "'both' (min and max lines), or 'range' (max-min). "
                            "Default: 'both'."
                        ),
                    ),
                    "title": types.Schema(
                        type="STRING",
                        description="Optional chart title.",
                    ),
                },
                required=["fields"],
            ),
        ),
        types.FunctionDeclaration(
            name="plot_patch_values",
            description=(
                "Plot patch-averaged values (from surfaceFieldValue function objects) "
                "over simulation iterations as an INTERACTIVE CHART. Shows how area-averaged "
                "quantities at specific patches (inlet, outlet, walls) evolve over time. "
                "Can compute PRESSURE DROP (inlet − outlet), TEMPERATURE DROP, or show "
                "raw patch-averaged values at each boundary. "
                "Use when the user asks: 'pressure drop', 'temperature drop', "
                "'pressure at inlet vs outlet', 'boil-off rate', 'average pressure at outlet', "
                "'how does temperature change at the wall', etc. "
                "This is DIFFERENT from plot_field_values (which shows global min/max) — "
                "this shows patch-specific averaged values."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(
                        type="STRING",
                        description=(
                            "Comma-separated field names. Examples: 'p', 'T', 'p,T'. "
                            "These are the fields averaged over each patch face."
                        ),
                    ),
                    "patches": types.Schema(
                        type="STRING",
                        description=(
                            "Comma-separated patch names. Examples: 'inlet,outlet', "
                            "'wall'. Leave empty to include all available patches."
                        ),
                    ),
                    "quantity": types.Schema(
                        type="STRING",
                        description=(
                            "'values' to plot raw patch-averaged values per patch, "
                            "'drop' or 'difference' to compute first_patch − second_patch "
                            "(e.g. pressure drop = inlet p − outlet p). "
                            "Default: 'values'."
                        ),
                    ),
                    "title": types.Schema(
                        type="STRING",
                        description="Optional chart title.",
                    ),
                },
                required=["fields"],
            ),
        ),
        types.FunctionDeclaration(
            name="plot_volume_values",
            description=(
                "Plot domain-wide volume-averaged or volume-integrated field values "
                "over simulation time as an INTERACTIVE CHART. Shows how volume-averaged "
                "pressure, temperature, or liquid volume evolve throughout the run. "
                "Use when the user asks: 'average pressure in the domain', "
                "'volume-averaged temperature', 'liquid volume over time', "
                "'domain average', 'bulk temperature', etc. "
                "This is DIFFERENT from plot_patch_values (patch-specific averages) "
                "and plot_field_values (global min/max)."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(
                        type="STRING",
                        description=(
                            "Comma-separated field names. "
                            "Examples: 'p', 'T', 'alpha.water'."
                        ),
                    ),
                    "title": types.Schema(
                        type="STRING",
                        description="Optional chart title.",
                    ),
                },
                required=["fields"],
            ),
        ),
        types.FunctionDeclaration(
            name="compare_runs",
            description=(
                "Compare simulation data across multiple runs of the same simulation "
                "as an INTERACTIVE CHART. Each run is plotted as a separate line so the "
                "user can see how results changed between runs. "
                "Use when the user asks: 'compare runs', 'compare pressure across runs', "
                "'how did run 2 differ', 'plot all runs together', etc. "
                "Only available when multiple runs exist (check all_runs in context)."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(
                        type="STRING",
                        description=(
                            "Comma-separated field names to compare. "
                            "Examples: 'p', 'Ux,Uy,Uz', 'T'."
                        ),
                    ),
                    "data_type": types.Schema(
                        type="STRING",
                        description=(
                            "'residuals' to compare solver convergence, "
                            "'field_values' to compare actual field min/max. "
                            "Default: 'residuals'."
                        ),
                    ),
                    "metric": types.Schema(
                        type="STRING",
                        description=(
                            "For field_values: 'min', 'max', or 'both'. Default: 'max'."
                        ),
                    ),
                },
                required=["fields"],
            ),
        ),
    ]
)
