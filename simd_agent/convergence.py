# simd_agent/convergence.py
"""CFD convergence assessment — single source of truth for the entire platform.

Computes convergence status from sim_progress data using CFD-correct logic:
- Steady-state (SIMPLE): uses INITIAL residuals (measure solution change)
- Transient (PIMPLE): uses FINAL residuals (inner-loop convergence quality)
- Field-specific convergence thresholds (pressure vs velocity vs turbulence)
- Oscillation detection via log-residual variance
- Continuity error trend assessment
- Courant number checks for transient runs
- Pressure convergence required for overall "converged" status

Used by:
  - orchestration.py  — computed during sim_progress streaming, persisted on run
  - chat/tools.py     — injected into snapshot for LLM context
  - frontend          — receives assessment via WebSocket events
"""

from __future__ import annotations

import math
from typing import Any, Literal

# ── Types ──────────────────────────────────────────────────────────────────

ConvergenceStatus = Literal[
    "converged",
    "converging",
    "oscillating",
    "stalling",
    "diverging",
]

SolverCategory = Literal["steady", "transient"]


# ── Field-specific convergence thresholds ──────────────────────────────────
# Based on standard CFD engineering practice:
#   - Pressure is the hardest to converge and most critical
#   - Velocity components converge tighter than pressure
#   - Turbulence quantities often plateau at higher levels — acceptable
#   - Energy (T/h) needs tight convergence for thermal accuracy

_FIELD_THRESHOLDS: dict[str, float] = {
    # Pressure
    "p":     1e-4,
    "p_rgh": 1e-4,
    # Velocity
    "Ux": 1e-5,
    "Uy": 1e-5,
    "Uz": 1e-5,
    # Turbulence — plateau at 1e-3 is normal and acceptable
    "k":       1e-3,
    "omega":   1e-3,
    "epsilon": 1e-3,
    "nut":     1e-3,
    "nuTilda": 1e-3,
    # Energy
    "T":      1e-6,
    "h":      1e-6,
    "e":      1e-6,
    "alphat": 1e-3,
    # Multiphase
    "alpha.water":  1e-4,
    "alpha.liquid": 1e-4,
}

_DEFAULT_THRESHOLD = 1e-4

# Fields whose convergence is REQUIRED for overall "converged" status.
# If any of these are not converged, the simulation is not converged —
# regardless of how many other fields are fine.
_CRITICAL_FIELDS = {"p", "p_rgh", "Ux", "Uy", "Uz"}


def _bare_field_name(field: str) -> str:
    """Strip the optional ``<region>:`` namespace from a residual key.

    Multi-region (CHT) cases ship residuals under namespaced keys like
    ``innerFluid:Ux`` / ``wall:h`` — the underlying physics field is
    just ``Ux`` / ``h``, so any lookup into the threshold table or
    critical-fields set must operate on the bare name.  Single-region
    fields have no colon and pass through unchanged.
    """
    return field.split(":", 1)[1] if ":" in field else field


def _threshold_for(field: str) -> float:
    """Return the convergence threshold for a field name.

    Multi-region-safe: ``innerFluid:Ux`` → looks up ``Ux`` threshold
    (1e-5), not the default 1e-4 fallback.  Without this fix the
    threshold for every CHT field was 1e-4 — much looser than physics
    requires, especially for velocity (Ux/Uy/Uz: 1e-5) and energy
    (T/h: 1e-6).
    """
    return _FIELD_THRESHOLDS.get(_bare_field_name(field), _DEFAULT_THRESHOLD)


# ── Helpers ────────────────────────────────────────────────────────────────

def _linear_slope(ys: list[float]) -> float:
    """Least-squares slope for evenly-spaced data (x = 0, 1, 2, ...)."""
    n = len(ys)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(ys) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(ys))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def _variance(ys: list[float]) -> float:
    """Population variance."""
    n = len(ys)
    if n < 2:
        return 0.0
    mean = sum(ys) / n
    return sum((y - mean) ** 2 for y in ys) / n


# ── Per-field assessment ───────────────────────────────────────────────────

def _assess_field(
    residuals: list[float],
    field: str,
) -> dict[str, Any]:
    """Assess convergence for a single field.

    Args:
        residuals: List of residual values (already selected: initial for
                   steady-state, final for transient) across all time steps.
        field: The field name (e.g. "Ux", "p", "k").

    Returns:
        Dict with field metrics and status.
    """
    if len(residuals) < 2:
        return {
            "field": field,
            "status": "converging",
            "firstResidual": residuals[0] if residuals else 0,
            "lastResidual": residuals[-1] if residuals else 0,
            "ordersDrop": 0.0,
            "recentSlope": 0.0,
            "threshold": _threshold_for(field),
        }

    first = residuals[0]
    last = residuals[-1]

    # Orders of magnitude drop (positive = improvement)
    orders_drop = 0.0
    if first > 0 and last > 0:
        orders_drop = math.log10(first) - math.log10(last)

    # Analysis window: last 20% of data, minimum 5 points
    window_size = max(5, len(residuals) // 5)
    window = residuals[-window_size:]

    # Slope of log10(residual) in the window
    log_window = [math.log10(v) if v > 0 else -20.0 for v in window]
    slope = _linear_slope(log_window)

    # Oscillation: variance of log10(residual) in the window
    # High variance + near-zero slope = oscillating
    log_variance = _variance(log_window)

    threshold = _threshold_for(field)

    # ── Status decision ────────────────────────────────────────────────
    status: ConvergenceStatus

    if last <= threshold and slope <= 0.005:
        # Below threshold and not rising → converged
        status = "converged"
    elif slope > 0.02:
        # Clear upward trend → diverging
        status = "diverging"
    elif log_variance > 0.15 and abs(slope) < 0.01:
        # High variance but no net trend → oscillating
        # (typical for pressure in SIMPLE with aggressive relaxation)
        status = "oscillating"
    elif abs(slope) < 0.005 and last > threshold:
        # Flat but above threshold → stalling
        status = "stalling"
    else:
        # Slope is negative (dropping) → still converging
        status = "converging"

    return {
        "field": field,
        "status": status,
        "firstResidual": first,
        "lastResidual": last,
        "ordersDrop": round(orders_drop, 2),
        "recentSlope": round(slope, 4),
        "threshold": threshold,
        "oscillationVariance": round(log_variance, 4),
    }


# ── Continuity error assessment ────────────────────────────────────────────

def _assess_continuity(
    progress: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Assess global continuity error trend.

    Growing continuity error means mass is not being conserved — a strong
    divergence signal even when field residuals look OK.
    """
    values: list[float] = []
    for step in progress:
        cont = step.get("continuity")
        if isinstance(cont, dict):
            g = cont.get("global")
            if g is not None:
                try:
                    values.append(abs(float(g)))
                except (TypeError, ValueError):
                    pass

    if len(values) < 5:
        return None

    # Slope of the absolute global continuity over the last 30%
    window_size = max(5, len(values) // 3)
    window = values[-window_size:]
    log_window = [math.log10(v) if v > 0 else -20.0 for v in window]
    slope = _linear_slope(log_window)

    last_val = values[-1]

    status: ConvergenceStatus
    if last_val < 1e-6 and slope <= 0.005:
        status = "converged"
    elif slope > 0.02:
        status = "diverging"
    elif slope > -0.003 and last_val > 1e-4:
        status = "stalling"
    else:
        status = "converging"

    return {
        "lastValue": last_val,
        "recentSlope": round(slope, 4),
        "status": status,
    }


# ── Courant number assessment (transient only) ────────────────────────────

def _assess_courant(
    progress: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Assess Courant number for transient simulations.

    High Courant numbers (> 1 for explicit, > 50 for PIMPLE) indicate
    potential numerical instability.
    """
    max_courant: float | None = None
    mean_courant: float | None = None

    for step in progress[-20:]:  # check recent steps
        co = step.get("courant")
        if isinstance(co, dict):
            try:
                mx = float(co.get("max", 0))
                mn = float(co.get("mean", 0))
                if max_courant is None or mx > max_courant:
                    max_courant = mx
                mean_courant = mn  # keep the last one
            except (TypeError, ValueError):
                pass

    if max_courant is None:
        return None

    if max_courant > 50:
        level = "critical"
    elif max_courant > 10:
        level = "warning"
    elif max_courant > 1:
        level = "acceptable"
    else:
        level = "good"

    return {
        "recentMax": round(max_courant, 3),
        "recentMean": round(mean_courant, 3) if mean_courant is not None else None,
        "level": level,
    }


# ── Auto-detect solver category from progress data ────────────────────────

def _infer_solver_category(progress: list[dict[str, Any]]) -> SolverCategory:
    """Infer whether the simulation is steady-state or transient from data.

    Heuristic: if simTime values are fractional (not integer-like) and
    increasing, it's transient. Otherwise steady-state.
    """
    if len(progress) < 3:
        return "steady"

    times: list[float] = []
    for step in progress[:10]:
        t = step.get("simTime", step.get("sim_time"))
        if t is not None:
            try:
                times.append(float(t))
            except (TypeError, ValueError):
                pass

    if len(times) < 3:
        return "steady"

    # Check if times are fractional (transient) or integer-like (steady)
    # In steady-state OF, simTime == iteration (1, 2, 3, ...)
    # In transient, simTime is fractional (0.001, 0.002, ...)
    iters: list[int] = []
    for step in progress[:10]:
        it = step.get("iteration")
        if it is not None:
            iters.append(int(it))

    if iters and times:
        # If simTime differs from iteration number, it's transient
        mismatches = sum(
            1 for t, i in zip(times, iters)
            if abs(t - i) > 0.01
        )
        if mismatches > len(times) // 2:
            return "transient"

    return "steady"


# ── Main assessment entry point ────────────────────────────────────────────

def compute_convergence(
    progress: list[dict[str, Any]],
    solver_category: SolverCategory | None = None,
    is_transient: bool | None = None,
) -> dict[str, Any] | None:
    """Compute a full convergence assessment from sim_progress data.

    Args:
        progress: List of sim_progress dicts (same format as stored in DB /
                  streamed to frontend). Each has: iteration, simTime,
                  residuals: {field: {initial, final, iters}}, courant,
                  continuity.
        solver_category: "steady" or "transient". If None, auto-detected
                         from the progress data.
        is_transient: Deprecated alias for solver_category. If provided and
                      solver_category is None, maps True→"transient".

    Returns:
        A convergence assessment dict, or None if insufficient data (< 5 steps).

    The returned dict is the single source of truth consumed by:
      - Frontend (SummaryTab, LiveTab) — display only
      - Chat system — injected into LLM context
      - Run finalization — persisted in final_result
    """
    if len(progress) < 5:
        return None

    # Resolve solver category
    if solver_category is None:
        if is_transient is True:
            solver_category = "transient"
        elif is_transient is False:
            solver_category = "steady"
        else:
            solver_category = _infer_solver_category(progress)

    # Determine which residual to use
    residual_key = "final" if solver_category == "transient" else "initial"

    # Collect field names from the last step
    last_step = progress[-1]
    field_names: list[str] = last_step.get("fields", [])
    if not field_names:
        residuals_dict = last_step.get("residuals", {})
        field_names = list(residuals_dict.keys())

    if not field_names:
        return None

    # ── Per-field assessment ────────────────────────────────────────────
    field_assessments: list[dict[str, Any]] = []

    for field in field_names:
        values: list[float] = []
        for step in progress:
            r = step.get("residuals", {}).get(field)
            if isinstance(r, dict):
                v = r.get(residual_key)
                # Fall back to the other residual type if preferred is absent
                if v is None:
                    v = r.get("initial" if residual_key == "final" else "final")
                if v is not None and v > 0:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass

        if len(values) < 2:
            continue

        assessment = _assess_field(values, field)
        field_assessments.append(assessment)

    if not field_assessments:
        return None

    # ── Continuity assessment ──────────────────────────────────────────
    continuity = _assess_continuity(progress)

    # ── Courant assessment (transient only) ────────────────────────────
    courant = _assess_courant(progress) if solver_category == "transient" else None

    # ── Overall status ─────────────────────────────────────────────────
    overall = _compute_overall_status(field_assessments, continuity, courant)

    # ── Summary statistics ─────────────────────────────────────────────
    converged_count = sum(1 for f in field_assessments if f["status"] == "converged")
    converged_fraction = converged_count / len(field_assessments) if field_assessments else 0

    # ── Actionable recommendations ──────────────────────────────────────
    recommendations = _generate_recommendations(
        overall, field_assessments, continuity, courant, solver_category,
    )

    return {
        "status": overall,
        "solverCategory": solver_category,
        "residualType": residual_key,
        "fields": field_assessments,
        "totalIterations": len(progress),
        "convergedFraction": round(converged_fraction, 3),
        "convergedCount": converged_count,
        "totalFields": len(field_assessments),
        "continuity": continuity,
        "courant": courant,
        "recommendations": recommendations,
    }


def _compute_overall_status(
    fields: list[dict[str, Any]],
    continuity: dict[str, Any] | None,
    courant: dict[str, Any] | None,
) -> ConvergenceStatus:
    """Determine overall convergence status from component assessments.

    Rules (in priority order):
    1. If ANY field is diverging OR continuity is diverging → diverging
    2. If Courant is critical → diverging
    3. If ALL fields converged AND continuity OK → converged
       (requires all critical fields present to actually be converged)
    4. If critical fields (p, velocity) are oscillating → oscillating
    5. If all fields are stalling or converged (but not all converged) → stalling
    6. Otherwise → converging
    """
    statuses = {f["field"]: f["status"] for f in fields}
    all_statuses = list(statuses.values())

    # Multi-region (CHT) cases ship field keys as ``<region>:<field>``.
    # The critical-field check below operates on bare physics names
    # (p, p_rgh, Ux, Uy, Uz) — without this mapping the check
    # vacuously passes (``critical_present = []``) and the overall
    # status falls through whatever fields happen to converge first,
    # which is wrong for steady CHT where pressure/momentum convergence
    # is the actual gate.
    bare_statuses: dict[str, list[str]] = {}
    for namespaced, status in statuses.items():
        bare_statuses.setdefault(_bare_field_name(namespaced), []).append(status)

    # 1. Any divergence → overall diverging
    if "diverging" in all_statuses:
        return "diverging"
    if continuity and continuity.get("status") == "diverging":
        return "diverging"

    # 2. Critical Courant → diverging
    if courant and courant.get("level") == "critical":
        return "diverging"

    # 3. All converged (with critical fields check)
    # A critical physics field is "converged overall" only when EVERY
    # region's copy of it is converged — innerFluid:Ux converged but
    # outerFluid:Ux still oscillating still counts as not-converged.
    critical_present = [f for f in _CRITICAL_FIELDS if f in bare_statuses]
    critical_converged = all(
        all(s == "converged" for s in bare_statuses[f])
        for f in critical_present
    )
    all_converged = all(s == "converged" for s in all_statuses)
    continuity_ok = continuity is None or continuity.get("status") in ("converged", "converging")

    if all_converged and critical_converged and continuity_ok:
        return "converged"

    # 4. Critical fields oscillating → oscillating
    # Any region's copy of a critical field oscillating counts.
    critical_oscillating = any(
        "oscillating" in bare_statuses.get(f, [])
        for f in _CRITICAL_FIELDS
    )
    if critical_oscillating:
        return "oscillating"

    # 5. All stalling or converged → stalling
    if all(s in ("stalling", "converged") for s in all_statuses):
        return "stalling"

    # 6. Default: converging
    return "converging"


# ── Recommendation types ──────────────────────────────────────────────────

RecommendationType = Literal[
    "relaxation",
    "time_step",
    "mesh_refinement",
    "more_iterations",
    "scheme_change",
]

RecommendationSeverity = Literal["info", "warning", "critical"]


# ── Default relaxation factor presets ─────────────────────────────────────
# Used to compute "reduce by X" recommendations.  The values here are the
# *conservative* targets — if the user's current factors are already at or
# below these, we don't recommend further reduction.

_RELAXATION_CONSERVATIVE: dict[str, float] = {
    "p": 0.15,
    "p_rgh": 0.15,
    "U": 0.5,
    "k": 0.4,
    "omega": 0.4,
    "epsilon": 0.4,
    "h": 0.2,
    "rho": 0.05,
}

_RELAXATION_DEFAULT: dict[str, float] = {
    "p": 0.3,
    "p_rgh": 0.3,
    "U": 0.7,
    "k": 0.7,
    "omega": 0.7,
    "epsilon": 0.7,
    "h": 0.5,
    "rho": 0.1,
}


def _field_to_relax_key(field: str) -> str:
    """Map a convergence field name to the relaxation factor key.

    Convergence reports Ux/Uy/Uz individually, but fvSolution uses U.

    Multi-region (CHT) cases ship the same physics fields under
    region-namespaced keys (``innerFluid:Ux``, ``wall:h``, …) — strip
    the ``<region>:`` prefix first so the same Ux/Uy/Uz → U collapse
    fires, and so the recommendation engine produces sensible
    relaxation-factor advice instead of treating ``innerFluid:Ux`` as a
    distinct never-relaxed field.
    """
    bare = field.split(":", 1)[1] if ":" in field else field
    if bare in ("Ux", "Uy", "Uz"):
        return "U"
    return bare


def _generate_recommendations(
    overall: ConvergenceStatus,
    fields: list[dict[str, Any]],
    continuity: dict[str, Any] | None,
    courant: dict[str, Any] | None,
    solver_category: SolverCategory,
) -> list[dict[str, Any]]:
    """Generate actionable recommendations based on convergence assessment.

    Returns a list of recommendation dicts, each with:
        type:        RecommendationType
        severity:    RecommendationSeverity
        title:       Short human-readable title
        description: 2-3 sentence explanation
        action:      Structured payload the frontend can send back to apply
    """
    if overall in ("converged", "converging"):
        return []

    recs: list[dict[str, Any]] = []

    osc_fields = [f for f in fields if f["status"] == "oscillating"]
    stall_fields = [f for f in fields if f["status"] == "stalling"]
    div_fields = [f for f in fields if f["status"] == "diverging"]

    # ── 1. Oscillating → reduce relaxation factors ────────────────────
    if osc_fields:
        affected = [f["field"] for f in osc_fields]
        # Build a changes dict: for each oscillating field, suggest the
        # conservative preset.  Also include related fields (if p oscillates,
        # tighten U too).
        changes: dict[str, float] = {}
        for name in affected:
            relax_key = _field_to_relax_key(name)
            if relax_key in _RELAXATION_CONSERVATIVE:
                changes[relax_key] = _RELAXATION_CONSERVATIVE[relax_key]
        # Always include p + U together since they're coupled
        if any(f in changes for f in ("p", "p_rgh")):
            changes.setdefault("U", _RELAXATION_CONSERVATIVE["U"])
        if "U" in changes:
            for pf in ("p", "p_rgh"):
                if any(fa["field"] == pf for fa in fields):
                    changes.setdefault(pf, _RELAXATION_CONSERVATIVE[pf])

        if changes:
            field_list = ", ".join(affected)
            change_list = ", ".join(f"{k}: {v}" for k, v in sorted(changes.items()))
            recs.append({
                "type": "relaxation",
                "severity": "warning",
                "title": "Adjust relaxation factors",
                "description": (
                    f"Residuals for {field_list} are oscillating without converging. "
                    f"Reducing relaxation factors will dampen these oscillations at the "
                    f"cost of slower convergence. Suggested values: {change_list}."
                ),
                "action": {
                    "type": "relaxation",
                    "changes": changes,
                },
            })

    # ── 2. Diverging → urgent scheme / relaxation / BC check ──────────
    if div_fields:
        affected = [f["field"] for f in div_fields]
        field_list = ", ".join(affected)

        # Suggest aggressive relaxation + upwind schemes
        changes = {}
        for name in affected:
            relax_key = _field_to_relax_key(name)
            if relax_key in _RELAXATION_CONSERVATIVE:
                changes[relax_key] = _RELAXATION_CONSERVATIVE[relax_key]
        # For diverging, always tighten the core fields
        for core in ("p", "p_rgh", "U"):
            if any(_field_to_relax_key(fa["field"]) == core for fa in fields):
                changes.setdefault(core, _RELAXATION_CONSERVATIVE.get(core, 0.2))

        recs.append({
            "type": "relaxation",
            "severity": "critical",
            "title": "Stabilize diverging fields",
            "description": (
                f"Residuals for {field_list} are increasing — the simulation is diverging. "
                f"Apply aggressive under-relaxation and switch to first-order upwind "
                f"schemes to stabilize, then gradually increase accuracy."
            ),
            "action": {
                "type": "relaxation",
                "changes": changes,
            },
        })

    # ── 3. High Courant (transient) → reduce time step ────────────────
    if courant and courant.get("level") in ("warning", "critical"):
        co_max = courant["recentMax"]
        severity: RecommendationSeverity = (
            "critical" if courant["level"] == "critical" else "warning"
        )
        # Target Co < 1 for safety
        reduction_factor = max(0.1, 1.0 / co_max) if co_max > 1 else 0.5
        recs.append({
            "type": "time_step",
            "severity": severity,
            "title": "Reduce time step",
            "description": (
                f"Courant number is {co_max:.1f} (target < 1.0 for stability). "
                f"Reduce deltaT by a factor of {1/reduction_factor:.0f}x or lower maxCo "
                f"to 0.5 to keep the simulation stable."
            ),
            "action": {
                "type": "time_step",
                "changes": {
                    "maxCo": 0.5,
                    "deltaT_factor": round(reduction_factor, 3),
                },
            },
        })

    # ── 4. Stalling → more iterations or mesh refinement ──────────────
    if stall_fields and not div_fields and not osc_fields:
        # _bare_field_name() handles CHT ``<region>:<field>`` namespacing —
        # without it every namespaced field reads as "not in CRITICAL"
        # (since the set has bare physics names only), wrongly classifying
        # innerFluid:Ux stalls as turbulence stalls.
        affected = [
            f for f in stall_fields
            if _bare_field_name(f["field"]) in _CRITICAL_FIELDS
        ]
        turb_stalling = [
            f for f in stall_fields
            if _bare_field_name(f["field"]) not in _CRITICAL_FIELDS
        ]

        # If only turbulence fields are stalling, that's often acceptable
        if affected:
            field_list = ", ".join(f["field"] for f in affected)
            # Compute how far from threshold the worst field is
            worst = max(affected, key=lambda f: f["lastResidual"])
            ratio = worst["lastResidual"] / worst["threshold"]

            if ratio < 10:
                # Close to convergence — just needs more iterations
                recs.append({
                    "type": "more_iterations",
                    "severity": "info",
                    "title": "Extend simulation",
                    "description": (
                        f"{field_list} plateaued within {ratio:.0f}x of target. "
                        f"Running more iterations ({2 if solver_category == 'steady' else 3}x current) "
                        f"may be sufficient to reach convergence."
                    ),
                    "action": {
                        "type": "more_iterations",
                        "changes": {
                            "iteration_multiplier": 2 if solver_category == "steady" else 3,
                        },
                    },
                })
            else:
                # Far from convergence — mesh or relaxation issue
                # Choose refinement strategy: wall refinement for turbulent
                # flows (boundary layer is the bottleneck), global for laminar
                all_field_names = {f["field"] for f in fields}
                has_turbulence = bool(all_field_names & {"k", "omega", "epsilon", "nuTilda"})
                strategy = "wall" if has_turbulence else "global"
                strategy_desc = (
                    "Refining near walls will improve boundary layer resolution."
                    if strategy == "wall"
                    else "Uniform global refinement will increase resolution everywhere."
                )

                recs.append({
                    "type": "mesh_refinement",
                    "severity": "warning",
                    "title": f"Refine mesh ({'near walls' if strategy == 'wall' else 'globally'})",
                    "description": (
                        f"{field_list} plateaued at {ratio:.0f}x above target threshold. "
                        f"The mesh may be too coarse to resolve gradients. {strategy_desc}"
                    ),
                    "action": {
                        "type": "mesh_refinement",
                        "changes": {
                            "refinement_level": 1,
                            "strategy": strategy,
                        },
                    },
                })

                # Also suggest relaxation if not already recommended
                if not any(r["type"] == "relaxation" for r in recs):
                    changes = {}
                    for f in affected:
                        relax_key = _field_to_relax_key(f["field"])
                        if relax_key in _RELAXATION_CONSERVATIVE:
                            changes[relax_key] = _RELAXATION_CONSERVATIVE[relax_key]
                    if changes:
                        recs.append({
                            "type": "relaxation",
                            "severity": "info",
                            "title": "Try adjusted relaxation",
                            "description": (
                                "If mesh refinement is not practical, try reducing relaxation "
                                "factors — this can sometimes break through a residual plateau "
                                "by allowing smaller per-iteration changes."
                            ),
                            "action": {
                                "type": "relaxation",
                                "changes": changes,
                            },
                        })

        elif turb_stalling:
            # Only turbulence stalling — usually acceptable, just note it
            field_list = ", ".join(f["field"] for f in turb_stalling)
            recs.append({
                "type": "more_iterations",
                "severity": "info",
                "title": "Turbulence fields plateaued",
                "description": (
                    f"{field_list} have plateaued, but this is common for turbulence "
                    f"quantities and usually acceptable if primary fields (p, U) are "
                    f"converged. No action required unless results look non-physical."
                ),
                "action": None,
            })

    return recs
