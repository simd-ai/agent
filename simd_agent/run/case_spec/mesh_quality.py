"""Mesh-quality-driven numerical decisions.

Extracts metrics from OpenFOAM checkMesh output (max non-orthogonality,
skewness, aspect ratio) and derives the tier + numerical knobs that the
deterministic fvSolution / fvSchemes builders consume:

  * ``mesh_quality_tier``         — "good" / "moderate" / "poor" / "unknown"
  * ``use_simplec``               — whether SIMPLEC's H1 correction is safe
  * ``n_non_ortho_correctors``    — extra pressure-equation passes per iter
  * ``mesh_max_non_orthogonality``, ``mesh_max_skewness``, ``mesh_max_aspect_ratio``

When no real checkMesh data is available (metrics are None / 0), returns a
conservative profile ("unknown" tier, no SIMPLEC, 1 non-ortho corrector)
that works on virtually any mesh.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.solvers.base import SolverPlugin

logger = logging.getLogger(__name__)


def _props_from_registry(solver: str) -> dict[str, Any] | None:
    """Derive the legacy _SOLVER_PROPS dict from the solver plugin registry.

    Returns None if the plugin is unavailable, so callers can fall back to
    the hardcoded _SOLVER_PROPS table.
    """
    try:
        from simd_agent.solvers import get_registry
    except Exception:
        return None
    plugin: SolverPlugin | None = get_registry().get(solver)
    if plugin is None:
        return None
    return {
        "algorithm": plugin.algorithm,
        "pressure_field": plugin.pressure_field,
        "transient": plugin.is_transient,
        "compressible": plugin.is_compressible,
        "multiphase": plugin.is_multiphase,
        "energy": "he" if plugin.supports_energy else "none",
        "needs_gravity": plugin.needs_gravity,
    }


def _mesh_quality_decisions(check_mesh: Any) -> dict[str, Any]:
    """Derive numerical strategy from real OpenFOAM checkMesh metrics.

    See module docstring for the full output schema.
    """
    defaults = {
        "use_simplec": False,
        "n_non_ortho_correctors": 1,
        "mesh_quality_tier": "unknown",
        "mesh_max_non_orthogonality": None,
        "mesh_max_skewness": None,
        "mesh_max_aspect_ratio": None,
    }

    if check_mesh is None:
        logger.info(
            "[MESH_QUALITY] check_mesh is None → tier='unknown', "
            "non-GAMG fallback (conservative)"
        )
        return defaults

    # Extract metrics — support both dict and Pydantic model
    if isinstance(check_mesh, dict):
        non_ortho = check_mesh.get("max_non_orthogonality")
        skew = check_mesh.get("max_skewness")
        aspect = check_mesh.get("max_aspect_ratio")
    else:
        non_ortho = getattr(check_mesh, "max_non_orthogonality", None)
        skew = getattr(check_mesh, "max_skewness", None)
        aspect = getattr(check_mesh, "max_aspect_ratio", None)

    logger.info(
        f"[MESH_QUALITY] Raw metrics: non_ortho={non_ortho}, "
        f"skew={skew}, aspect={aspect}"
    )

    # If no real data (all None or all zero), return conservative defaults
    if not non_ortho and not skew and not aspect:
        logger.info(
            "[MESH_QUALITY] All metrics are None/0 → tier='unknown', "
            "non-GAMG fallback (conservative)"
        )
        return defaults

    non_ortho = non_ortho or 0.0
    skew = skew or 0.0
    aspect = aspect or 0.0

    # ── Tier classification ──────────────────────────────────────────
    if non_ortho < 40 and skew < 0.5 and aspect < 50:
        tier = "good"
    elif non_ortho < 65 and skew < 2.0 and aspect < 100:
        tier = "moderate"
    else:
        tier = "poor"

    # ── SIMPLEC decision ─────────────────────────────────────────────
    # SIMPLEC modifies the pressure equation via H1 correction. On meshes
    # with high non-orthogonality, the modified matrix creates near-zero
    # entries in GAMG coarse levels → SIGFPE in GAMGSolver::scale.
    use_simplec = non_ortho < 65 and skew < 0.8 and aspect < 100

    # ── Non-orthogonal correctors ────────────────────────────────────
    if non_ortho < 5:
        n_correctors = 0
    elif non_ortho < 40:
        n_correctors = 1
    else:
        n_correctors = 2

    _use_gamg = tier == "good" or (tier == "moderate" and non_ortho < 50)
    logger.info(
        f"[MESH_QUALITY] Decision: tier='{tier}', non_ortho={non_ortho:.1f}°, "
        f"skew={skew:.2f}, aspect={aspect:.1f} → "
        f"GAMG={'YES' if _use_gamg else 'NO (PBiCGStab)'}, "
        f"SIMPLEC={'YES' if use_simplec else 'NO'}, "
        f"nNonOrthoCorr={n_correctors}"
    )

    return {
        "use_simplec": use_simplec,
        "n_non_ortho_correctors": n_correctors,
        "mesh_quality_tier": tier,
        "mesh_max_non_orthogonality": non_ortho,
        "mesh_max_skewness": skew,
        "mesh_max_aspect_ratio": aspect,
    }
