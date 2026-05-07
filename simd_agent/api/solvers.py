# simd_agent/api/solvers.py
"""Read-only endpoints for the solver registry.

Exposes metadata about available solvers so the frontend can:
  - Show a solver picker with capabilities
  - Validate solver names before submission
  - Display solver documentation
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from simd_agent.solvers import get_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/solvers", tags=["solvers"])


class SolverInfo(BaseModel):
    name: str
    algorithm: str
    pressure_field: str
    is_transient: bool
    is_compressible: bool
    supports_energy: bool
    needs_gravity: bool
    is_multiphase: bool


class SolverMatch(BaseModel):
    name: str
    score: float
    reason: str
    warnings: list[str]


@router.get("")
async def list_solvers() -> list[SolverInfo]:
    """List all registered solver plugins with their capabilities."""
    registry = get_registry()
    return [SolverInfo(**info) for info in registry.solver_info()]


@router.get("/{solver_name}")
async def get_solver(solver_name: str) -> SolverInfo:
    """Get info for a specific solver."""
    registry = get_registry()
    plugin = registry.get(solver_name)
    if not plugin:
        raise HTTPException(404, f"Solver '{solver_name}' not found")
    return SolverInfo(
        name=plugin.name,
        algorithm=plugin.algorithm,
        pressure_field=plugin.pressure_field,
        is_transient=plugin.is_transient,
        is_compressible=plugin.is_compressible,
        supports_energy=plugin.supports_energy,
        needs_gravity=plugin.needs_gravity,
        is_multiphase=plugin.is_multiphase,
    )


@router.get("/{solver_name}/required-files")
async def get_required_files(solver_name: str, config: str = "{}") -> dict[str, Any]:
    """Get the list of OpenFOAM files this solver would generate for a config."""
    import json

    registry = get_registry()
    plugin = registry.get(solver_name)
    if not plugin:
        raise HTTPException(404, f"Solver '{solver_name}' not found")

    try:
        parsed_config = json.loads(config)
    except json.JSONDecodeError:
        parsed_config = {}

    return {
        "solver": solver_name,
        "files": plugin.required_files(parsed_config),
    }


@router.post("/match")
async def match_solver(config: dict[str, Any]) -> list[SolverMatch]:
    """Given a simulation config, rank all solvers by match score.

    Returns all solvers sorted by score (highest first).
    """
    registry = get_registry()
    results = []
    for plugin in registry.all_solvers():
        try:
            match = plugin.matches(config)
            results.append(
                SolverMatch(
                    name=plugin.name,
                    score=match.score,
                    reason=match.reason,
                    warnings=match.warnings,
                )
            )
        except Exception as e:
            logger.warning("Solver %s.matches() failed: %s", plugin.name, e)
    results.sort(key=lambda r: r.score, reverse=True)
    return results
