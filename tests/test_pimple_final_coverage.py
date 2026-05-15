# tests/test_pimple_final_coverage.py
"""Tests that every solved field has a ``<field>Final`` variant for PIMPLE.

The OpenFOAM PIMPLE algorithm consults ``<field>Final`` on the final outer
iteration of each time step.  A missing entry is fatal:

    FOAM FATAL IO ERROR: Entry 'rhoFinal' not found in dictionary
    "system/fvSolution/solvers"

This was the user-reported failure on rhoPimpleFoam — the ``rho`` solver
entry existed but its ``rhoFinal`` did not.  The fix lives on
``SolverPlugin._build_rho_solver_block``: it emits ``rhoFinal`` when the
plugin's algorithm is PIMPLE / PISO, omits it for SIMPLE.

Coverage applies regardless of turbulence regime (laminar / RAS / LES) —
``rho`` is solved in every compressible PIMPLE case.
"""

from __future__ import annotations

import re

from simd_agent.solvers.compressible.rhoPimpleFoam.solver import RhoPimpleFoamSolver
from simd_agent.solvers.compressible.rhoSimpleFoam.solver import RhoSimpleFoamSolver
from simd_agent.solvers.heatTransfer.buoyantPimpleFoam.solver import BuoyantPimpleFoamSolver
from simd_agent.solvers.heatTransfer.buoyantSimpleFoam.solver import BuoyantSimpleFoamSolver


_BASE_CFG = {
    "physics": {
        "compressibility": "compressible",
        "heat_transfer": True,
        "time_scheme": "transient",
        "turbulence_model": "kEpsilon",
    },
    "fluid": {"rho": 1.18, "mu": 1.81e-5, "Cp": 1006, "k": 0.026, "temperature": 300},
    "boundary_conditions": {
        "inlet": {"patch_class": "inlet", "pressure": {"value": 1.0e5}, "temperature": {"value": 293}},
        "outlet": {"patch_class": "outlet", "pressure": {"value": 1.0e5}, "temperature": {"value": 293}},
        "walls": {"patch_class": "wall"},
    },
    "mesh": {},
}


def _top_level_solver_entries(fvs: str) -> list[str]:
    """Return only the top-level solver entries (excludes sub-blocks)."""
    solvers_block = fvs.split("solvers")[1].split("\n}")[0]
    return re.findall(
        r'^\s{4}(\w+|"[^"]+")\n\s{4}\{', solvers_block, re.MULTILINE
    )


def _has_final_for(entry: str, entries: list[str]) -> bool:
    """``rho`` needs ``rhoFinal``; ``"(U|k)"`` needs ``"(U|k)Final"``."""
    if entry.startswith('"') and entry.endswith('"'):
        inner = entry[1:-1]
        return f'"{inner}Final"' in entries
    return f"{entry}Final" in entries


# ── rhoPimpleFoam: every solved field has a Final ───────────────────────────


class TestRhoPimpleFoamFinalCoverage:
    def test_ras_regime_full_coverage(self):
        plugin = RhoPimpleFoamSolver()
        fvs = plugin.render_deterministic_files(_BASE_CFG)["system/fvSolution"]
        entries = _top_level_solver_entries(fvs)
        # rho is the regression case from the user-reported error.
        assert "rho" in entries
        assert "rhoFinal" in entries
        # Every non-Final solver has a Final pair.
        non_final = [e for e in entries if not e.endswith(('Final"', "Final"))]
        for e in non_final:
            assert _has_final_for(e, entries), (
                f"Missing Final for solver entry {e!r}; entries={entries}"
            )

    def test_laminar_regime_full_coverage(self):
        plugin = RhoPimpleFoamSolver()
        cfg = {
            **_BASE_CFG,
            "physics": {
                **_BASE_CFG["physics"],
                "flow_regime": "laminar",
                "turbulence_model": "laminar",
            },
        }
        fvs = plugin.render_deterministic_files(cfg)["system/fvSolution"]
        entries = _top_level_solver_entries(fvs)
        # Laminar still needs rho/rhoFinal — they're compressible-physics
        # solved fields independent of turbulence regime.
        assert "rho" in entries
        assert "rhoFinal" in entries
        # U + UFinal for momentum (no regex group when only U is present).
        assert "U" in entries
        assert "UFinal" in entries
        # h + hFinal for energy.
        assert "h" in entries
        assert "hFinal" in entries

    def test_les_regime_full_coverage(self):
        plugin = RhoPimpleFoamSolver()
        cfg = {
            **_BASE_CFG,
            "physics": {
                **_BASE_CFG["physics"],
                "flow_regime": "turbulent",
                "simulation_type": "LES",
                "turbulence_model": "LES",
            },
        }
        fvs = plugin.render_deterministic_files(cfg)["system/fvSolution"]
        entries = _top_level_solver_entries(fvs)
        # LES still uses ρ + ρFinal (compressible variant of pitzDaily).
        assert "rho" in entries
        assert "rhoFinal" in entries


# ── rhoSimpleFoam (SIMPLE): no Final variants ────────────────────────────────


class TestRhoSimpleFoamHasNoFinals:
    def test_no_rho_final_in_simple(self):
        plugin = RhoSimpleFoamSolver()
        cfg = {**_BASE_CFG, "physics": {**_BASE_CFG["physics"], "time_scheme": "steady"}}
        fvs = plugin.render_deterministic_files(cfg)["system/fvSolution"]
        entries = _top_level_solver_entries(fvs)
        assert "rho" in entries
        # SIMPLE has no concept of a final outer iteration → no Final
        # variants are needed or expected.
        assert "rhoFinal" not in entries
        assert "pFinal" not in entries


# ── Buoyant PIMPLE / SIMPLE — same invariant ───────────────────────────────


class TestBuoyantSolversFinalCoverage:
    def test_buoyantPimpleFoam_has_rhoFinal(self):
        plugin = BuoyantPimpleFoamSolver()
        if not plugin.is_compressible:
            return  # Boussinesq incompressible — no rho field at all.
        cfg = {
            **_BASE_CFG,
            "physics": {**_BASE_CFG["physics"], "gravity": True},
            "boundary_conditions": {
                **_BASE_CFG["boundary_conditions"],
            },
        }
        fvs = plugin.render_deterministic_files(cfg)["system/fvSolution"]
        entries = _top_level_solver_entries(fvs)
        assert "rho" in entries
        assert "rhoFinal" in entries

    def test_buoyantSimpleFoam_has_no_rhoFinal(self):
        plugin = BuoyantSimpleFoamSolver()
        if not plugin.is_compressible:
            return  # Boussinesq incompressible — skip.
        cfg = {**_BASE_CFG, "physics": {**_BASE_CFG["physics"], "time_scheme": "steady", "gravity": True}}
        fvs = plugin.render_deterministic_files(cfg)["system/fvSolution"]
        entries = _top_level_solver_entries(fvs)
        if "rho" in entries:
            assert "rhoFinal" not in entries
