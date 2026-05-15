"""fvSchemes / fvSolution block builders as free functions.

Hoisted out of ``SolverPlugin`` for the same reasons as ``legacy_fixers``
and ``bc_fixers``: each helper is essentially a pure function of the
typed ``FvBuildContext`` plus a handful of plugin identity attributes
(``pressure_field``, ``algorithm``, ``is_compressible``, ``energy_var``,
…).  Calling them via ``self._build_X(...)`` hid that.

Each function takes:

  * ``ctx``     — the typed ``FvBuildContext`` built by ``_fv_context``.
  * ``plugin``  — the calling plugin instance.  Only its public class
                  attributes are read (``pressure_field``, ``algorithm``,
                  etc.) and one method (``turbulence_fields``).  Using
                  ``SolverPlugin`` as the type captures the contract
                  without circular import (only TYPE_CHECKING).

Returns: ready-to-emit OpenFOAM dict text.

The block builders here are *paradigm-agnostic*; the family bases
(SteadyBase / TransientBase) own the SIMPLE / PIMPLE blocks because
those differ structurally per algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simd_agent.solvers.base import SolverPlugin
    from simd_agent.solvers.contexts import FvBuildContext


# ────────────────────────────────────────────────────────────────────────────
# fvSolution — pressure + equation solver blocks
# ────────────────────────────────────────────────────────────────────────────


def pressure_solver_block(
    plugin: "SolverPlugin",
    ctx: "FvBuildContext",
    is_simple: bool | None = None,
) -> tuple[str, str]:
    """Build ``p`` (or ``p_rgh``) solver block and optional ``pFinal``.

    The (GAMG vs PBiCGStab, coarsestLevelCorr settings, rhoPimpleFoam-
    isothermal special-case) decisions live in
    ``resolve_pressure_solver_strategy``.  This function is a pure
    renderer over the resolved strategy.
    """
    from simd_agent.run.case_spec import resolve_pressure_solver_strategy

    if is_simple is None:
        is_simple = plugin.algorithm == "SIMPLE"
    pf = plugin.pressure_field
    heat = ctx.heat_transfer_active

    strategy = resolve_pressure_solver_strategy(
        solver_name=plugin.name,
        is_compressible=plugin.is_compressible,
        mesh_tier=ctx.tier,
        heat_transfer_active=heat,
    )

    rel_tol_str = f"{strategy.rel_tol:g}"

    if strategy.top_level == "GAMG":
        assert strategy.coarsest is not None
        cl = strategy.coarsest
        p_block = (
            f"    {pf}\n"
            "    {\n"
            "        solver          GAMG;\n"
            f"        smoother        {strategy.smoother_or_precond};\n"
            f"        nCoarsestCells  {strategy.n_coarsest_cells};\n"
            f"        tolerance       {strategy.tolerance:g};\n"
            f"        relTol          {rel_tol_str};\n"
            "        coarsestLevelCorr\n"
            "        {\n"
            f"            solver          {cl.solver};\n"
            f"            preconditioner  {cl.preconditioner};\n"
            f"            tolerance       {cl.tolerance:g};\n"
            f"            relTol          {cl.rel_tol:g};\n"
            "        }\n"
            "    }\n"
        )
    else:
        # Direct Krylov path (PBiCGStab or PCG) — no coarsestLevelCorr.
        p_block = (
            f"    {pf}\n"
            "    {\n"
            f"        solver          {strategy.top_level};\n"
            f"        preconditioner  {strategy.smoother_or_precond};\n"
            f"        tolerance       {strategy.tolerance:g};\n"
            f"        relTol          {rel_tol_str};\n"
            "    }\n"
        )

    p_final_block = ""
    if not is_simple:
        p_final_block = (
            f"\n    {pf}Final\n"
            "    {\n"
            f"        ${pf};\n"
            "        relTol          0;\n"
            "    }\n"
        )
    return p_block, p_final_block


def equation_solver_block(
    plugin: "SolverPlugin",
    eq_fields: list[str],
    is_simple: bool | None = None,
) -> tuple[str, str]:
    """Build the equation regex solver block + its PIMPLE Final variant.

    Energy split: when the solver carries an energy equation, the energy
    variable (``e`` or ``h``) is broken out of the smoothSolver regex
    group into its own ``PBiCG + DILU`` block — matches the OF
    rhoSimpleFoam reference (smoothSolver under-converges scalar
    transport on asymmetric matrices).
    """
    if is_simple is None:
        is_simple = plugin.algorithm == "SIMPLE"

    if plugin.supports_energy and plugin.energy_var in eq_fields:
        non_energy = [f for f in eq_fields if f != plugin.energy_var]
    else:
        non_energy = list(eq_fields)

    if len(non_energy) == 1:
        eq_regex = non_energy[0]
    elif len(non_energy) >= 2:
        eq_regex = f'"({"|".join(non_energy)})"'
    else:
        eq_regex = ""

    eq_block = ""
    if eq_regex:
        eq_block = (
            f"\n    {eq_regex}\n"
            "    {\n"
            "        solver          smoothSolver;\n"
            "        smoother        symGaussSeidel;\n"
            "        tolerance       1e-05;\n"
            "        relTol          0.1;\n"
            "    }\n"
        )

    if plugin.supports_energy and plugin.energy_var in eq_fields:
        eq_block += (
            f"\n    {plugin.energy_var}\n"
            "    {\n"
            "        solver          PBiCG;\n"
            "        preconditioner  DILU;\n"
            "        tolerance       1e-06;\n"
            "        relTol          0.1;\n"
            "    }\n"
        )

    eq_final_block = ""
    if not is_simple:
        if eq_regex:
            if eq_regex.startswith('"'):
                inner = eq_regex[1:-1]
                final_regex = f'"{inner}Final"'
            else:
                final_regex = f"{eq_regex}Final"
            eq_final_block = (
                f"\n    {final_regex}\n"
                "    {\n"
                "        solver          smoothSolver;\n"
                "        smoother        symGaussSeidel;\n"
                "        tolerance       1e-06;\n"
                "        relTol          0;\n"
                "    }\n"
            )
        if plugin.supports_energy and plugin.energy_var in eq_fields:
            eq_final_block += (
                f"\n    {plugin.energy_var}Final\n"
                "    {\n"
                "        solver          PBiCG;\n"
                "        preconditioner  DILU;\n"
                "        tolerance       1e-06;\n"
                "        relTol          0;\n"
                "    }\n"
            )
    return eq_block, eq_final_block


# ────────────────────────────────────────────────────────────────────────────
# fvSchemes — ddt / grad / div / laplacian / snGrad / interpolation / wallDist
# ────────────────────────────────────────────────────────────────────────────


def ddt_block(
    plugin: "SolverPlugin",
    ctx: "FvBuildContext | None" = None,
) -> str:
    """ddtSchemes — driven by ``ctx.regime_profile.ddt_scheme``.

    LES needs ``backward`` (2nd-order time); SIMPLE-mode steady uses
    ``steadyState``; transient defaults to ``Euler``.  Falls back to
    plugin's algorithm-driven default when ctx is missing (legacy tests).
    """
    if ctx is not None and ctx.regime_profile is not None:
        ddt = ctx.regime_profile.ddt_scheme
    else:
        ddt = "Euler" if plugin.is_transient else "steadyState"
    return (
        "ddtSchemes\n"
        "{\n"
        f"    default         {ddt};\n"
        "}\n"
    )


def grad_block(plugin: "SolverPlugin", ctx: "FvBuildContext") -> str:
    """gradSchemes — cellLimited grad(U) for compressible gas only."""
    if plugin.is_compressible and ctx.profile == "gas":
        grad_u_line = "    grad(U)         cellLimited Gauss linear 1;\n"
    else:
        grad_u_line = ""
    return (
        "gradSchemes\n"
        "{\n"
        "    default         Gauss linear;\n"
        f"{grad_u_line}"
        "}\n"
    )


def div_block(plugin: "SolverPlugin", ctx: "FvBuildContext") -> str:
    """divSchemes — driven by ``ctx.regime_profile`` when set.

    Profile path: every per-regime scheme (laminar / RAS / LES) is
    encoded in the resolved ``TurbulenceRegimeProfile``.  Legacy fallback
    kept for unit tests that build ``FvBuildContext`` without a profile.
    """
    from simd_agent.run.case_spec import resolve_div_phi_h_scheme

    speed_tier = ctx.speed_tier
    profile = ctx.profile
    turb_model = ctx.turb_model
    rp = ctx.regime_profile

    lines: list[str] = ["    default         none;"]

    if rp is not None:
        lines.append(f"    div(phi,U)      {rp.div_phi_U};")
        if plugin.supports_energy:
            lines.append(
                f"    div(phi,{plugin.energy_var})      {rp.div_phi_energy};"
            )
            ke_name = "Ekp" if plugin.energy_var == "e" else "K"
            lines.append(
                f"    div(phi,{ke_name})      {rp.div_phi_K};"
            )
        if plugin.is_compressible and not plugin.needs_gravity:
            lines.append(
                f"    div({rp.pressure_flux},p)     {rp.div_phi_p};"
            )
        if rp.div_phi_turb is not None:
            turb_fields = plugin.turbulence_fields(turb_model)
            transported = [
                f for f in turb_fields
                if f in ("k", "omega", "epsilon", "nuTilda")
            ]
            if transported:
                lines.append("")
                for f in transported:
                    lines.append(f"    div(phi,{f})    {rp.div_phi_turb};")
    else:
        # Legacy literal path (pre-Phase-5).
        _bc_temps = list(ctx.bc_temps)
        if plugin.is_compressible:
            if plugin.algorithm == "SIMPLE":
                lines.append("    div(phi,U)      bounded Gauss upwind;")
            else:
                _high_dp = ctx.pressure_ratio >= 3.0
                if (
                    profile == "gas"
                    and speed_tier in ("low", "moderate")
                    and not _high_dp
                ):
                    lines.append("    div(phi,U)      bounded Gauss linearUpwindV grad(U);")
                else:
                    lines.append("    div(phi,U)      bounded Gauss upwind;")
            if plugin.supports_energy:
                _h_scheme = resolve_div_phi_h_scheme(
                    is_compressible_energy=True,
                    bc_temps=_bc_temps if _bc_temps else None,
                )
                lines.append(
                    f"    div(phi,{plugin.energy_var})      {_h_scheme};"
                )
                if plugin.energy_var == "e":
                    lines.append("    div(phi,Ekp)    bounded Gauss upwind;")
                else:
                    lines.append("    div(phi,K)      bounded Gauss upwind;")
            if not plugin.needs_gravity:
                lines.append("    div(phid,p)     Gauss upwind;")
        else:
            if speed_tier == "high":
                lines.append("    div(phi,U)      bounded Gauss upwind;")
            else:
                lines.append("    div(phi,U)      bounded Gauss linearUpwind grad(U);")

        turb_fields = (
            plugin.turbulence_fields(turb_model)
            if turb_model != "laminar"
            else []
        )
        transported = [
            f for f in turb_fields
            if f in ("k", "omega", "epsilon", "nuTilda")
        ]
        if transported:
            lines.append("")
            turb_scheme = (
                "bounded Gauss upwind" if speed_tier == "high"
                else "bounded Gauss limitedLinear 1"
            )
            for f in transported:
                lines.append(f"    div(phi,{f})    {turb_scheme};")

    # Viscous stress tensor — same form in all regimes.
    lines.append("")
    if plugin.is_compressible:
        lines.append("    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;")
    else:
        lines.append("    div((nuEff*dev2(T(grad(U))))) Gauss linear;")

    block_body = "\n".join(lines)
    return (
        "divSchemes\n"
        "{\n"
        f"{block_body}\n"
        "}\n"
    )


def laplacian_block(plugin: "SolverPlugin", ctx: "FvBuildContext") -> str:
    """laplacianSchemes — mesh-quality-blended ``corrected``."""
    scheme = mesh_blended_scheme(ctx, kind="laplacian")
    return (
        "laplacianSchemes\n"
        "{\n"
        f"    default         {scheme};\n"
        "}\n"
    )


def sngrad_block(plugin: "SolverPlugin", ctx: "FvBuildContext") -> str:
    """snGradSchemes — mesh-quality-blended ``corrected``."""
    scheme = mesh_blended_scheme(ctx, kind="sngrad")
    return (
        "snGradSchemes\n"
        "{\n"
        f"    default         {scheme};\n"
        "}\n"
    )


def mesh_blended_scheme(ctx: "FvBuildContext", kind: str) -> str:
    """Pick laplacian / snGrad scheme from mesh tier + non-orthogonality.

    Good orthogonal meshes get plain ``corrected`` (accuracy-preferred).
    Moderate / unknown meshes get ``limited corrected 0.5``.
    Poor meshes (non-ortho ≥ 65°) get ``limited corrected 0.33``.
    """
    tier = ctx.tier
    non_ortho = ctx.non_ortho
    if tier == "good" and non_ortho < 40:
        return "Gauss linear corrected" if kind == "laplacian" else "corrected"
    if non_ortho >= 65 or tier == "poor":
        return (
            "Gauss linear limited corrected 0.33"
            if kind == "laplacian" else "limited corrected 0.33"
        )
    return (
        "Gauss linear limited corrected 0.5"
        if kind == "laplacian" else "limited corrected 0.5"
    )


def interpolation_block() -> str:
    return (
        "interpolationSchemes\n"
        "{\n"
        "    default         linear;\n"
        "}\n"
    )


def flux_required_block(plugin: "SolverPlugin") -> str:
    return (
        "fluxRequired\n"
        "{\n"
        "    default         no;\n"
        f"    {plugin.pressure_field};\n"
        "}\n"
    )


def wall_dist_block(turb_model: str) -> str:
    """wallDist block — emitted only when a wall-aware turbulence model is active."""
    if turb_model == "laminar":
        return ""
    return (
        "wallDist\n"
        "{\n"
        "    method          meshWave;\n"
        "}\n"
    )
