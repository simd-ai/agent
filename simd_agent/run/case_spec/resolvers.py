"""Physics resolvers — pure-Python decisions that used to live in
post-generation validators.

Phase 2 of the LLM/validator-boundary redesign: every regex check in
``genai_codegen.py`` that auto-corrects an LLM-generated file (Check 7c,
7d, 7e, 3c2 …) is replaced by a resolver here that decides the right
value BEFORE the file is written.  The renderer reads the resolved
strategy and emits a correct file from the start; the validator check
becomes redundant and is deleted.

Resolvers in this module are:
  • Pure functions of their inputs (config + solver identity + mesh tier).
  • Total — they always return a valid strategy.
  • Unit-tested in ``tests/test_case_spec_resolvers.py``.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from .strategies import (
    CoarsestLevelCorr,
    CompressibleBounds,
    PressureSolverStrategy,
    TurbulenceRegimeProfile,
    TurbulenceSpec,
)


# ────────────────────────────────────────────────────────────────────────────
# Pressure solver — was Check 7c (GAMG hardening) + Check 7e (isothermal rho*)
# ────────────────────────────────────────────────────────────────────────────


def resolve_pressure_solver_strategy(
    solver_name: str,
    is_compressible: bool,
    mesh_tier: str,
    heat_transfer_active: bool,
) -> PressureSolverStrategy:
    """Pick the right ``solvers.p`` strategy for the given physics + mesh.

    Decision tree (encodes the historical regex-validator fixes as code):

      1. ``rhoPimpleFoam + isothermal``  → ``PBiCGStab + DILU`` (NOT GAMG).
         The pressure matrix is asymmetric (div(phid,p)) and at cold start
         GAMG's coarsest-level scale step SIGFPEs without h–ρ coupling to
         regularise it.  Replaces Check 7e.

      2. ``Poor mesh (tier='poor')``     → top-level fallback:
         compressible → ``PBiCGStab + DILU`` on the top-level asymmetric
         matrix; incompressible → ``PCG + DIC``.  GAMG agglomeration over
         tet meshes with non-ortho > 65° tends to produce singular coarse
         levels regardless of safeguards.

      3. ``Otherwise``                   → ``GAMG + GaussSeidel`` with
         ``nCoarsestCells = 20`` and a ``coarsestLevelCorr`` of
         ``PCG + DIC``.  The coarsest matrix is always symmetric after
         agglomeration; PCG + DIC is fast (3–5 iter) and OpenFOAM-valid.
         DILU at the coarsest level is impossible (Pydantic Literal).
         Replaces Check 7c.

         The 20-cell coarsest cap matches the OpenFOAM reference
         ``rhoSimpleFoam/angledDuctExplicitFixedCoeff`` tutorial — a value
         the OF maintainers have tested in production.  Earlier we used
         500 to leave the agglomeration safety margin for huge meshes,
         but the trade-off was: slower convergence on every-day case
         sizes because the coarse solve never gets a clean small matrix.
         The OF default is actually 10; 20 is a conservative middle.
    """
    # Rule 1 — isothermal rhoPimpleFoam: GAMG can SIGFPE at cold start.
    if solver_name == "rhoPimpleFoam" and not heat_transfer_active:
        return PressureSolverStrategy(
            top_level="PBiCGStab",
            smoother_or_precond="DILU",
            tolerance=1e-7,
            rel_tol=0.01,
            coarsest=None,
        )

    # Rule 2 — poor mesh: skip GAMG entirely, direct solver fallback.
    if mesh_tier == "poor":
        if is_compressible:
            return PressureSolverStrategy(
                top_level="PBiCGStab",
                smoother_or_precond="DILU",
                tolerance=1e-6,
                rel_tol=0.1,
                coarsest=None,
            )
        return PressureSolverStrategy(
            top_level="PCG",
            smoother_or_precond="DIC",
            tolerance=1e-6,
            rel_tol=0.1,
            coarsest=None,
        )

    # Rule 3 — default GAMG with safe coarsest level (PCG + DIC).
    return PressureSolverStrategy(
        top_level="GAMG",
        smoother_or_precond="GaussSeidel",
        n_coarsest_cells=20,
        tolerance=1e-6,
        rel_tol=0.1,
        coarsest=CoarsestLevelCorr(
            solver="PCG",
            preconditioner="DIC",
            tolerance=1e-9,
            rel_tol=0.0,
        ),
    )


def resolve_pressure_solver_from_config(
    solver_name: str,
    is_compressible: bool,
    config: dict[str, Any],
    mesh_tier: str | None = None,
) -> PressureSolverStrategy:
    """Convenience wrapper: extract the resolver inputs from a config dict.

    Used by the per-solver renderers in ``solvers/*/solver.py`` which already
    have ``config`` and the solver's plugin attributes in scope.
    """
    if mesh_tier is None:
        from .mesh_quality import _mesh_quality_decisions
        mesh = config.get("mesh", {}) or {}
        check_mesh = mesh.get("check_mesh") or mesh.get("checkMesh")
        mesh_tier = _mesh_quality_decisions(check_mesh)["mesh_quality_tier"]

    phys = config.get("physics") or {}
    heat = bool(
        config.get("heat_transfer")
        or phys.get("heat_transfer")
        or phys.get("energy")
    )
    return resolve_pressure_solver_strategy(
        solver_name=solver_name,
        is_compressible=is_compressible,
        mesh_tier=mesh_tier,
        heat_transfer_active=heat,
    )


# ────────────────────────────────────────────────────────────────────────────
# Compressible bounds — was Check 3c2 (fvOptions max clamp)
# ────────────────────────────────────────────────────────────────────────────


def resolve_compressible_bounds(
    is_compressible: bool,
    profile: str,
    rho: float | None,
    bc_temps: list[float] | None,
    eos_t_ceiling: float | None,
    op_p: float,
    mach: float,
    inlet_p: float | None = None,
) -> CompressibleBounds:
    """Resolve rhoMin / rhoMax / pMin / pMax / transonic in one place.

    Decision tree:

      Gas profile (perfectGas, no EOS ceiling):
        rho ∈ [0.1, ρ_max] where ρ_max = 1.5·(p_high / (R·T_cold))
        — derived from the *coldest* BC temperature and the *highest*
        boundary pressure (inlet if known, else operating).  The old
        hard-coded ρ_max = 10 was below the actual density at moderate
        compressor inlets (air @ 1.4 MPa, 280 K → 17.8 kg/m³) and
        silently clamped 30–60 % of inlet cells, breaking continuity.
        transonic = Mach > 0.5.
        p_min / p_max derived from operating pressure (5%–20×).

      Cryogenic profile (icoPolynomial, EOS ceiling known):
        rho ∈ [0.5·ρ_inlet, 1.5·ρ_inlet] when ρ is known.
        transonic always off (liquid sound speed ≫ inlet U).
        p_min / p_max from operating pressure.

      Incompressible: all None, transonic=False.
    """
    if not is_compressible:
        return CompressibleBounds(transonic=False)

    transonic = profile == "gas" and mach > 0.5
    # p_min / p_max — wide enough to cover the inlet–outlet pressure ratio
    # comfortably, but tight enough that a divergent solve gets clamped
    # before it reaches the ±1e+30 regime that triggers SIGFPE in GAMG.
    p_high = max(op_p, inlet_p or 0.0)
    p_min = max(1e3, op_p * 0.05)
    p_max = max(p_high * 1.5, op_p * 20.0, 5e5)

    if profile == "gas":
        # Ideal-gas density at the coldest BC temperature.
        # R_specific = 287 J/(kg·K) is a good approximation for air-like
        # gases (N₂, O₂, dry air); ±5 % off for CO₂, methane.  That tolerance
        # is well within our 1.5× safety factor.
        R_AIR = 287.0
        t_cold = min(bc_temps) if bc_temps else 288.15
        if t_cold <= 0:
            t_cold = 288.15
        rho_inlet_estimate = p_high / (R_AIR * t_cold)
        # Match OpenFOAM rhoSimpleFoam tutorial shape:
        #   rho_min = 0.5 · ρ_inlet,   rho_max = 1.5 · ρ_inlet.
        # We additionally apply absolute safety floors / ceilings so the
        # bounds never collapse on a sparse / cold-startup case:
        #   * rho_min ≥ 0.1 keeps GAMG safe even at vacuum chamber pressures
        #   * rho_max ≥ 10 preserves the "loose safety net" for moderate
        #     atmospheric cases where the 1.5× of 1.2 kg/m³ would give 1.8
        #   * rho_max ≤ 200 prevents absurd bounds in supersonic shock cases.
        rho_min = max(0.1, rho_inlet_estimate * 0.5)
        rho_max = min(200.0, max(10.0, rho_inlet_estimate * 1.5))
        # Guarantee the strict-less-than invariant (Pydantic enforces it).
        if rho_min >= rho_max:
            rho_min = min(rho_min, rho_max * 0.5)
        return CompressibleBounds(
            rho_min=rho_min,
            rho_max=rho_max,
            p_min=p_min,
            p_max=p_max,
            transonic=transonic,
        )

    # Cryogenic
    if rho is None or rho <= 0:
        return CompressibleBounds(
            p_min=p_min, p_max=p_max, transonic=False
        )
    return CompressibleBounds(
        rho_min=rho * 0.5,
        rho_max=rho * 1.5,
        p_min=p_min,
        p_max=p_max,
        transonic=False,
    )


def resolve_fv_options_max(
    profile: str,
    bc_temps: list[float],
    eos_t_ceiling: float | None,
    t_floor: float | None = None,
) -> float:
    """Resolve the ``limitTemperature.max`` value for ``system/fvOptions``.

    Gas (perfectGas, no EOS ceiling): max = min(3000 K, max(BC_T·1.5, BC_T+200)).
    Cryogenic (icoPolynomial ceiling known): max = 0.9 × eos_t_ceiling, with
    a lower bound of 3·t_floor to keep the bounds inverted-min invariant.
    Replaces the gas + cryogenic branches of Check 3c2.
    """
    if profile == "cryogenic" and eos_t_ceiling is not None:
        floor = (t_floor or 1.0) * 3.0
        return max(floor, eos_t_ceiling * 0.9)
    # Gas: derive from BC temperatures
    bc_max = max(bc_temps) if bc_temps else 500.0
    return min(3000.0, max(bc_max * 1.5, bc_max + 200.0))


# ────────────────────────────────────────────────────────────────────────────
# Energy convection scheme — was Check 7d (div(phi,h) upwind for ΔT > 100K)
# ────────────────────────────────────────────────────────────────────────────


def resolve_turbulence_spec(
    solver_plugin: Any,  # SolverPlugin — avoid circular import
    validated_config: dict[str, Any],
) -> TurbulenceSpec:
    """Solver-aware turbulence resolution.

    Reads from every known config shape (precheck nests it under
    ``turbulence.model``; the canonical schema uses ``physics.turbulence_model``;
    legacy configs may put it at the top level).  Applies the plugin's
    ``default_turbulence_model`` when nothing is set, instead of silently
    demoting to laminar — the historical bug that caused rhoSimpleFoam to
    ship with ``simulationType laminar`` and diverge with SIGFPE.

    Priority (highest first):
      1. Explicit ``flow_regime == "laminar"``  → laminar everywhere.
      2. Explicit ``turbulence_model`` from any known config path → respected
         (must be in ``solver_plugin.valid_turbulence_models``).
      3. Plugin's ``default_turbulence_model`` — applied when the user /
         planner failed to pick one.

    Raises ``ValueError`` if the resolved model isn't valid for the solver
    (so misconfiguration shows up at build_case_spec time, not after a
    multi-hour OpenFOAM SIGFPE).
    """
    phys = validated_config.get("physics") or {}
    if not isinstance(phys, dict):
        phys = {}
    turb_obj = validated_config.get("turbulence") or {}
    if not isinstance(turb_obj, dict):
        turb_obj = {}

    # ── flow_regime ─────────────────────────────────────────────────────
    _flow_regime_raw = (
        phys.get("flow_regime")
        or phys.get("flowRegime")
        or validated_config.get("flow_regime")
        or validated_config.get("flowRegime")
        or "turbulent"
    )
    flow_regime: Literal["laminar", "turbulent"] = (
        "laminar" if str(_flow_regime_raw).lower() == "laminar" else "turbulent"
    )

    # ── model lookup across every known field name & nesting ────────────
    _model_candidates = (
        phys.get("turbulence_model"),
        phys.get("turbulenceModel"),
        validated_config.get("turbulence_model"),
        validated_config.get("turbulenceModel"),
        turb_obj.get("model"),
        turb_obj.get("RASModel"),
        turb_obj.get("turbulenceModel"),
    )
    user_model: str | None = next(
        (m for m in _model_candidates if isinstance(m, str) and m), None
    )

    # ── pick the resolved model ─────────────────────────────────────────
    if flow_regime == "laminar":
        model = "laminar"
    elif user_model:
        model = user_model
    else:
        # No explicit model.  Use the plugin's default rather than silently
        # demoting to laminar (the failure mode that caused SIGFPE on
        # rhoSimpleFoam for moderate-Re forced-convection cases).
        model = getattr(solver_plugin, "default_turbulence_model", "kOmegaSST")

    # ── validate ────────────────────────────────────────────────────────
    valid: frozenset[str] = getattr(
        solver_plugin, "valid_turbulence_models", frozenset()
    )
    if valid and model not in valid:
        raise ValueError(
            f"Solver {getattr(solver_plugin, 'name', '?')!r} does not support "
            f"turbulence model {model!r}.  Valid models: {sorted(valid)}."
        )

    # ── simulation_type derived from model ──────────────────────────────
    simulation_type: Literal["laminar", "RAS", "LES"]
    if model == "laminar":
        simulation_type = "laminar"
    elif "LES" in model:
        simulation_type = "LES"
    else:
        simulation_type = "RAS"

    # ── wall_functions: per-plugin default; could be config-overridable ─
    wall_functions = bool(turb_obj.get("wall_functions", True)) if turb_obj else True

    return TurbulenceSpec(
        flow_regime=flow_regime,
        model=model,
        simulation_type=simulation_type,
        wall_functions=wall_functions,
    )


def resolve_div_phi_h_scheme(
    is_compressible_energy: bool,
    bc_temps: list[float] | None,
) -> str:
    """Pick the divergence scheme for the energy field convection term.

    Large ΔT (> 100 K) drives enthalpy overshoots when linearUpwind is used;
    those overshoots clamp 50 %+ of cells via fvOptions limitTemperature,
    leaving the h equation ill-conditioned and PIMPLE never converging.
    Replaces Check 7d.

    Returns the OpenFOAM scheme string, e.g.::
        "bounded Gauss upwind"          # large ΔT — robust
        "bounded Gauss linearUpwind grad(h)"   # moderate ΔT — accurate
    """
    if not is_compressible_energy:
        return "bounded Gauss upwind"
    if bc_temps and len(bc_temps) >= 2:
        delta_t = max(bc_temps) - min(bc_temps)
        if delta_t > 100.0:
            return "bounded Gauss upwind"
    # Default for compressible energy solvers — upwind is the textbook
    # safe choice; linearUpwind is only worth it for moderate ΔT.
    return "bounded Gauss upwind"


# ────────────────────────────────────────────────────────────────────────────
# Turbulence regime profile — per-(simulation_type × algorithm) scheme bundle
# ────────────────────────────────────────────────────────────────────────────
#
# Encodes the per-regime choices observed across the three OpenFOAM 4.x
# rhoPimpleFoam reference tutorials:
#
#   compressible/rhoPimpleFoam/laminar/helmholtzResonance
#   compressible/rhoPimpleFoam/ras/angledDuct
#   compressible/rhoPimpleFoam/les/pitzDaily
#
# plus the rhoSimpleFoam RAS reference (angledDuctExplicitFixedCoeff) for the
# SIMPLE-mode steady case.  Renderers in ``solvers/base.py`` read this
# resolved profile and emit the right scheme line for the regime; no inline
# branching, no string-tag comparisons.


def _laminar_properties_block() -> str:
    return "simulationType  laminar;\n"


def _ras_properties_block(model: str) -> str:
    return (
        "simulationType  RAS;\n\n"
        "RAS\n"
        "{\n"
        f"    RASModel        {model};\n"
        "    turbulence      on;\n"
        "    printCoeffs     on;\n"
        "}\n"
    )


def _les_properties_block(model: str) -> str:
    # Minimal LES block — covers kEqn / Smagorinsky / dynamicKEqn.
    # The cubeRootVol delta is the textbook OF choice for unstructured
    # meshes; sub-block ``cubeRootVolCoeffs { deltaCoeff 1; }`` keeps
    # the OpenFOAM dict reader happy across forks.
    return (
        "simulationType  LES;\n\n"
        "LES\n"
        "{\n"
        f"    LESModel        {model};\n"
        "    turbulence      on;\n"
        "    printCoeffs     on;\n"
        "    delta           cubeRootVol;\n\n"
        "    cubeRootVolCoeffs\n"
        "    {\n"
        "        deltaCoeff      1;\n"
        "    }\n"
        "}\n"
    )


def resolve_regime_profile(
    simulation_type: Literal["laminar", "RAS", "LES"],
    turb_model: str,
    algorithm: Literal["SIMPLE", "PIMPLE", "PISO"],
    is_compressible: bool,
    energy_var: str = "h",
) -> TurbulenceRegimeProfile:
    """Resolve the full per-regime scheme bundle in one place.

    The three OF reference tutorials drive every choice here:

      laminar (helmholtzResonance) — accuracy-preferred, no turb fields:
        ``ddt Euler``, ``div(phi,U) Gauss limitedLinearV 1``,
        ``div(phi,e) Gauss limitedLinear 1``, ``div(phi,K) Gauss
        limitedLinear 1``, ``div(phiv,p) Gauss limitedLinear 1``.

      RAS (angledDuct) — robustness-preferred, transported k/ε/ω:
        ``ddt Euler``, ``div(phi,U) Gauss upwind``,
        ``div(phi,h) Gauss upwind``, ``div(phi,K) Gauss linear``,
        ``div(phid,p) Gauss upwind``, ``div(phi,k) Gauss upwind``.
        SIMPLE mode (rhoSimpleFoam) uses ``ddt steadyState`` instead of
        ``Euler``.

      LES (pitzDaily) — eddy-resolved, second-order time:
        ``ddt backward``, ``div(phi,U) Gauss LUST grad(U)``,
        ``div(phi,e) Gauss LUST grad(e)``, ``div(phi,K) Gauss linear``,
        ``div(phiv,p) Gauss linear``, ``div(phi,k) Gauss limitedLinear 1``.

    Pressure flux:
      * RAS rhoPimple uses ``phid`` (compressibility-coupled flux,
        because ``transonic`` is set explicitly).
      * Laminar + LES use ``phiv`` (kinematic — they integrate the
        pressure equation in a low-Mach form).
      * SIMPLE-mode steady (rhoSimpleFoam) uses ``phid``.
    """
    # ── ddt ──
    if algorithm == "SIMPLE":
        ddt = "steadyState"
    elif simulation_type == "LES":
        ddt = "backward"  # 2nd-order time accuracy for resolved turbulence
    else:
        ddt = "Euler"

    # ── turbulenceProperties block ──
    if simulation_type == "laminar":
        tp_block = _laminar_properties_block()
    elif simulation_type == "LES":
        tp_block = _les_properties_block(turb_model)
    else:
        tp_block = _ras_properties_block(turb_model)

    # ── div(phi,*) schemes ──
    if simulation_type == "laminar":
        div_U = "Gauss limitedLinearV 1"
        div_energy = "Gauss limitedLinear 1"
        div_K = "Gauss limitedLinear 1"
        div_p = "Gauss limitedLinear 1"
        div_turb: str | None = None
        flux = "phiv"
    elif simulation_type == "LES":
        div_U = "Gauss LUST grad(U)"
        div_energy = f"Gauss LUST grad({energy_var})"
        div_K = "Gauss linear"
        div_p = "Gauss linear"
        div_turb = "Gauss limitedLinear 1"
        flux = "phiv"
    else:
        # RAS — robustness pattern from the OF rhoPimpleFoam ras tutorial.
        # rhoSimpleFoam (SIMPLE) uses the same RAS scheme set; the algorithm
        # affects ddt + pressure-flux choice, not divSchemes.
        div_U = "bounded Gauss upwind"
        div_energy = "bounded Gauss upwind"
        # div(phi,K) — Gauss linear for rhoPimpleFoam (transient, h energy),
        # bounded Gauss upwind elsewhere.  rhoSimpleFoam tutorial uses
        # ``bounded Gauss upwind`` for div(phi,Ekp)/K so SIMPLE keeps the
        # safer choice.
        div_K = "Gauss linear" if algorithm == "PIMPLE" else "bounded Gauss upwind"
        div_p = "Gauss upwind"
        div_turb = "bounded Gauss upwind" if algorithm == "PIMPLE" \
            else "bounded Gauss limitedLinear 1"
        flux = "phid" if is_compressible else "phiv"

    pressure_flux = cast(Literal["phid", "phiv"], flux)

    return TurbulenceRegimeProfile(
        simulation_type=simulation_type,
        ddt_scheme=ddt,
        div_phi_U=div_U,
        div_phi_energy=div_energy,
        div_phi_K=div_K,
        div_phi_p=div_p,
        div_phi_turb=div_turb,
        pressure_flux=pressure_flux,
        turbulence_properties_block=tp_block,
    )
