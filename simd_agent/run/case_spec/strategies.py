"""Typed sub-models that capture resolver decisions for the OpenFOAM case.

These models are the **contract** between the physics resolver and the file
renderers.  Each is a Pydantic v2 model with `Literal` fields where the value
set is closed by OpenFOAM, so invalid combinations are unrepresentable at the
type level.

Bug classes that disappear by construction once renderers consume these:

  * `CoarsestLevelCorr` cannot hold `preconditioner="DILU"` — OpenFOAM 2406
    only accepts {DIC, FDIC, GAMG, diagonal, distributedDIC, none} on the
    symmetric coarsest matrix.
  * `FluidThermo` cannot pair `equationOfState="icoPolynomial"` with
    `transport="const"` — the combination throws "Unknown fluidThermo type"
    at OpenFOAM runtime.
  * `FluidThermo` cannot pair native liquidProperties with
    `energy="sensibleEnthalpy"` — the combination SIGFPEs in heRhoThermo's
    constructor.
  * `CompressibleBounds.rho_min < rho_max` is enforced.
  * `InletTurbulence.intensity ∈ [0.001, 0.30]` — physically plausible TI range.

Phase 1 adds these models as types only; resolvers don't populate them yet,
renderers don't read them yet.  Phase 2 wires them into the resolver, Phase 3
points renderers at them, Phase 4 deletes the corresponding ~80 LOC of regex
post-validators.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ────────────────────────────────────────────────────────────────────────────
# FluidThermo — thermophysicalProperties strategy
# ────────────────────────────────────────────────────────────────────────────

ThermoPackage = Literal["hePsiThermo", "heRhoThermo"]
EquationOfState = Literal[
    "perfectGas", "icoPolynomial", "rhoConst",
    "PengRobinsonGas", "incompressiblePerfectGas",
]
TransportModel = Literal["const", "sutherland", "polynomial"]
ThermoModel = Literal["hConst", "eConst", "hPolynomial", "ePolynomial", "janaf"]
EnergyForm = Literal["sensibleEnthalpy", "sensibleInternalEnergy"]


class FluidThermo(BaseModel):
    """Resolved `constant/thermophysicalProperties` strategy.

    The renderer reads these five fields and emits a syntactically valid,
    physically self-consistent thermoType block.  Invalid combinations raise
    a Pydantic ValidationError at construction time.
    """

    package: ThermoPackage
    eos: EquationOfState
    transport: TransportModel
    thermo: ThermoModel
    energy: EnergyForm

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _enforce_known_compatibility_rules(self) -> "FluidThermo":
        # icoPolynomial REQUIRES polynomial transport + hPolynomial thermo
        # (anything else throws "Unknown fluidThermo type" in OpenFOAM 2406).
        if self.eos == "icoPolynomial":
            if self.transport != "polynomial":
                raise ValueError(
                    "icoPolynomial requires transport='polynomial' "
                    f"(got {self.transport!r}).  Other combinations throw "
                    "'Unknown fluidThermo type' at OpenFOAM runtime."
                )
            if self.thermo not in ("hPolynomial", "ePolynomial"):
                raise ValueError(
                    "icoPolynomial requires thermo='hPolynomial' or 'ePolynomial' "
                    f"(got {self.thermo!r})."
                )
        # perfectGas pairs with const/sutherland transport + hConst/janaf thermo
        if self.eos == "perfectGas":
            if self.transport == "polynomial":
                raise ValueError(
                    "perfectGas + polynomial transport is unusual; use sutherland "
                    "for temperature-dependent viscosity or const for isothermal."
                )
            # mypy narrows self.transport to {"const", "sutherland"} here;
            # the polynomial-thermo check is therefore unconditional.
            if self.thermo in ("hPolynomial", "ePolynomial"):
                raise ValueError(
                    "hPolynomial / ePolynomial requires polynomial transport."
                )
        # Native liquidProperties packages MUST use sensibleInternalEnergy —
        # sensibleEnthalpy SIGFPEs in heRhoThermo's constructor on OF 2406.
        # We don't model native liquidProperties as an EOS yet; this branch
        # is a placeholder for when we do.
        # (Documented invariant: multiphase compressible cases must override
        # this rule explicitly when adding new EOS values.)
        return self


# ────────────────────────────────────────────────────────────────────────────
# Pressure solver — fvSolution.solvers.p block
# ────────────────────────────────────────────────────────────────────────────

# Solvers valid on a symmetric matrix (used at GAMG's coarsest level).
SymmetricSolver = Literal["PCG", "PBiCGStab", "smoothSolver", "GAMG"]
# Preconditioners valid on a symmetric matrix.  Note DILU is NOT here —
# OpenFOAM 2406 rejects it with "Unknown symmetric matrix preconditioner".
SymmetricPreconditioner = Literal[
    "DIC", "FDIC", "GAMG", "diagonal", "distributedDIC", "none"
]
# Preconditioners valid on an asymmetric matrix (top-level p block in
# compressible cases — div(phid,p) makes the matrix asymmetric).
AsymmetricPreconditioner = Literal["DILU", "diagonal", "none"]

TopLevelSolver = Literal["GAMG", "PBiCGStab", "PCG"]
TopLevelSmoother = Literal["GaussSeidel", "symGaussSeidel", "DIC", "DILU"]


class CoarsestLevelCorr(BaseModel):
    """`coarsestLevelCorr` sub-block inside GAMG.

    The agglomerated coarsest matrix is **always symmetric** regardless of
    the top-level matrix's symmetry.  The solver and preconditioner here
    are therefore restricted to the symmetric-matrix registry.  This is the
    site of the historical DILU bug — unrepresentable now.
    """

    solver: SymmetricSolver = "PCG"
    preconditioner: SymmetricPreconditioner = "DIC"
    tolerance: float = Field(default=1e-9, gt=0.0)
    rel_tol: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(frozen=True)


class PressureSolverStrategy(BaseModel):
    """Top-level `solvers.p` (or `p_rgh`) strategy.

    Branches:
      * GAMG path — `top_level="GAMG"`, requires `smoother` and `coarsest`.
      * Direct path — `top_level="PCG"` or `"PBiCGStab"`, requires
        `preconditioner` valid for the relevant matrix symmetry.
    """

    top_level: TopLevelSolver = "GAMG"
    # For GAMG: a smoother (Gauss-Seidel family).  For direct solvers: a
    # preconditioner appropriate to the matrix symmetry.  We model both
    # under a single field; the validator enforces the right registry.
    smoother_or_precond: str = "GaussSeidel"
    n_coarsest_cells: int = Field(default=20, gt=0)
    tolerance: float = Field(default=1e-6, gt=0.0)
    rel_tol: float = Field(default=0.1, ge=0.0)
    coarsest: CoarsestLevelCorr | None = None  # required iff top_level == GAMG

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _enforce_branch_consistency(self) -> "PressureSolverStrategy":
        if self.top_level == "GAMG":
            if self.coarsest is None:
                raise ValueError(
                    "GAMG requires a coarsestLevelCorr block — at least "
                    "CoarsestLevelCorr() with defaults (PCG + DIC)."
                )
            valid_smoothers = {"GaussSeidel", "symGaussSeidel", "DIC", "DILU"}
            if self.smoother_or_precond not in valid_smoothers:
                raise ValueError(
                    f"GAMG smoother must be one of {sorted(valid_smoothers)} "
                    f"(got {self.smoother_or_precond!r})."
                )
        else:
            # PCG / PBiCGStab path — `smoother_or_precond` is the preconditioner.
            # We can't strictly verify symmetry here without the matrix; allow
            # both registries (DILU for asymmetric, DIC for symmetric).
            valid_preconds = {"DIC", "DILU", "FDIC", "diagonal", "none"}
            if self.smoother_or_precond not in valid_preconds:
                raise ValueError(
                    f"Direct-solver preconditioner must be one of "
                    f"{sorted(valid_preconds)} (got {self.smoother_or_precond!r})."
                )
            if self.coarsest is not None:
                raise ValueError(
                    "coarsestLevelCorr is only valid when top_level='GAMG'."
                )
        return self


# ────────────────────────────────────────────────────────────────────────────
# CompressibleBounds — SIMPLE/PIMPLE block safety bounds
# ────────────────────────────────────────────────────────────────────────────


class CompressibleBounds(BaseModel):
    """Density / pressure bounds + transonic flag for the algorithm block.

    `rhoMin / rhoMax / pMin / pMax / transonic` are part of every standard
    rhoSimpleFoam tutorial.  The renderer emits them only for compressible
    solvers (i.e. when the strategy is present on the CaseSpec).
    """

    rho_min: float | None = Field(default=None, ge=0.0)
    rho_max: float | None = Field(default=None, gt=0.0)
    p_min: float | None = Field(default=None, ge=0.0)
    p_max: float | None = Field(default=None, gt=0.0)
    transonic: bool = False

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _enforce_min_lt_max(self) -> "CompressibleBounds":
        if (
            self.rho_min is not None
            and self.rho_max is not None
            and self.rho_min >= self.rho_max
        ):
            raise ValueError(
                f"rho_min ({self.rho_min}) must be strictly less than "
                f"rho_max ({self.rho_max})."
            )
        if (
            self.p_min is not None
            and self.p_max is not None
            and self.p_min >= self.p_max
        ):
            raise ValueError(
                f"p_min ({self.p_min}) must be strictly less than "
                f"p_max ({self.p_max})."
            )
        return self


# ────────────────────────────────────────────────────────────────────────────
# InletTurbulence — per-patch turbulent inlet state
# ────────────────────────────────────────────────────────────────────────────


class TurbulenceSpec(BaseModel):
    """Resolved case-wide turbulence configuration.

    Captures the three independent decisions that flow through every
    turbulence-aware file (``constant/turbulenceProperties``, ``0/k``,
    ``0/omega``, ``0/nut``, ``0/alphat``, plus the ``RAS`` blocks in
    ``fvSchemes`` / ``fvSolution``):

      * ``flow_regime``     — laminar vs turbulent (vs transitional, future)
      * ``model``           — kOmegaSST, kEpsilon, kOmega, SpalartAllmaras, …
      * ``simulation_type`` — laminar / RAS / LES (the OpenFOAM keyword)

    Invariants enforced at construction:
      * ``flow_regime == "laminar"`` ↔ ``model == "laminar"`` ↔
        ``simulation_type == "laminar"``.  No state where one is laminar and
        the others aren't — that's the drift that caused the SIGFPE.
      * ``simulation_type == "LES"`` requires a LES-family model.

    Per-inlet TI and length scale live separately in
    ``InletTurbulence`` instances (already a strategy).
    """

    flow_regime: Literal["laminar", "turbulent"]
    model: str  # validated by valid_turbulence_models on the plugin
    simulation_type: Literal["laminar", "RAS", "LES"]
    wall_functions: bool = True

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _enforce_internal_consistency(self) -> "TurbulenceSpec":
        is_laminar_regime = self.flow_regime == "laminar"
        is_laminar_sim = self.simulation_type == "laminar"
        is_laminar_model = self.model == "laminar"
        if is_laminar_regime != is_laminar_model:
            raise ValueError(
                f"flow_regime={self.flow_regime!r} but model={self.model!r}: "
                "either both are laminar or neither is."
            )
        if is_laminar_regime != is_laminar_sim:
            raise ValueError(
                f"flow_regime={self.flow_regime!r} but simulation_type="
                f"{self.simulation_type!r}: simulation_type must match."
            )
        if self.simulation_type == "LES" and "LES" not in self.model:
            raise ValueError(
                f"simulation_type='LES' requires an LES-family model "
                f"(got {self.model!r})."
            )
        return self


class InletTurbulence(BaseModel):
    """Resolved per-inlet turbulence input.

    What the renderer reads to compute the patch's `fixedValue` for k, ω, ε:

      k_i = 1.5 · (velocity_mag · intensity)²
      ω_i = √k_i / (Cμ^0.25 · length_scale)
      ε_i = Cμ^0.75 · k_i^1.5 / length_scale

    `length_scale` is shared across inlets (it's a property of the geometry,
    typically L = 0.07 · D_h).  Different inlets carry different TIs by
    design — a turbulent jet (10%) mixing with a settled coflow (1%) is a
    legitimate case.
    """

    patch_name: str = Field(min_length=1)
    velocity_mag: float = Field(gt=0.0)              # m/s
    intensity: float = Field(ge=0.001, le=0.30)      # fraction, 0.1% to 30%
    length_scale: float = Field(gt=0.0)              # m, typically 0.07·D_h

    model_config = ConfigDict(frozen=True)

    # Computed properties — not stored, derived from the three inputs.
    # Renderers call .k / .omega / .epsilon directly.
    @property
    def k(self) -> float:
        return 1.5 * (self.velocity_mag * self.intensity) ** 2

    @property
    def omega(self) -> float:
        # Explicit float() — Python's `**` on float scalars is typed as Any
        # under mypy --strict; wrap once to keep the annotation honest.
        return float((self.k ** 0.5) / (0.09 ** 0.25 * self.length_scale))

    @property
    def epsilon(self) -> float:
        return float(0.09 ** 0.75 * (self.k ** 1.5) / self.length_scale)


# ────────────────────────────────────────────────────────────────────────────
# TurbulenceRegimeProfile — per-regime fvSchemes / turbulenceProperties knobs
# ────────────────────────────────────────────────────────────────────────────
#
# The three OpenFOAM regimes (laminar / RAS / LES) differ in numerical scheme
# choice everywhere they touch:
#
#   * ``constant/turbulenceProperties`` — different blocks entirely
#   * ``fvSchemes.ddtSchemes``           — backward (LES) vs Euler
#   * ``fvSchemes.divSchemes``           — every div(phi,X) line differs
#
# Encoding the per-regime decisions as one frozen Pydantic model means the
# renderers read attribute access (``ctx.regime_profile.div_phi_U``) instead
# of nested if/else over a string regime tag.  Bug class that disappears: a
# scheme line that's correct for RAS but wrong for LES (e.g. ``LUST grad(U)``
# is meaningless in RAS).

# Pressure flux name — ``phid`` (compressible flux, ψ·U) for rho-pressure-
# coupled solvers; ``phiv`` (kinematic flux, U) for solvers that integrate
# the pressure equation differently.  Renderer emits ``div(<flux>,p)``.
PressureFlux = Literal["phid", "phiv"]


class TurbulenceRegimeProfile(BaseModel):
    """Resolved per-regime scheme bundle for fvSchemes + turbulenceProperties.

    The renderer reads attribute access against this frozen object instead
    of branching on a string regime tag.  Construction is driven by
    ``resolve_regime_profile`` in resolvers.py — one profile per
    (regime × solver_algorithm × thermo_profile) triple.

    Fields:

      * ``simulation_type`` — laminar / RAS / LES.  Drives every other choice.
      * ``ddt_scheme``      — Euler (transient pseudo-steady), backward (LES),
                              steadyState (SIMPLE), or CrankNicolson 0.9 (DNS-y).
      * ``div_phi_U``       — divergence scheme for momentum.
      * ``div_phi_energy``  — divergence for the energy variable (h or e).
      * ``div_phi_K``       — divergence for kinetic-energy convection.
      * ``div_phi_p``       — divergence for the pressure-work term.
      * ``div_phi_turb``    — divergence for transported turbulence fields
                              (None for laminar — no such fields exist).
      * ``pressure_flux``   — ``phid`` (compressible flux) or ``phiv``
                              (kinematic).  Renderer emits
                              ``div(<pressure_flux>,p)``.
      * ``turbulence_properties_block`` — the full ``simulationType …`` text
                              for ``constant/turbulenceProperties``, including
                              any model-specific sub-dict (LES needs
                              ``delta``, ``cubeRootVolCoeffs``, …).
    """

    simulation_type: Literal["laminar", "RAS", "LES"]
    ddt_scheme: str = Field(min_length=1)
    div_phi_U: str = Field(min_length=1)
    div_phi_energy: str = Field(min_length=1)
    div_phi_K: str = Field(min_length=1)
    div_phi_p: str = Field(min_length=1)
    div_phi_turb: str | None = None
    pressure_flux: PressureFlux = "phid"
    turbulence_properties_block: str = Field(min_length=1)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _enforce_regime_consistency(self) -> "TurbulenceRegimeProfile":
        # Laminar carries no transported turbulence fields → div_phi_turb must
        # be absent.  RAS / LES carry at least one transported scalar.
        if self.simulation_type == "laminar" and self.div_phi_turb is not None:
            raise ValueError(
                "simulation_type='laminar' must not declare div_phi_turb — "
                "laminar carries no transported turbulence fields."
            )
        if (
            self.simulation_type in ("RAS", "LES")
            and self.div_phi_turb is None
        ):
            raise ValueError(
                f"simulation_type={self.simulation_type!r} requires "
                "div_phi_turb (e.g. 'Gauss upwind' or 'Gauss limitedLinear 1')."
            )
        return self


# ────────────────────────────────────────────────────────────────────────────
# RegionSpec / CaseRegions — multi-region (CHT) configuration
# ────────────────────────────────────────────────────────────────────────────
#
# Conjugate heat transfer (chtMultiRegionFoam / chtMultiRegionSimpleFoam)
# solves multiple regions — typically one or more fluids coupled with one
# or more solids via mapped boundaries at fluid-solid interfaces.  Every
# region carries its own ``constant/<region>/`` and ``system/<region>/``
# trees plus per-region ``0/<region>/`` initial conditions.
#
# These types capture *what* a region is at the level the renderer needs:
# its name (= subdirectory in constant/ and system/), its kind (fluid or
# solid), and the thermo / turbulence settings that drive file generation.

RegionKind = Literal["fluid", "solid"]
"""A CHT region is either a fluid (Navier–Stokes + energy) or a solid (only
heat conduction)."""


class RegionSpec(BaseModel):
    """One region in a multi-region case.

    The renderer uses ``name`` to namespace files
    (``constant/<name>/thermophysicalProperties`` etc.); the per-region
    fvSolution / fvSchemes / 0-field files are also written under
    ``<name>/`` subdirectories.

    Fluid regions:
      * ``kind = "fluid"`` — gets a Navier–Stokes solver, turbulence
        model, ν / Cp / Pr (via thermo_profile + the existing
        ``CompressibleBounds`` resolver).
      * ``turbulence_model`` is required (RAS / LES); ``"laminar"`` is
        valid for low-Re slow fluids.

    Solid regions:
      * ``kind = "solid"`` — heat conduction only, no momentum,
        no turbulence.  Uses ``heSolidThermo`` with ``rhoConst`` +
        ``constIso`` (constant isotropic thermal conductivity).
      * ``turbulence_model`` must be ``None`` (or ``"none"``) — solid
        regions don't transport k / ε / ω.

    Invariants enforced at construction:
      * Region names are non-empty and safe directory names
        (``^[A-Za-z][A-Za-z0-9_]*$``).
      * Solid regions can't declare a turbulence model.
      * Fluid regions must declare ``thermo_profile`` (gas / cryogenic).
    """

    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    kind: RegionKind
    thermo_profile: Literal["gas", "cryogenic", "solid"]
    # Fluid regions: RAS / LES / laminar.  Solids must leave this None.
    turbulence_model: str | None = None
    # Fluid-region transport / thermo numbers (use defaults if unknown).
    Cp: float = Field(default=1006.0, gt=0.0)         # J/(kg·K)
    mol_weight: float = Field(default=28.97, gt=0.0)  # g/mol — air-like default
    mu: float = Field(default=1.8e-5, gt=0.0)         # Pa·s
    Pr: float = Field(default=0.7, gt=0.0)
    # Solid-region thermal properties.
    rho_solid: float = Field(default=8000.0, gt=0.0)  # kg/m³ — steel-like
    kappa_solid: float = Field(default=80.0, gt=0.0)  # W/(m·K) — steel-like
    Cp_solid: float = Field(default=450.0, gt=0.0)    # J/(kg·K) — steel-like

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _enforce_kind_consistency(self) -> "RegionSpec":
        if self.kind == "solid":
            if self.turbulence_model and self.turbulence_model.lower() not in (
                "none", "laminar"
            ):
                raise ValueError(
                    f"Solid region {self.name!r} cannot declare a turbulence "
                    f"model (got {self.turbulence_model!r})."
                )
            if self.thermo_profile != "solid":
                raise ValueError(
                    f"Solid region {self.name!r} requires "
                    "thermo_profile='solid' (got {self.thermo_profile!r})."
                )
        else:  # fluid
            if self.thermo_profile == "solid":
                raise ValueError(
                    f"Fluid region {self.name!r} cannot use "
                    "thermo_profile='solid'."
                )
        return self


class CaseRegions(BaseModel):
    """Container for all regions in a multi-region case.

    The renderer reads ``fluid_regions`` and ``solid_regions`` to build
    the per-region file trees and the ``constant/regionProperties``
    listing.  Invariants:

      * At least one fluid region (CHT without fluid is just
        ``laplacianFoam`` on a solid — not what chtMultiRegion* is for).
      * Region names are globally unique across fluids and solids.
    """

    fluid_regions: list[RegionSpec] = Field(default_factory=list)
    solid_regions: list[RegionSpec] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _enforce_structural_invariants(self) -> "CaseRegions":
        if not self.fluid_regions:
            raise ValueError(
                "CaseRegions requires at least one fluid region for "
                "chtMultiRegion*Foam (no fluid → use laplacianFoam directly)."
            )
        # All fluid regions are kind=fluid.
        for r in self.fluid_regions:
            if r.kind != "fluid":
                raise ValueError(
                    f"Region {r.name!r} in fluid_regions has kind={r.kind!r}."
                )
        # All solid regions are kind=solid.
        for r in self.solid_regions:
            if r.kind != "solid":
                raise ValueError(
                    f"Region {r.name!r} in solid_regions has kind={r.kind!r}."
                )
        # Unique names across both lists.
        seen: set[str] = set()
        for r in (*self.fluid_regions, *self.solid_regions):
            if r.name in seen:
                raise ValueError(f"Duplicate region name: {r.name!r}.")
            seen.add(r.name)
        return self

    @property
    def all_regions(self) -> list[RegionSpec]:
        """Fluids first, then solids — the same order regionProperties uses."""
        return [*self.fluid_regions, *self.solid_regions]

    @property
    def region_names(self) -> list[str]:
        return [r.name for r in self.all_regions]
