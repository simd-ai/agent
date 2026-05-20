# tests/test_multi_region_isolation.py
"""Guard tests: no single-region function is allowed to run for a CHT case.

The single-region validator / file-manifest / heuristic-solver paths are
written for the flat single-region case tree (``0/<field>``,
``system/<dict>``, ``constant/<dict>``).  Letting them touch a
multi-region (CHT) case produces silent damage — they stamp
"missing top-level ``0/T``" warnings, rewrite per-region boundary lists,
or worst case strip files the deterministic per-region renderer emitted.

These tests pin the defensive layout that prevents that:

  1. :func:`simd_agent.solvers.is_multi_region_solver` is the single
     source of truth and resolves correctly for every CHT plugin.
  2. :meth:`SolverPlugin.validate_full` does NOT call the single-region
     :func:`validate_generated_files` when ``plugin.is_multi_region`` is True.
  3. Each single-region entry point self-defends — calling it directly
     with a CHT solver returns a safe no-op rather than corrupting input.

Add a new check here for every new single-region helper that gains
public callers, so the multi-region pipeline can never silently
inherit single-region behaviour.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from simd_agent.solvers import get_registry, is_multi_region_solver


# ── 1. is_multi_region_solver dispatch ────────────────────────────────────

@pytest.mark.parametrize("solver, expected", [
    ("chtMultiRegionSimpleFoam", True),
    ("chtMultiRegionFoam", True),
    ("simpleFoam", False),
    ("pimpleFoam", False),
    ("rhoSimpleFoam", False),
    ("rhoPimpleFoam", False),
    ("buoyantSimpleFoam", False),
    ("buoyantBoussinesqSimpleFoam", False),
    (None, False),
    ("", False),
    ("not_a_real_solver", False),
])
def test_is_multi_region_solver_dispatch(solver, expected):
    assert is_multi_region_solver(solver) is expected


# ── 2. validate_full does NOT invoke the single-region validator for CHT ──

def _minimal_cht_config():
    return {
        "regions": {
            "fluid": [
                {"name": "innerFluid", "fluid_preset": "air", "interfaces": ["wall"]},
            ],
            "solid": [
                {"name": "wall", "solid_preset": "stainless", "interfaces": ["innerFluid"]},
            ],
        },
        "mesh": {"patches": [
            {"name": "innerFluid_inlet", "type": "patch"},
            {"name": "innerFluid_outlet", "type": "patch"},
            {"name": "wall_left_end", "type": "patch"},
            {"name": "wall_right_end", "type": "patch"},
            {"name": "front", "type": "empty"},
            {"name": "back", "type": "empty"},
        ]},
        "physics": {"time_scheme": "steady"},
    }


@pytest.mark.parametrize("cht_solver", [
    "chtMultiRegionSimpleFoam",
    "chtMultiRegionFoam",
])
def test_validate_full_skips_single_region_validator_for_cht(cht_solver):
    """Plugin.validate_full() must not call validate_generated_files for CHT.

    base.py:validate_full gates the single-region validator on
    ``not self.is_multi_region``; this test catches a regression where
    that guard is removed.
    """
    plugin = get_registry().get(cht_solver)
    assert plugin is not None, f"{cht_solver} plugin not registered"
    assert plugin.is_multi_region is True

    cfg = _minimal_cht_config()
    if cht_solver == "chtMultiRegionFoam":
        cfg["physics"]["time_scheme"] = "transient"

    files = {"system/controlDict": f"application {cht_solver};\n"}

    with patch(
        "simd_agent.run.single_region.validator.validate_generated_files"
    ) as spy:
        spy.return_value = (files, [])
        result = plugin.validate_full(files, cfg)

    assert spy.call_count == 0, (
        f"validate_full called the single-region validator for "
        f"multi-region solver {cht_solver!r} ({spy.call_count} time(s)). "
        f"This produces silent damage to the per-region case tree."
    )

    # Deterministic renderer must still have populated the per-region tree.
    assert "constant/regionProperties" in result.files
    assert "system/fvSchemes" in result.files  # top-level placeholder
    assert "system/fvSolution" in result.files  # top-level placeholder
    assert "system/innerFluid/fvSchemes" in result.files

    # Top-level constant/g — ESI v2406's chtMultiRegionSimpleFoam reads
    # gravity from case/constant/g (top level) with MUST_READ; the case
    # aborts on startup if missing.  Foundation OpenFOAM only reads the
    # per-region copy below.  We emit both for build-family compatibility.
    assert "constant/g" in result.files, (
        "Top-level constant/g is required by ESI v2406's "
        "chtMultiRegionSimpleFoam — emit it alongside per-region copies."
    )
    assert "dimensions      [0 1 -2 0 0 0 0]" in result.files["constant/g"]
    assert "(0 -9.81 0)" in result.files["constant/g"]
    # The location header should NOT name a region (it's a top-level file)
    assert 'location    "constant"' in result.files["constant/g"], (
        "Top-level constant/g must have location=\"constant\", not a "
        "region-tagged location copied from the per-region builder."
    )

    # Per-region gravity must still be there for Foundation-style lookups.
    assert "constant/innerFluid/g" in result.files


# ── 3. Self-defending early-returns inside single-region entry points ─────

@pytest.mark.parametrize("cht_solver", [
    "chtMultiRegionSimpleFoam",
    "chtMultiRegionFoam",
])
def test_validate_generated_files_noops_for_cht(cht_solver):
    """An unguarded import-and-call must not damage a CHT case."""
    from simd_agent.run.single_region import validate_generated_files

    files = {
        "0/innerFluid/T": "...",
        "0/innerFluid/U": "...",
        "system/controlDict": "...",
        "constant/regionProperties": "...",
    }
    out_files, out_issues = validate_generated_files(files, cht_solver, {})

    assert out_files == files, "single-region validator must pass through CHT files unchanged"
    assert out_issues == [], "single-region validator must emit no issues for CHT"


@pytest.mark.parametrize("cht_solver, time_scheme", [
    ("chtMultiRegionSimpleFoam", "steady"),
    ("chtMultiRegionFoam", "transient"),
])
def test_build_required_files_list_pins_cht_to_controldict(cht_solver, time_scheme):
    """For CHT the LLM only owns system/controlDict; everything else is deterministic."""
    from simd_agent.run.single_region import build_required_files_list

    cfg = _minimal_cht_config()
    cfg["physics"]["time_scheme"] = time_scheme
    required = build_required_files_list(cht_solver, cfg)
    assert required == ["system/controlDict"], (
        f"CHT required-files manifest must be exactly ['system/controlDict']; "
        f"got {required!r}"
    )


@pytest.mark.parametrize("time_scheme, expected_solver", [
    ("steady", "chtMultiRegionSimpleFoam"),
    ("transient", "chtMultiRegionFoam"),
])
def test_determine_solver_short_circuits_to_cht_for_multi_region(
    time_scheme, expected_solver,
):
    """determine_solver (single-region heuristic) must short-circuit to CHT
    when ``config["regions"]`` has both fluid and solid lists — otherwise it
    would run single-region physics rules on multi-region topology and pick
    a wrong solver (typically buoyantSimpleFoam)."""
    from simd_agent.run.single_region import determine_solver

    cfg = _minimal_cht_config()
    cfg["physics"]["time_scheme"] = time_scheme
    assert determine_solver(cfg) == expected_solver


# ── 4. Single-region path is left alone ────────────────────────────────────

@pytest.mark.parametrize("field", ["p", "p_rgh", "U", "T"])
def test_constraint_patches_emit_matching_bc_type(field):
    """Symmetry / empty / wedge patches MUST emit the matching constraint
    patchField type (no ``value`` entry) — anything else triggers an
    "inconsistent patch and patchField types" fatal at solver startup.

    The renderer infers the role from the patch name suffix (``_symmetry``)
    and the mesh-side type (``empty`` / ``wedge``).  This test pins both
    paths so a future refactor of the per-field BC pickers can't
    accidentally drop a fall-through that emits ``calculated`` or
    ``zeroGradient`` on a constraint patch.
    """
    plugin = get_registry().get("chtMultiRegionSimpleFoam")
    cfg = _minimal_cht_config()
    # Add a name-suffix-based symmetry patch + mesh-type-based empty patches.
    cfg["mesh"]["patches"] = [
        {"name": "innerFluid_inlet", "type": "patch"},
        {"name": "innerFluid_outlet", "type": "patch"},
        {"name": "innerFluid_symmetry", "type": "patch"},  # gmshToFoam default
        {"name": "wall_left_end", "type": "patch"},
        {"name": "wall_right_end", "type": "patch"},
        {"name": "front", "type": "empty"},
        {"name": "back", "type": "empty"},
    ]

    files = plugin.render_deterministic_files(cfg)

    # Choose the field file from a region that owns this patch family.
    region = "wall" if field == "T" else "innerFluid"
    fpath = f"0/{region}/{field}"
    content = files.get(fpath, "")
    assert content, f"Missing expected file {fpath}"

    # Symmetry patch is owned by innerFluid (T tests skip — wall's T file
    # doesn't enumerate innerFluid patches).
    if region == "innerFluid":
        sym_block = _extract_patch_block(content, "innerFluid_symmetry")
        assert sym_block, f"innerFluid_symmetry block missing from {fpath}"
        assert "type            symmetry;" in sym_block, (
            f"innerFluid_symmetry must use 'type symmetry;' in {fpath}; "
            f"got:\n{sym_block}"
        )
        # Constraint patches take NO value entry.
        assert "value" not in sym_block, (
            f"Constraint patch innerFluid_symmetry must not have a "
            f"'value' entry in {fpath}; got:\n{sym_block}"
        )

    # front / back are mesh-type "empty" — picked up via the shared
    # constraint-patch branch in region_patches().  They appear in every
    # region's BC file.
    for empty_patch in ("front", "back"):
        block = _extract_patch_block(content, empty_patch)
        if block:  # only assert when the renderer included the patch
            assert "type            empty;" in block, (
                f"{empty_patch} must use 'type empty;' in {fpath}; "
                f"got:\n{block}"
            )
            assert "value" not in block, (
                f"Constraint patch {empty_patch} must not have a 'value' "
                f"entry in {fpath}; got:\n{block}"
            )


def _extract_patch_block(content: str, patch_name: str) -> str:
    """Return the body of one boundary-field block (between ``{`` and ``}``)."""
    import re
    m = re.search(
        rf"{re.escape(patch_name)}\s*\{{([^{{}}]*)\}}",
        content,
    )
    return m.group(1) if m else ""


def test_solid_region_ships_p_p_rgh_U_for_esi_solver_lookup():
    """ESI v2406's chtMultiRegionSimpleFoam walks every region's
    objectRegistry at startup and aborts when ``0/<solid>/{p,p_rgh,U}``
    are missing.  Foundation OpenFOAM-4.x ships top-level templates that
    changeDictionaryDict distributes to solids for the same reason.

    Pin that the renderer emits passive solid-region p / p_rgh / U files:

      * type = ``calculated`` for ``p`` and ``p_rgh`` (NOT
        ``fixedFluxPressure`` — that's fluid-only, it needs the density
        gradient which solid meshes don't compute)
      * type = ``noSlip`` for ``U`` (solids don't move)
      * coupled CHT interface patches use the same passive types.
    """
    plugin = get_registry().get("chtMultiRegionSimpleFoam")
    cfg = _minimal_cht_config()

    files = plugin.render_deterministic_files(cfg)
    for field in ("T", "p", "p_rgh", "U"):
        assert f"0/wall/{field}" in files, (
            f"Solid region 'wall' must ship 0/wall/{field}; "
            f"ESI v2406 chtMultiRegionSimpleFoam aborts at startup without it."
        )

    p_solid = files["0/wall/p"]
    p_rgh_solid = files["0/wall/p_rgh"]
    u_solid = files["0/wall/U"]

    # p / p_rgh: no fixedFluxPressure in the solid file (fluid-only BC).
    assert "fixedFluxPressure" not in p_rgh_solid, (
        "0/<solid>/p_rgh must NOT use fixedFluxPressure — it needs the "
        "density gradient OpenFOAM only computes in fluid meshes.  Use "
        "'calculated' on every patch instead (matches OpenFOAM-4.x "
        "multiRegionHeaterRadiation's distributed template)."
    )

    # Coupled CHT interface block must be present + use calculated/noSlip.
    assert "wall_to_innerFluid" in p_solid
    assert "wall_to_innerFluid" in p_rgh_solid
    assert "wall_to_innerFluid" in u_solid
    # Inside the wall_to_innerFluid block of U, the type must be noSlip
    # (not inletOutlet or other fluid BCs).
    u_block = _extract_patch_block(u_solid, "wall_to_innerFluid")
    assert "noSlip" in u_block, (
        f"Solid coupled U block must use noSlip; got:\n{u_block}"
    )


@pytest.mark.parametrize("preset, expected_rho_min, expected_rho_max", [
    # All bounds = 0.2× / 2.0× nominal density, where nominal is the
    # operating-point density cross-checked against Wikipedia / NIST.
    ("air",    0.241,   2.408),    # nominal 1.2041 kg/m³ at 293.15 K
    ("water",  199.642, 1996.42),  # nominal 998.21 at 293.15 K
    ("oil",    176.0,   1760.0),   # nominal 880 (SAE 30 at 293 K)
    ("helium", 0.0333,  0.3328),   # gas He at 293 K — nominal 0.1664
    ("ln2",    161.6,   1616.0),   # liquid N₂ at 77.36 K — nominal 808
    ("lox",    228.2,   2282.0),   # liquid O₂ at 90.19 K — nominal 1141
    ("lh2",    14.17,   141.7),    # liquid H₂ at 20.27 K — nominal 70.85
    ("lng",    84.56,   845.6),    # liquid methane at 111.65 K — nominal 422.8
    ("lhe",    25.0,    250.0),    # liquid He-4 at 4.2 K — nominal 125
])
def test_rho_bounds_scale_with_fluid_preset(preset, expected_rho_min, expected_rho_max):
    """The SIMPLE block's rhoMin/rhoMax must bracket the operating-point
    density for the selected fluid preset.  Using air-tuned bounds
    (0.2 / 2.0) on a liquid like LOX (ρ ≈ 1141 kg/m³) would clip every
    cell every iteration, stalling the rho-h-p coupling.  These bounds
    were the missing piece that let the previous Regascold run drift
    to ``Min/max rho: 0  9801`` and SIGFPE in compressibleTurbulence::divide.
    """
    import re
    plugin = get_registry().get("chtMultiRegionSimpleFoam")
    cfg = {
        "regions": {
            "fluid": [{"name": "fluidA", "fluid_preset": preset, "interfaces": ["solidS"]}],
            "solid": [{"name": "solidS", "solid_preset": "stainless", "interfaces": ["fluidA"]}],
        },
        "mesh": {"patches": []},
    }
    files = plugin.render_deterministic_files(cfg)
    fvs = files["system/fluidA/fvSolution"]

    rho_min_m = re.search(r"rhoMin\s+([\d.eE+\-]+)\s*;", fvs)
    rho_max_m = re.search(r"rhoMax\s+([\d.eE+\-]+)\s*;", fvs)
    assert rho_min_m, f"rhoMin missing from SIMPLE block for preset {preset!r}"
    assert rho_max_m, f"rhoMax missing from SIMPLE block for preset {preset!r}"

    assert float(rho_min_m.group(1)) == pytest.approx(expected_rho_min, rel=0.01), (
        f"rhoMin for preset {preset!r} should be ~{expected_rho_min} "
        f"(0.2× nominal); got {rho_min_m.group(1)}"
    )
    assert float(rho_max_m.group(1)) == pytest.approx(expected_rho_max, rel=0.01), (
        f"rhoMax for preset {preset!r} should be ~{expected_rho_max} "
        f"(2.0× nominal); got {rho_max_m.group(1)}"
    )


def test_fvSolution_steady_CHT_has_tight_pressure_relaxation():
    """Steady chtMultiRegionSimpleFoam needs ``p_rgh 0.3`` (not the canonical
    0.7) to damp pressure-velocity coupling oscillation on buoyancy-driven
    CHT — observed on the Regascold LN2/water case at 322 steps where
    7/9 fields converged but innerFluid:p_rgh and outerFluid:p_rgh kept
    climbing (OoM drop ≤ 1.1).  Also pin: U 0.3, h 0.7, rho 1.0,
    nNonOrthogonalCorrectors 1 (mesh imperfections), and absence of the
    bogus ``nCorrectors`` SIMPLE key.
    """
    import re
    plugin = get_registry().get("chtMultiRegionSimpleFoam")
    cfg = _minimal_cht_config()
    files = plugin.render_deterministic_files(cfg)
    fvs = files["system/innerFluid/fvSolution"]

    # Tight p_rgh relaxation — the actual fix for buoyancy oscillation.
    p_rgh_relax = re.search(r"p_rgh\s+(0\.[0-9]+)\s*;", fvs)
    assert p_rgh_relax and float(p_rgh_relax.group(1)) <= 0.4, (
        f"Steady CHT p_rgh relaxation must be ≤ 0.4 to damp buoyancy-driven "
        f"oscillation; got {p_rgh_relax.group(1) if p_rgh_relax else 'missing'}"
    )

    # Standard underrelaxation values pinned.
    u_relax = re.search(r"\bU\s+(0\.[0-9]+)\s*;", fvs)
    h_relax = re.search(r"\bh\s+(0\.[0-9]+)\s*;", fvs)
    assert u_relax and float(u_relax.group(1)) <= 0.5
    assert h_relax and float(h_relax.group(1)) <= 0.9

    # fields block has rho 1.0 and p_rgh.
    assert re.search(r"fields\s*\{[^}]*rho\s+1\.0\s*;[^}]*\}", fvs)
    assert re.search(r"fields\s*\{[^}]*p_rgh\s+0\.[0-9]+\s*;[^}]*\}", fvs)

    # Non-orthogonal correctors enabled for mesh-imperfection tolerance.
    simple_m = re.search(r"SIMPLE\s*\{([^}]+)\}", fvs)
    assert simple_m
    non_ortho = re.search(r"nNonOrthogonalCorrectors\s+(\d+)", simple_m.group(1))
    assert non_ortho and int(non_ortho.group(1)) >= 1, (
        f"Steady CHT needs nNonOrthogonalCorrectors ≥ 1 for mesh imperfections; "
        f"got {non_ortho.group(1) if non_ortho else 'missing'}"
    )

    # SIMPLE must NOT carry PIMPLE-only keys.
    assert "nCorrectors" not in simple_m.group(1), (
        "SIMPLE block leaked a PIMPLE-only key (nCorrectors / nOuterCorrectors)"
    )
    assert "nOuterCorrectors" not in simple_m.group(1)


def test_fvSolution_transient_CHT_uses_pimple_outer_correctors():
    """Transient chtMultiRegionFoam uses PIMPLE with outer correctors,
    NOT the steady SIMPLE keys.  Pin:

      * ``nOuterCorrectors`` ≥ 2 (the inner mini-SIMPLE loop)
      * ``nCorrectors`` (PISO pressure correctors per outer)
      * residualControl block (early-exit on convergence)
      * ``*Final`` relaxation = 1.0 (canonical PIMPLE convention — last
        outer corrector must solve the un-relaxed equation for stability)
      * NO tight ``p_rgh 0.3`` (the outer correctors handle the coupling;
        static 0.7 is the right static relaxation for transient)
    """
    import re
    plugin = get_registry().get("chtMultiRegionFoam")
    cfg = _minimal_cht_config()
    cfg["physics"]["time_scheme"] = "transient"
    files = plugin.render_deterministic_files(cfg)
    fvs = files["system/innerFluid/fvSolution"]

    # PIMPLE block, not SIMPLE
    assert re.search(r"^PIMPLE\s*\{", fvs, re.MULTILINE), (
        "Transient CHT must emit a PIMPLE block, not SIMPLE"
    )
    assert not re.search(r"^SIMPLE\s*\{", fvs, re.MULTILINE), (
        "Transient CHT must NOT emit a SIMPLE block"
    )

    pimple_m = re.search(r"PIMPLE\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}", fvs)
    assert pimple_m
    block = pimple_m.group(1)

    n_outer = re.search(r"nOuterCorrectors\s+(\d+)", block)
    assert n_outer and int(n_outer.group(1)) >= 2, (
        f"PIMPLE outer-corrector count must be ≥ 2 to handle the "
        f"rho-h-p coupling per timestep; got {n_outer.group(1) if n_outer else 'missing'}"
    )

    n_inner = re.search(r"nCorrectors\s+(\d+)", block)
    assert n_inner, "PIMPLE block must declare nCorrectors (PISO inner step)"

    non_ortho = re.search(r"nNonOrthogonalCorrectors\s+(\d+)", block)
    assert non_ortho and int(non_ortho.group(1)) >= 1, (
        "PIMPLE needs nNonOrthogonalCorrectors ≥ 1 for mesh tolerance"
    )

    # residualControl — early exit on convergence (saves compute on
    # well-behaved time steps).
    assert "residualControl" in block, (
        "PIMPLE block should include residualControl for early-exit"
    )

    # Final-iteration relaxation = 1.0 (OpenFOAM canonical PIMPLE convention).
    assert re.search(r"UFinal\s+1\.0\s*;", fvs), (
        "PIMPLE must reset U relaxation to 1.0 on the final outer corrector"
    )
    assert re.search(r"hFinal\s+1\.0\s*;", fvs), (
        "PIMPLE must reset h relaxation to 1.0 on the final outer corrector"
    )
    assert re.search(r"p_rghFinal\s+1\.0\s*;", fvs), (
        "PIMPLE must reset p_rgh relaxation to 1.0 on the final outer corrector"
    )

    # Static relaxation is the looser PIMPLE value (~0.7), NOT the
    # SIMPLE-tight 0.3 — the outer correctors do the damping.
    p_rgh_static = re.search(r'"p_rgh\.\*"\s+(0\.[0-9]+)\s*;', fvs)
    assert p_rgh_static and float(p_rgh_static.group(1)) >= 0.5, (
        f"Transient PIMPLE p_rgh static relaxation should be 0.5-0.7 "
        f"(outer correctors handle coupling); got "
        f"{p_rgh_static.group(1) if p_rgh_static else 'missing'}"
    )


def test_fvSolution_steady_and_transient_share_density_bounds():
    """Both algorithms must emit ``rhoMin``/``rhoMax`` bounds derived from
    the fluid preset — same safety contract regardless of SIMPLE vs PIMPLE.
    Without bounds, density can drift to 0 and SIGFPE in
    ``compressibleTurbulenceModel::divide`` on the first ill-conditioned
    step.
    """
    import re
    cfg = _minimal_cht_config()
    for solver, scheme in [
        ("chtMultiRegionSimpleFoam", "steady"),
        ("chtMultiRegionFoam",       "transient"),
    ]:
        cfg["physics"]["time_scheme"] = scheme
        plugin = get_registry().get(solver)
        files = plugin.render_deterministic_files(cfg)
        fvs = files["system/innerFluid/fvSolution"]

        rho_min = re.search(r"rhoMin\s+([\d.eE+\-]+)", fvs)
        rho_max = re.search(r"rhoMax\s+([\d.eE+\-]+)", fvs)
        assert rho_min and rho_max, (
            f"{solver} must emit rhoMin/rhoMax safety bounds (it's compressible)"
        )
        # Sane values: lower bound < nominal < upper bound; both positive.
        lo, hi = float(rho_min.group(1)), float(rho_max.group(1))
        assert 0 < lo < hi, f"{solver} bounds nonsensical: {lo} → {hi}"


def test_build_sim_progress_flattens_cht_regions_for_chart():
    """The runner ships CHT per-region residuals under ``p["regions"]`` and
    leaves top-level ``residuals`` / ``continuity`` empty.  Without
    flattening, the frontend's live-convergence chart receives an empty
    ``residuals`` dict and renders blank — observed in the wild as a
    fully-iterating chtMultiRegion run with no chart data.

    This pins the agent-side flattener:

      1. Per-region residuals are flattened into namespaced ``{region}:{field}``
         keys so the existing residual chart shows them as separate lines.
      2. The worst-case (max ``|cumulative|``) region's continuity surfaces
         as the top-level summary so the "Continuity sum" tile has a value.
      3. The full per-region payload is forwarded under ``regions`` so any
         future per-region UI component can consume it without another
         shape change.
    """
    from simd_agent.run.orchestration import build_sim_progress

    payload = {
        "iteration": 1, "simTime": 1.0,
        "residuals": {}, "continuity": None, "courant": None,
        "execution": {"stepSeconds": 0.61, "clockSeconds": 1.0, "label": "0.61s"},
        "regions": {
            "innerFluid": {
                "kind": "fluid",
                "residuals": {
                    "Ux":    {"initial": 1.0, "final": 0.0862687, "iters": 15},
                    "p_rgh": {"initial": 0.997, "final": 0.0068, "iters": 5},
                },
                "continuity": {"local": 9.54, "global": 1.16, "cumulative": 1.16},
            },
            "outerFluid": {
                "kind": "fluid",
                "residuals": {"Ux": {"initial": 1.0, "final": 0.088, "iters": 18}},
                # outerFluid has the worst-case cumulative — must surface to top.
                "continuity": {"local": 18.06, "global": 2.38, "cumulative": 3.54},
            },
            "wall": {
                "kind": "solid",
                "residuals": {"h": {"initial": 1.0, "final": 0.075, "iters": 6}},
                "continuity": None,
            },
        },
    }

    # Runner ships fields=[] for CHT (all residuals live under regions/),
    # which is what a real CHT payload looks like.  Force this state to
    # pin the fix: ``fields`` must NOT pass through as empty.
    payload["fields"] = []

    out = build_sim_progress(payload)

    # 1. Flattened residual keys for the chart.
    assert set(out["residuals"].keys()) == {
        "innerFluid:Ux", "innerFluid:p_rgh",
        "outerFluid:Ux",
        "wall:h",
    }
    # Values are correctly normalised.
    assert out["residuals"]["innerFluid:Ux"] == {
        "initial": 1.0, "final": 0.0862687, "iters": 15,
    }

    # 1b. ``fields`` MUST be populated from the flattened residual keys
    # when the runner ships an empty list.  Without this, the frontend's
    # ``LiveTab.tsx`` reads ``latest.fields`` (empty) and the residual
    # chart has zero series to draw — the original "blank chart for CHT"
    # bug.  Test runs even though convergence has its own fallback,
    # because the chart path doesn't.
    assert len(out["fields"]) > 0, (
        "build_sim_progress must populate 'fields' from residuals when "
        "the runner-supplied fields list is empty — otherwise the "
        "frontend chart receives nothing to plot."
    )
    assert set(out["fields"]) == set(out["residuals"].keys()), (
        "'fields' must match the keys of 'residuals' so the chart's "
        "series enumeration is consistent with the data."
    )

    # 2. Continuity summary surfaces the worst region (outerFluid: cum=3.54).
    assert out["continuity"]["cumulative"] == 3.54

    # 3. Per-region payload still forwarded for drill-in UIs.
    assert "regions" in out
    assert set(out["regions"].keys()) == {"innerFluid", "outerFluid", "wall"}
    assert out["regions"]["wall"]["kind"] == "solid"
    assert out["regions"]["innerFluid"]["residuals"]["Ux"]["final"] == 0.0862687


@pytest.mark.parametrize("field, expected_relax_key", [
    # Single-region (existing behaviour, must keep working)
    ("Ux", "U"),
    ("Uy", "U"),
    ("Uz", "U"),
    ("p", "p"),
    ("h", "h"),
    ("k", "k"),
    # Multi-region namespaced fields produced by build_sim_progress —
    # region prefix must be stripped before the Ux/Uy/Uz → U collapse,
    # otherwise the recommendation engine never matches CHT fields and
    # never proposes relaxation-factor changes.
    ("innerFluid:Ux", "U"),
    ("outerFluid:Uy", "U"),
    ("innerFluid:Uz", "U"),
    ("innerFluid:p_rgh", "p_rgh"),
    ("wall:h", "h"),
    ("outerFluid:k", "k"),
])
def test_field_to_relax_key_handles_cht_namespaced_fields(field, expected_relax_key):
    """Multi-region residual keys carry a ``<region>:`` prefix.  The
    relax-key mapper must strip the prefix so recommendations like
    "increase U relaxation" fire for CHT cases just like single-region.
    """
    from simd_agent.convergence import _field_to_relax_key
    assert _field_to_relax_key(field) == expected_relax_key


def test_build_sim_progress_single_region_passthrough():
    """Single-region runs must be unaffected by the CHT flattener — the
    top-level ``residuals`` / ``continuity`` flow straight through, and
    the result must NOT carry a ``regions`` key (no per-region data exists).
    """
    from simd_agent.run.orchestration import build_sim_progress

    out = build_sim_progress({
        "iteration": 5, "simTime": 5.0,
        "residuals": {"Ux": 0.001, "p": 0.0001},
        "continuity": {"local": 1e-6, "global": 1e-7, "cumulative": 1e-7},
        "courant": {"mean": 0.5, "max": 1.2},
    })

    assert sorted(out["residuals"].keys()) == ["Ux", "p"]
    assert ":" not in "".join(out["residuals"].keys()), (
        "Single-region residual keys must not be namespaced"
    )
    assert out["continuity"]["cumulative"] == 1e-7
    assert out["courant"]["max"] == 1.2
    assert "regions" not in out, (
        "Single-region build_sim_progress must NOT carry a 'regions' key"
    )


@pytest.mark.parametrize("field, expected", [
    # Single-region (existing behaviour must be preserved)
    ("Ux",                  1e-5),
    ("Uy",                  1e-5),
    ("Uz",                  1e-5),
    ("p",                   1e-4),
    ("p_rgh",               1e-4),
    ("h",                   1e-6),
    ("T",                   1e-6),
    ("k",                   1e-3),
    ("epsilon",             1e-3),
    ("omega",               1e-3),
    # Multi-region — namespaced keys must hit the same thresholds
    # via _bare_field_name().  Without this fix, every namespaced field
    # fell through to the 1e-4 default, which is much looser than
    # physics requires (Ux: 1e-5, h: 1e-6).
    ("innerFluid:Ux",       1e-5),
    ("outerFluid:Uy",       1e-5),
    ("innerFluid:h",        1e-6),
    ("wall:h",              1e-6),
    ("innerFluid:p_rgh",    1e-4),
    ("outerFluid:k",        1e-3),
    # Unknown field → default fallback (single + multi)
    ("nonsense",            1e-4),
    ("region:nonsense",     1e-4),
])
def test_threshold_for_handles_namespaced_fields(field, expected):
    """Convergence threshold lookup must operate on the bare physics name,
    not the ``<region>:<field>`` namespaced key the multi-region pipeline
    produces.  Otherwise CHT cases were using 1e-4 as the convergence
    threshold for every field — too loose for velocity (true: 1e-5) and
    energy (true: 1e-6), causing false-converged status.
    """
    from simd_agent.convergence import _threshold_for
    assert _threshold_for(field) == expected


def test_overall_status_critical_fields_respect_cht_namespacing():
    """When multi-region fields like ``innerFluid:Ux`` show up in the
    field assessments, the critical-fields check must still gate on
    pressure + momentum convergence.  Before the fix the check vacuously
    passed because no namespaced key matched the bare ``Ux``/``p_rgh``
    set, so the overall status fell through to whatever the turbulence
    fields happened to report.
    """
    from simd_agent.convergence import _compute_overall_status

    # innerFluid:Ux still oscillating, everything else converged →
    # overall should be "oscillating" (critical field unsettled).
    fields = [
        {"field": "innerFluid:Ux",    "status": "oscillating",  "lastResidual": 1e-3, "threshold": 1e-5},
        {"field": "innerFluid:Uy",    "status": "converged",    "lastResidual": 1e-6, "threshold": 1e-5},
        {"field": "innerFluid:p_rgh", "status": "converged",    "lastResidual": 1e-5, "threshold": 1e-4},
        {"field": "innerFluid:h",     "status": "converged",    "lastResidual": 1e-7, "threshold": 1e-6},
        {"field": "wall:h",           "status": "converged",    "lastResidual": 1e-7, "threshold": 1e-6},
    ]
    status = _compute_overall_status(fields, continuity=None, courant=None)
    assert status == "oscillating", (
        f"Critical field innerFluid:Ux still oscillating → overall must "
        f"be 'oscillating'; got {status!r}"
    )

    # Now all critical fields (innerFluid:Ux + outerFluid:Ux + …) converged
    # → overall converged.
    fields_all_ok = [
        {"field": "innerFluid:Ux",    "status": "converged", "lastResidual": 1e-6, "threshold": 1e-5},
        {"field": "outerFluid:Ux",    "status": "converged", "lastResidual": 1e-6, "threshold": 1e-5},
        {"field": "innerFluid:p_rgh", "status": "converged", "lastResidual": 1e-5, "threshold": 1e-4},
        {"field": "outerFluid:p_rgh", "status": "converged", "lastResidual": 1e-5, "threshold": 1e-4},
        {"field": "wall:h",           "status": "converged", "lastResidual": 1e-7, "threshold": 1e-6},
    ]
    status = _compute_overall_status(fields_all_ok, continuity=None, courant=None)
    assert status == "converged"


def test_plot_field_over_iterations_tool_is_removed():
    """``plot_field_over_iterations`` was removed: its job overlapped
    with the always-on residual chart in LiveTab, so it cost LLM
    tokens for redundant data.  Pin its absence from both the
    TOOL_REGISTRY dispatch dict and the Gemini function-declaration
    schema, so a future refactor can't accidentally restore it.
    """
    from simd_agent.chat.tools import TOOL_REGISTRY, CHAT_TOOLS_SCHEMA

    assert "plot_field_over_iterations" not in TOOL_REGISTRY

    declared_names: list[str] = []
    for tool in CHAT_TOOLS_SCHEMA:
        for fd in getattr(tool, "function_declarations", []) or []:
            declared_names.append(fd.name)
    assert "plot_field_over_iterations" not in declared_names, (
        "plot_field_over_iterations re-added to the Gemini schema — "
        "either intentional (delete this test) or accidental (remove "
        "the FunctionDeclaration block)."
    )


def _make_cht_vtk_index_snap():
    """Synthetic snapshot carrying the multi-region VTK precompute index."""
    from unittest.mock import MagicMock
    snap = MagicMock()
    snap.vtk_result = {}
    snap.sim_progress = []
    snap.vtk_index = {
        "regions": {
            "innerFluid": {
                "fields": [
                    {"name": "U", "num_components": 3, "range": [0.0, 1.8]},
                    {"name": "T", "num_components": 1, "range": [298.0, 305.0]},
                    {"name": "p_rgh", "num_components": 1, "range": [99800.0, 100200.0]},
                ],
                "timesteps": [{"time": 100.0}, {"time": 200.0}, {"time": 579.0}],
            },
            "outerFluid": {
                "fields": [
                    {"name": "U", "num_components": 3, "range": [0.0, 0.9]},
                    {"name": "T", "num_components": 1, "range": [299.0, 310.0]},
                ],
                "timesteps": [{"time": 579.0}],
            },
            "wall": {
                "fields": [
                    {"name": "T", "num_components": 1, "range": [299.5, 308.0]},
                ],
                "timesteps": [{"time": 579.0}],
            },
        },
    }
    return snap


def test_compute_field_stats_drills_into_one_region():
    """``compute_field_stats(field='T', region='wall')`` must return the
    solid wall's T stats only — not the fluid regions' temperature."""
    from simd_agent.chat.tools import compute_field_stats
    out = compute_field_stats(
        {"field": "T", "region": "wall"}, _make_cht_vtk_index_snap(),
    )
    assert out["source"] == "vtk_index_region"
    assert out["region"] == "wall"
    assert out["min"] == 299.5
    assert out["max"] == 308.0
    assert out["sim_time"] == 579.0


def test_compute_field_stats_per_region_breakdown_when_no_filter():
    """When no ``region`` arg is supplied and the case is multi-region,
    the tool must return per-region rows so the LLM can compare
    across regions ("which region has the hottest temperature?").
    """
    from simd_agent.chat.tools import compute_field_stats
    out = compute_field_stats({"field": "T"}, _make_cht_vtk_index_snap())
    assert out["source"] == "vtk_index_regions"
    assert set(out["regions"].keys()) == {"innerFluid", "outerFluid", "wall"}
    # outerFluid is the hottest region (max=310) — surfaceable from per-region rows.
    hottest = max(out["regions"].items(), key=lambda kv: kv[1]["max"])
    assert hottest[0] == "outerFluid"
    assert hottest[1]["max"] == 310.0


def test_compute_field_stats_drops_regions_that_lack_field():
    """``U`` only exists in the two fluid regions — the wall (solid)
    has no velocity field.  Per-region breakdown must list 2 regions,
    not 3."""
    from simd_agent.chat.tools import compute_field_stats
    out = compute_field_stats({"field": "U"}, _make_cht_vtk_index_snap())
    assert set(out["regions"].keys()) == {"innerFluid", "outerFluid"}
    assert "wall" not in out["regions"]


def test_compute_field_stats_unknown_region_returns_no_data():
    """Asking for a region that doesn't exist should not crash — it
    falls through to the legacy sources (residual trend / patch BCs)
    and returns the standard error envelope if nothing matches."""
    from simd_agent.chat.tools import compute_field_stats
    out = compute_field_stats(
        {"field": "T", "region": "doesNotExist"},
        _make_cht_vtk_index_snap(),
    )
    # Falls through to other sources; sim_progress and patches are empty,
    # so the "no data found" envelope is the expected response.
    assert "error" in out or out.get("source") != "vtk_index_region"


def test_single_region_path_unaffected():
    """The guards must not change behaviour for single-region solvers."""
    from simd_agent.run.single_region import (
        build_required_files_list,
        determine_solver,
    )

    cfg = {"physics": {"time_scheme": "steady", "heat_transfer": False}}
    assert determine_solver(cfg) == "simpleFoam"

    # build_required_files_list for a single-region solver should return the
    # plugin-owned manifest (multiple files, not pinned to controlDict).
    required = build_required_files_list("simpleFoam", cfg)
    assert "system/controlDict" in required
    assert len(required) > 1, "single-region manifest should include many files"
