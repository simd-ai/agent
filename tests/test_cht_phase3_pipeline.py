# tests/test_cht_phase3_pipeline.py
"""End-to-end Phase 3 CHT pipeline smoke tests.

Exercises the new pieces that wire chtMultiRegionSimpleFoam through the
orchestrator/packaging/runner-client stack:

  * ``_detect_regions_from_mesh`` populates ``config["regions"]`` from
    mesh patch names so the plugin matches.
  * ``ChtMultiRegionSimpleFoamSolver.required_files()`` returns only the
    LLM-targeted slice (just ``system/controlDict``).
  * ``plugin.validate_full()`` on a minimal LLM output produces the full
    multi-region tree with coupled CHT BCs and per-region thermo.
  * ``SimulationServerClient.submit_run(multi_region=True)`` adds the
    ``multi_region`` form field, and ``submit_run(multi_region=False)``
    leaves the form data identical to the legacy single-region payload
    (no regression for simpleFoam / rhoSimpleFoam / etc.).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from simd_agent.run.multi_region import detect_regions_from_mesh as _detect_regions_from_mesh
from simd_agent.run.orchestration import (
    _force_cht_solver_if_multi_region,
    _precheck_solver,
)
from simd_agent.run.simulation_server_client import (
    SimulationServerClient,
    SimRunMode,
    SimSubmitResponse,
)
from simd_agent.solvers import get_registry


# Mesh patches matching the cyl_cht_2d generator (innerFluid_*, wall_*,
# outerFluid_*, front, back).
_CYL_CHT_PATCHES = [
    {"name": "innerFluid_inlet",     "type": "patch"},
    {"name": "innerFluid_outlet",    "type": "patch"},
    {"name": "innerFluid_symmetry",  "type": "symmetry"},
    {"name": "wall_left_end",        "type": "patch"},
    {"name": "wall_right_end",       "type": "patch"},
    {"name": "outerFluid_inlet",     "type": "patch"},
    {"name": "outerFluid_outlet",    "type": "patch"},
    {"name": "outerFluid_top",       "type": "patch"},
    {"name": "front",                "type": "empty"},
    {"name": "back",                 "type": "empty"},
]


class TestRegionDetection:
    """``_detect_regions_from_mesh`` recognises CHT topology."""

    def test_cyl_cht_geometry_yields_three_regions(self):
        cfg = {"mesh": {"patches": _CYL_CHT_PATCHES}}
        regions = _detect_regions_from_mesh(cfg)
        assert regions is not None
        fluid_names = sorted(r["name"] for r in regions["fluid"])
        solid_names = sorted(r["name"] for r in regions["solid"])
        assert fluid_names == ["innerFluid", "outerFluid"]
        assert solid_names == ["wall"]

    def test_interfaces_are_fluid_solid_crossproduct(self):
        cfg = {"mesh": {"patches": _CYL_CHT_PATCHES}}
        regions = _detect_regions_from_mesh(cfg)
        for fl in regions["fluid"]:
            assert fl["interfaces"] == ["wall"]
        for so in regions["solid"]:
            assert sorted(so["interfaces"]) == ["innerFluid", "outerFluid"]

    def test_single_region_pipe_returns_none(self):
        cfg = {"mesh": {"patches": [
            {"name": "inlet",  "type": "patch"},
            {"name": "outlet", "type": "patch"},
            {"name": "walls",  "type": "wall"},
            {"name": "front",  "type": "empty"},
            {"name": "back",   "type": "empty"},
        ]}}
        assert _detect_regions_from_mesh(cfg) is None

    def test_natural_convection_two_walls_returns_none(self):
        """hotCylinder + coldWalls + front + back: no '_' separator on the
        boundary names, so no region prefixes — must NOT be classified as
        a multi-region case (it's a single-region buoyantBoussinesqSimpleFoam
        problem)."""
        cfg = {"mesh": {"patches": [
            {"name": "hotCylinder", "type": "patch"},
            {"name": "coldWalls",   "type": "patch"},
            {"name": "front",       "type": "empty"},
            {"name": "back",        "type": "empty"},
        ]}}
        assert _detect_regions_from_mesh(cfg) is None

    def test_empty_patches_skipped(self):
        cfg = {"mesh": {"patches": []}}
        assert _detect_regions_from_mesh(cfg) is None

    def test_preset_inferred_from_prefix(self):
        cfg = {"mesh": {"patches": [
            {"name": "ln2Stream_inlet",   "type": "patch"},
            {"name": "ln2Stream_outlet",  "type": "patch"},
            {"name": "steelWall_left",    "type": "patch"},
            {"name": "waterAnnulus_inlet","type": "patch"},
            {"name": "waterAnnulus_outlet","type": "patch"},
        ]}}
        regions = _detect_regions_from_mesh(cfg)
        assert regions is not None
        presets = {r["name"]: r.get("fluid_preset") or r.get("solid_preset")
                   for r in regions["fluid"] + regions["solid"]}
        assert presets["ln2Stream"]    == "ln2"
        assert presets["waterAnnulus"] == "water"
        assert presets["steelWall"]    == "stainless"


class TestForceCHTSolverIfMultiRegion:
    """Multi-region topology forces chtMultiRegion{Simple,}Foam deterministically.

    Live physics edits (time-scheme toggle) flip the variant on the next
    run; single-region cases return None so the LLM selector takes over
    unchanged.
    """

    _cht_regions = {
        "fluid": [{"name": "innerFluid"}, {"name": "outerFluid"}],
        "solid": [{"name": "wall"}],
    }

    def test_steady_returns_simple_variant(self):
        cfg = {"regions": self._cht_regions, "time_scheme": "steady"}
        assert _force_cht_solver_if_multi_region(cfg) == "chtMultiRegionSimpleFoam"

    def test_transient_returns_pimple_variant(self):
        cfg = {"regions": self._cht_regions, "time_scheme": "transient"}
        assert _force_cht_solver_if_multi_region(cfg) == "chtMultiRegionFoam"

    def test_default_is_steady(self):
        cfg = {"regions": self._cht_regions}
        # No time-scheme stated → default to steady → SIMPLE variant.
        assert _force_cht_solver_if_multi_region(cfg) == "chtMultiRegionSimpleFoam"

    def test_nested_physics_time_scheme(self):
        cfg = {
            "regions": self._cht_regions,
            "physics": {"time_scheme": "transient"},
        }
        assert _force_cht_solver_if_multi_region(cfg) == "chtMultiRegionFoam"

    def test_single_region_returns_none(self):
        # No solid → not CHT → LLM selector takes over.
        assert _force_cht_solver_if_multi_region(
            {"regions": {"fluid": [{"name": "water"}], "solid": []}},
        ) is None
        # No fluid → not CHT.
        assert _force_cht_solver_if_multi_region(
            {"regions": {"fluid": [], "solid": [{"name": "wall"}]}},
        ) is None

    def test_no_regions_block_returns_none(self):
        assert _force_cht_solver_if_multi_region({}) is None
        assert _force_cht_solver_if_multi_region({"regions": None}) is None
        # Defensively: non-dict regions value.
        assert _force_cht_solver_if_multi_region({"regions": "oops"}) is None


class TestPrecheckSolverHonored:
    """A5 — the run flow honors a precheck-saved solver before invoking the LLM."""

    def test_known_solver_returned_verbatim(self):
        for name in (
            "simpleFoam", "pimpleFoam", "rhoSimpleFoam", "rhoPimpleFoam",
            "buoyantSimpleFoam", "buoyantBoussinesqSimpleFoam",
            "chtMultiRegionSimpleFoam", "chtMultiRegionFoam",
        ):
            assert _precheck_solver({"solver": {"openfoam_solver": name}}) == name

    def test_camelcase_alias_accepted(self):
        # Frontend writes camelCase; precheck writes snake_case.  Both must work.
        assert (
            _precheck_solver({"solver": {"openfoamSolver": "simpleFoam"}})
            == "simpleFoam"
        )

    def test_unknown_solver_returns_none(self):
        assert _precheck_solver({"solver": {"openfoam_solver": "imaginaryFoam"}}) is None

    def test_missing_field_returns_none(self):
        # Each path falls through to the SolverSelector LLM extractor.
        assert _precheck_solver({}) is None
        assert _precheck_solver({"solver": {}}) is None
        assert _precheck_solver({"solver": None}) is None
        assert _precheck_solver({"solver": {"openfoam_solver": ""}}) is None
        assert _precheck_solver({"solver": {"openfoam_solver": "   "}}) is None
        assert _precheck_solver({"solver": {"openfoam_solver": 123}}) is None


class TestMultiRegionLLMManifest:
    """``required_files`` must return ONLY LLM-targeted files for CHT."""

    def test_required_files_is_controldict_only(self):
        plug = get_registry().get("chtMultiRegionSimpleFoam")
        files = plug.required_files({})
        assert files == ["system/controlDict"]

    def test_all_case_files_includes_per_region_tree(self):
        plug = get_registry().get("chtMultiRegionSimpleFoam")
        cfg = {"mesh": {"patches": _CYL_CHT_PATCHES}}
        cfg["regions"] = _detect_regions_from_mesh(cfg)
        files = set(plug.all_case_files(cfg))
        assert "constant/regionProperties" in files
        assert "constant/innerFluid/thermophysicalProperties" in files
        assert "constant/outerFluid/thermophysicalProperties" in files
        assert "constant/wall/thermophysicalProperties" in files
        assert "0/innerFluid/T" in files
        assert "0/wall/T" in files


class TestValidateFullProducesMultiRegionTree:
    """Plugin.validate_full() on a minimal LLM payload produces the full
    case tree with the right content."""

    def test_validate_full_emits_full_tree_with_coupled_BCs(self):
        plug = get_registry().get("chtMultiRegionSimpleFoam")
        cfg = {"mesh": {"patches": _CYL_CHT_PATCHES}}
        cfg["regions"] = _detect_regions_from_mesh(cfg)

        # Pretend the LLM emitted a controlDict.  Even if it forgot the
        # application line, the plugin's validator should land us on
        # chtMultiRegionSimpleFoam after the fix-up pass.
        llm_files = {
            "system/controlDict": (
                "FoamFile { version 2.0; format ascii; class dictionary; "
                "object controlDict; }\n"
                "application     chtMultiRegionSimpleFoam;\n"
                "startTime 0; endTime 2000; deltaT 1; writeInterval 200;\n"
            ),
        }
        result = plug.validate_full(llm_files, cfg)
        out = result.files

        # All deterministic files materialise
        assert "constant/regionProperties" in out
        assert "constant/innerFluid/thermophysicalProperties" in out
        assert "constant/outerFluid/thermophysicalProperties" in out
        assert "constant/wall/thermophysicalProperties" in out

        # Per-region 0/<region>/T present with coupled CHT BC at interfaces
        for region in ("innerFluid", "outerFluid", "wall"):
            path = f"0/{region}/T"
            assert path in out, f"missing {path}"
            # Each region must carry the coupled BC type at every interface
            assert (
                "compressible::turbulentTemperatureCoupledBaffleMixed" in out[path]
            ), f"{path} missing coupled CHT BC"

        # Fluid regions have U, p, p_rgh, k, epsilon, nut
        for region in ("innerFluid", "outerFluid"):
            for f in ("U", "p", "p_rgh"):
                assert f"0/{region}/{f}" in out, f"missing 0/{region}/{f}"

        # Solid region carries T (the equation actually solved) PLUS
        # passive p / p_rgh / U files — ESI v2406's chtMultiRegionSimpleFoam
        # walks every region's objectRegistry at startup and aborts if
        # those scalars/vector are missing.  Turbulence fields stay
        # fluid-only.
        for required in ("T", "U", "p", "p_rgh"):
            assert f"0/wall/{required}" in out, (
                f"solid wall region must ship 0/wall/{required} for the "
                f"ESI v2406 objectRegistry lookup"
            )
        for fluid_only in ("k", "epsilon", "omega", "nut", "alphat"):
            assert f"0/wall/{fluid_only}" not in out, (
                f"solid wall region must not carry turbulence field "
                f"0/wall/{fluid_only}"
            )

    def test_fluid_regions_get_alphat_when_RAS_turbulence_active(self):
        """Regression: chtMultiRegion* needs 0/<fluid>/alphat for RAS turbulence.

        Without alphat the ESI v2406 solver aborts at startup with
        ``cannot find file 0/<fluid>/alphat`` because every fluid region's
        objectRegistry registers alphat for the energy equation.
        """
        plug = get_registry().get("chtMultiRegionSimpleFoam")
        cfg = {"mesh": {"patches": _CYL_CHT_PATCHES}}
        cfg["regions"] = _detect_regions_from_mesh(cfg)
        # Activate kOmegaSST on every fluid region (region_extractor would
        # do this in the live pipeline; we stamp it directly here).
        for r in cfg["regions"]["fluid"]:
            r["turbulence_model"] = "kOmegaSST"

        out = plug.render_deterministic_files(cfg)

        # Every fluid region must have alphat
        for region in ("innerFluid", "outerFluid"):
            path = f"0/{region}/alphat"
            assert path in out, f"missing {path}"
            # Coupled CHT interface walls must use compressible::alphatWallFunction
            assert "compressible::alphatWallFunction" in out[path], (
                f"{path} must use compressible::alphatWallFunction on walls"
            )

        # Solid region must NOT carry alphat — it has no energy turbulence
        assert "0/wall/alphat" not in out

    def test_fluid_regions_get_omega_for_kOmegaSST(self):
        """Regression: kOmegaSST requires 0/<fluid>/omega per fluid region.

        Without omega the ESI v2406 solver aborts with
        ``cannot find file 0/<fluid>/omega``.  This used to be missed
        because the renderer's hand-coded ``if "komega" in tm`` branch
        only built ``k``/``nut``/``alphat`` and never ``omega``.
        """
        plug = get_registry().get("chtMultiRegionSimpleFoam")
        cfg = {"mesh": {"patches": _CYL_CHT_PATCHES}}
        cfg["regions"] = _detect_regions_from_mesh(cfg)
        for r in cfg["regions"]["fluid"]:
            r["turbulence_model"] = "kOmegaSST"

        out = plug.render_deterministic_files(cfg)
        for region in ("innerFluid", "outerFluid"):
            path = f"0/{region}/omega"
            assert path in out, f"missing {path}"
            assert "omegaWallFunction" in out[path], (
                f"{path} must use omegaWallFunction on walls"
            )
            # Dimensions [0 0 -1 0 0 0 0] — specific dissipation rate (s⁻¹)
            assert "[0 0 -1 0 0 0 0]" in out[path]

    def test_turbulence_models_produce_correct_file_set(self):
        """Audit every turbulence model: ensure each produces the right 0/* files
        and only those files.  Single source of truth lives in
        ``TURBULENCE_FIELDS`` (base.py) + the CHT override (alphat)."""
        plug = get_registry().get("chtMultiRegionSimpleFoam")

        # Expected per-fluid-region files, including base scalars + alphat.
        # Test fixture has two fluid regions (innerFluid, outerFluid) and
        # one solid (wall).  Always-present fluid files: T, U, p, p_rgh.
        cases = {
            "laminar":         set(),
            "kEpsilon":        {"k", "epsilon", "nut", "alphat"},
            "kOmega":          {"k", "omega", "nut", "alphat"},
            "kOmegaSST":       {"k", "omega", "nut", "alphat"},
        }

        for model, expected_turb_fields in cases.items():
            cfg = {"mesh": {"patches": _CYL_CHT_PATCHES}}
            cfg["regions"] = _detect_regions_from_mesh(cfg)
            for r in cfg["regions"]["fluid"]:
                r["turbulence_model"] = model
            out = plug.render_deterministic_files(cfg)

            for region in ("innerFluid", "outerFluid"):
                # Base scalar/vector fields always present
                for base in ("T", "U", "p", "p_rgh"):
                    assert f"0/{region}/{base}" in out, (
                        f"{model}: missing base field 0/{region}/{base}"
                    )
                # Expected turbulence fields are present
                for field in expected_turb_fields:
                    assert f"0/{region}/{field}" in out, (
                        f"{model}: missing 0/{region}/{field}"
                    )
                # Non-expected turbulence fields are absent (no spurious files)
                _all_turb = {"k", "epsilon", "omega", "nut", "alphat"}
                for absent in _all_turb - expected_turb_fields:
                    assert f"0/{region}/{absent}" not in out, (
                        f"{model}: unexpected 0/{region}/{absent} was rendered"
                    )

            # Solid never gets turbulence fields regardless of model
            for absent in ("k", "epsilon", "omega", "nut", "alphat"):
                assert f"0/wall/{absent}" not in out, (
                    f"{model}: solid wall must not carry 0/wall/{absent}"
                )

        # regionProperties lists exactly the three regions
        rp = out["constant/regionProperties"]
        assert "innerFluid" in rp
        assert "outerFluid" in rp
        assert "wall" in rp

    def test_solid_thermo_uses_solid_preset_props(self):
        plug = get_registry().get("chtMultiRegionSimpleFoam")
        cfg = {"mesh": {"patches": _CYL_CHT_PATCHES}}
        cfg["regions"] = _detect_regions_from_mesh(cfg)
        out = plug.render_deterministic_files(cfg)
        wall_thermo = out["constant/wall/thermophysicalProperties"]
        # stainless preset is the default for "wall_" prefix → ρ=7900, κ=16.2
        assert "heSolidThermo" in wall_thermo
        assert "kappa" in wall_thermo
        assert "7900" in wall_thermo
        assert "16.2" in wall_thermo  # stainless steel conductivity


class TestSubmitRunFlag:
    """The multi_region flag flows correctly through SimulationServerClient."""

    @staticmethod
    def _make_stub_client(base_url: str = "http://x") -> tuple[SimulationServerClient, AsyncMock]:
        """Build a SimulationServerClient with a stubbed httpx client.

        ``_get_client`` checks ``self._client.is_closed`` — a vanilla
        ``MagicMock`` returns a truthy MagicMock for unknown attributes,
        causing the client to be recreated and our stub bypassed.  We
        explicitly set ``is_closed = False`` so the existing client is reused.
        """
        client = SimulationServerClient(base_url=base_url)
        stub = AsyncMock()
        stub.is_closed = False
        stub.post = AsyncMock(return_value=MagicMock(
            raise_for_status=lambda: None,
            json=lambda: {
                "run_id": "r1",
                "status": "queued",
                "mode": "test",
                "events_url": "/api/run/r1/events",
                "status_url": "/api/run/r1/status",
            },
        ))
        client._client = stub  # noqa: SLF001 — test injects the http stub
        return client, stub

    @pytest.mark.asyncio
    async def test_multi_region_true_adds_form_field(self):
        client, stub = self._make_stub_client()
        await client.submit_run(
            case_zip=b"PK\x03\x04",
            mode=SimRunMode.TEST,
            run_id="r1",
            multi_region=True,
        )
        call = stub.post.call_args
        data = call.kwargs.get("data") or (call.args[1] if len(call.args) > 1 else {})
        assert data.get("multi_region") == "true", (
            f"multi_region flag missing from form data: {data}"
        )

    @pytest.mark.asyncio
    async def test_multi_region_false_omits_form_field(self):
        """Single-region cases (default) must NOT send the multi_region flag.

        This guarantees existing simpleFoam / rhoSimpleFoam / buoyant* /
        buoyantBoussinesq* submissions are byte-identical to before — no
        runner-side behavioural change for them.
        """
        client, stub = self._make_stub_client()
        await client.submit_run(
            case_zip=b"PK\x03\x04",
            mode=SimRunMode.TEST,
            run_id="r1",
            # multi_region omitted — defaults to False
        )
        call = stub.post.call_args
        data = call.kwargs.get("data") or (call.args[1] if len(call.args) > 1 else {})
        assert "multi_region" not in data, (
            f"single-region payload must not contain multi_region: {data}"
        )
