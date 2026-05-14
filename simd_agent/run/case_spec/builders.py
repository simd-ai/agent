"""Build a CaseSpec from a validated SimulationConfig.

``build_case_spec`` is the ONLY function that reads ``validated_config``.
All downstream generators read from the returned ``CaseSpec`` only.  This
is the seam between "raw user input" and "resolved decisions"; nothing
upstream of this needs to know about CaseSpec, and nothing downstream
needs to know about validated_config.
"""

from __future__ import annotations

import logging
from typing import Any

from .density import _density_bounds_for_profile, _estimate_inlet_mach
from .mesh_quality import _mesh_quality_decisions, _props_from_registry
from .resolvers import resolve_turbulence_spec
from .spec import _SOLVER_PROPS, _TURB_FIELDS, CaseSpec
from .thermo_profile import _select_thermo_profile

logger = logging.getLogger(__name__)


def _safe_float(v: Any) -> float | None:
    """Convert value to float, returning None on failure."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_case_spec(solver: str, validated_config: dict[str, Any]) -> CaseSpec:
    """Compute CaseSpec deterministically — no LLM, no side effects."""
    props = (
        _props_from_registry(solver)
        or _SOLVER_PROPS.get(solver)
        or _SOLVER_PROPS["simpleFoam"]
    )
    phys = validated_config.get("physics", {}) or {}

    # ── N0: Mesh patch map ───────────────────────────────────────────────────
    mesh_raw = validated_config.get("mesh", {})
    mesh = mesh_raw if isinstance(mesh_raw, dict) else {}
    mesh_patches_raw = mesh.get("patches", [])

    patch_names: list[str] = []
    patch_type_by_name: dict[str, str] = {}
    is_2d = False

    for mp in mesh_patches_raw:
        if isinstance(mp, dict):
            name = mp.get("name", "")
            ptype = mp.get("type", "patch")
        elif hasattr(mp, "name"):
            name = mp.name
            ptype = getattr(mp, "type", "patch")
        else:
            continue
        if not name:
            continue
        # Force frontAndBack → empty (2D constraint)
        nl = name.lower().replace("_", "")
        if nl in ("frontandback", "frontback"):
            ptype = "empty"
            is_2d = True
        if ptype in ("empty", "wedge"):
            is_2d = True
        patch_names.append(name)
        patch_type_by_name[name] = ptype

    # If no mesh patches in config, derive from boundary_conditions keys
    if not patch_names:
        bcs = validated_config.get("boundary_conditions", {}) or {}
        for k in bcs:
            nl = k.lower().replace("_", "")
            if nl in ("frontandback", "frontback"):
                ptype = "empty"
                is_2d = True
            elif nl in ("front", "back") and isinstance(bcs[k], dict) and bcs[k].get("patch_type") in ("wedge", "empty"):
                ptype = str(bcs[k]["patch_type"])
                is_2d = True
            elif nl == "wall":
                ptype = "wall"
            else:
                # Check if BC itself declares a 2D type
                bc_entry = bcs[k]
                if isinstance(bc_entry, dict) and bc_entry.get("patch_type") in ("empty", "wedge"):
                    ptype = str(bc_entry["patch_type"])
                    is_2d = True
                else:
                    ptype = "patch"
            patch_names.append(k)
            patch_type_by_name[k] = ptype

    # ── N1: Solver spec ──────────────────────────────────────────────────────
    algorithm = props["algorithm"]
    pressure_field = props["pressure_field"]
    transient = props["transient"]
    compressible = props["compressible"]
    needs_gravity = props["needs_gravity"]

    # ── N2: Phase spec ───────────────────────────────────────────────────────
    multiphase = props["multiphase"]
    phase_names_cfg = (
        validated_config.get("phases")
        or phys.get("phases")
        or []
    )
    if multiphase and not phase_names_cfg:
        # Derive phase names from fluid name — avoid hardcoding "water/air" for cryogenic cases
        _fluid_name_lower = (
            validated_config.get("fluid", {}) or phys.get("fluid", {}) or {}
        )
        _fluid_name_lower = (_fluid_name_lower.get("name") or "").lower()
        if "helium" in _fluid_name_lower or "lhe" in _fluid_name_lower:
            phase_names_cfg = ["liquidHelium", "heliumVapour"]
        elif "nitrogen" in _fluid_name_lower or "ln2" in _fluid_name_lower:
            phase_names_cfg = ["liquidNitrogen", "nitrogenVapour"]
        elif "hydrogen" in _fluid_name_lower or "lh2" in _fluid_name_lower:
            phase_names_cfg = ["liquidHydrogen", "hydrogenVapour"]
        elif "oxygen" in _fluid_name_lower or "lox" in _fluid_name_lower:
            phase_names_cfg = ["liquidOxygen", "oxygenVapour"]
        elif "methane" in _fluid_name_lower or "lng" in _fluid_name_lower:
            phase_names_cfg = ["liquidMethane", "methaneVapour"]
        elif "water" in _fluid_name_lower or "h2o" in _fluid_name_lower:
            phase_names_cfg = ["water", "air"]
        else:
            phase_names_cfg = ["liquid", "gas"]
    alpha_fields: list[str] = []
    if multiphase and phase_names_cfg:
        # VOF: only the PRIMARY alpha field for 2-phase
        alpha_fields = [f"alpha.{phase_names_cfg[0]}"]

    # ── N3: Thermo spec ──────────────────────────────────────────────────────
    energy = props["energy"]
    energy_field: str | None = None
    if energy == "he":
        # Default: use sensibleEnthalpy → "h" for rho* solvers.
        # energy_field is used in fvSchemes/fvSolution only.
        # We do NOT generate a 0/h or 0/e file — the thermo package initialises
        # the energy field from 0/T at startup (standard OpenFOAM tutorial pattern).
        energy_field = "h"
        if validated_config.get("energy_formulation", "") == "sensibleInternalEnergy":
            energy_field = "e"

    # ── N4: Turbulence spec ──────────────────────────────────────────────────
    # Resolved solver-aware: reads the model from every known config shape
    # (precheck nests under ``turbulence.model``; canonical is
    # ``physics.turbulence_model``; legacy puts it at the top).  Falls back
    # to the plugin's ``default_turbulence_model`` rather than silently
    # demoting to laminar — that drift was the cause of the SIGFPE cascade
    # on moderate-Re forced-convection rhoSimpleFoam cases.
    try:
        from simd_agent.solvers import get_registry as _get_reg_for_turb
        _plugin_for_turb = _get_reg_for_turb().get(solver)
    except Exception:
        _plugin_for_turb = None

    turbulence_spec = resolve_turbulence_spec(_plugin_for_turb, validated_config)
    turb_model = turbulence_spec.model
    flow_regime = turbulence_spec.flow_regime
    sim_type = turbulence_spec.simulation_type
    turb_fields = _TURB_FIELDS.get(turb_model, _TURB_FIELDS["kOmegaSST"])

    # If energy solver + turbulence: add alphat
    if compressible and turb_fields:
        if "alphat" not in turb_fields:
            turb_fields = list(turb_fields) + ["alphat"]

    # ── N4b: Turbulence initial values (from frontend's turbulence block) ────
    _turb_cfg = validated_config.get("turbulence", {}) or {}
    turbulence_initial_values: dict[str, float] = {}
    for _tf, _aliases in [
        ("k",       ["k"]),
        ("omega",   ["omega"]),
        ("epsilon", ["epsilon"]),
        ("nut",     ["nut"]),
    ]:
        for _alias in _aliases:
            _v = _safe_float(_turb_cfg.get(_alias))
            if _v is not None:
                turbulence_initial_values[_tf] = _v
                break

    # ── N5: Field set ────────────────────────────────────────────────────────
    required_0 = ["U", pressure_field]
    # Buoyant solvers read BOTH p_rgh (solved) AND p (absolute, for post-processing).
    if pressure_field == "p_rgh" and compressible:
        required_0.append("p")
    if energy == "he":
        # 0/T only — the thermo package reads T and initialises h/e internally.
        # Providing 0/h causes "Negative initial temperature" because the
        # back-conversion is reference-sensitive.
        required_0.append("T")
    required_0 += alpha_fields
    required_0 += [f for f in turb_fields if f not in required_0]

    required_0_fields = [f"0/{f}" for f in required_0]

    # ── N6: Constant files ───────────────────────────────────────────────────
    required_const: list[str] = []
    if compressible:
        required_const.append("constant/thermophysicalProperties")
    else:
        required_const.append("constant/transportProperties")
    required_const.append("constant/turbulenceProperties")
    if needs_gravity:
        required_const.append("constant/g")

    # ── Time control ─────────────────────────────────────────────────────────
    solver_cfg = validated_config.get("solver", {})
    if not isinstance(solver_cfg, dict):
        solver_cfg = {}

    logger.info(f"[CASE_SPEC] Time control — transient={transient}, solver_cfg={solver_cfg}")
    logger.info(
        f"[CASE_SPEC] top-level max_iterations={validated_config.get('max_iterations')}, "
        f"end_time={validated_config.get('end_time')}"
    )

    if not transient:
        # Steady solvers: endTime = iteration count.
        end_time = float(
            validated_config.get("max_iterations")
            or solver_cfg.get("max_iterations")
            or solver_cfg.get("maxIterations")
            or solver_cfg.get("endTime")
            or solver_cfg.get("end_time")
            or 1000
        )
        delta_t = 1.0
    else:
        # Transient solvers: endTime = physical seconds.
        _raw_end_time = (
            validated_config.get("end_time")
            or solver_cfg.get("end_time")
        )
        _raw_end_time_camel = solver_cfg.get("endTime")
        _raw_max_iter = (
            solver_cfg.get("max_iterations")
            or solver_cfg.get("maxIterations")
            or validated_config.get("max_iterations")
        )
        if _raw_end_time:
            end_time = float(_raw_end_time)
        elif _raw_end_time_camel and _raw_max_iter:
            _et = float(_raw_end_time_camel)
            _mi = float(_raw_max_iter)
            end_time = _mi if _et <= 10 and _mi > _et else _et
        elif _raw_end_time_camel:
            end_time = float(_raw_end_time_camel)
        elif _raw_max_iter:
            end_time = float(_raw_max_iter)
        else:
            end_time = 10.0
        delta_t = float(
            validated_config.get("delta_t")
            or solver_cfg.get("deltaT")
            or solver_cfg.get("delta_t")
            or 0.001
        )

    logger.info(f"[CASE_SPEC] RESOLVED end_time={end_time}, delta_t={delta_t}")

    # ── Transient time control (maxCo, writeInterval, func writeInterval) ────
    if transient:
        max_co = 2.0 if algorithm == "PIMPLE" else 0.5
        _target_snapshots = max(30, min(100, int(end_time * 10)))
        write_interval = end_time / _target_snapshots
        func_write_interval = write_interval * 2
        max_delta_t = write_interval
        logger.info(
            f"[CASE_SPEC] Transient control: maxCo={max_co}, "
            f"writeInterval={write_interval:.6g} ({_target_snapshots} snapshots), "
            f"funcWriteInterval={func_write_interval:.6g}, maxDeltaT={max_delta_t:.6g}"
        )
    else:
        max_co = 0.5
        write_interval = 1.0
        func_write_interval = 1.0
        max_delta_t = 1.0

    # ── Physical properties ───────────────────────────────────────────────────
    fluid = (
        validated_config.get("fluid", {})
        or validated_config.get("material", {})
        or phys.get("fluid", {})
        or {}
    )
    bcs = validated_config.get("boundary_conditions", {}) or {}
    inlet_bc = bcs.get("inlet", {}) or {}

    def _bc_value(field_entry: Any) -> Any:
        """Unwrap a BC entry that may be a flat value OR {"type": ..., "value": ...} dict."""
        if isinstance(field_entry, dict):
            return field_entry.get("value")
        return field_entry

    def _vel(bc: dict) -> list[float]:
        raw = bc.get("velocity") or bc.get("U")

        if isinstance(raw, dict) and raw.get("type") == "flowRateInletVelocity":
            mv = raw.get("meanVelocity")
            if isinstance(mv, (int, float)):
                return [float(mv), 0.0, 0.0]
            return [0.0, 0.0, 0.0]

        v = _bc_value(raw)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            return [float(v[0]), float(v[1]), float(v[2])]
        if isinstance(v, (int, float)):
            return [float(v), 0.0, 0.0]
        mag = bc.get("velocity_magnitude") or bc.get("speed") or bc.get("U_mag")
        if mag is not None:
            return [float(_bc_value(mag) or mag), 0.0, 0.0]
        return [0.0, 0.0, 0.0]

    inlet_vel = _vel(inlet_bc)
    inlet_T = _safe_float(
        _bc_value(inlet_bc.get("temperature") or inlet_bc.get("T"))
        or validated_config.get("inlet_temperature")
    )
    wall_bc = bcs.get("wall", {}) or {}
    wall_T = _safe_float(
        _bc_value(wall_bc.get("temperature") or wall_bc.get("T"))
        or validated_config.get("wall_temperature")
    )
    # Operating pressure: prefer explicit config value; fall back to outlet fixedValue, then inlet fixedValue
    _inlet_p_raw = inlet_bc.get("pressure")
    _inlet_p_val = _bc_value(_inlet_p_raw) if isinstance(_inlet_p_raw, dict) else None
    _inlet_p = _inlet_p_val if (_inlet_p_val is not None and isinstance(_inlet_p_val, (int, float)) and _inlet_p_val > 1000) else None
    outlet_bc = bcs.get("outlet", {}) or {}
    _outlet_p_raw = outlet_bc.get("pressure")
    _outlet_p_val = _bc_value(_outlet_p_raw) if isinstance(_outlet_p_raw, dict) else None
    _outlet_p = _outlet_p_val if (_outlet_p_val is not None and isinstance(_outlet_p_val, (int, float)) and _outlet_p_val > 1000) else None
    op_p = _safe_float(
        validated_config.get("operating_pressure")
        or validated_config.get("pressure")
        or _outlet_p
        or _inlet_p
    ) or 101325.0

    fluid_name = (fluid.get("name") or "").strip()
    nu = _safe_float(fluid.get("nu") or fluid.get("kinematic_viscosity"))
    rho = _safe_float(fluid.get("rho") or fluid.get("density"))
    mu = _safe_float(fluid.get("mu") or fluid.get("dynamic_viscosity"))
    cp = _safe_float(fluid.get("Cp") or fluid.get("cp") or fluid.get("specific_heat"))
    pr = _safe_float(fluid.get("Pr") or fluid.get("prandtl"))

    if nu is None and mu is not None and rho is not None and rho > 0:
        nu = mu / rho

    def _normalize_inlet_u_bc(ibc: dict, _rho: float | None) -> None:
        """Ensure flowRateInletVelocity contains required keys + safe placeholder value."""
        u = ibc.get("velocity") or ibc.get("U")
        if not isinstance(u, dict):
            return
        if u.get("type") != "flowRateInletVelocity":
            return
        for key in ("massFlowRate", "volumetricFlowRate", "meanVelocity"):
            if key in ibc and key not in u:
                u[key] = ibc[key]
        if "massFlowRate" in u:
            if not compressible:
                # Incompressible solvers have NO rho field — convert to volumetricFlowRate.
                if _rho is not None and _rho > 0:
                    try:
                        _mdot = float(u["massFlowRate"])
                        _q = _mdot / _rho
                        del u["massFlowRate"]
                        u["volumetricFlowRate"] = _q
                    except (TypeError, ValueError):
                        u.setdefault("rhoInlet", float(_rho))
            else:
                # Compressible: rhoInlet is the startup fallback before the rho field exists.
                if _rho is not None:
                    u.setdefault("rhoInlet", float(_rho))
        u["value"] = [0.0, 0.0, 0.0]

    _normalize_inlet_u_bc(inlet_bc, rho)

    # ── fvOptions temperature limits ─────────────────────────────────────────
    fv_options_t_min: float | None = None
    fv_options_eos_t_ceiling: float | None = None
    fv_options_bc_temps: list[float] = []
    _required_system = ["system/controlDict", "system/fvSchemes", "system/fvSolution"]

    # Respect plugin's required_files() — drop any system file the plugin
    # excludes (e.g. simpleFoam generates fvSolution deterministically and
    # removes it from required_files), otherwise the parallel codegen will
    # waste an LLM call on a file the validator will throw away.
    try:
        from simd_agent.solvers import get_registry as _get_reg
        _plugin = _get_reg().get(solver)
    except Exception:
        _plugin = None
    if _plugin is not None:
        _plugin_files = set(_plugin.required_files(validated_config))
        _required_system = [f for f in _required_system if f in _plugin_files]
        # Phase 4: also filter constant + 0/ files by plugin's required_files()
        # so deterministic files (e.g. constant/turbulenceProperties) aren't
        # sent to the LLM only to be discarded by validate().
        required_const = [f for f in required_const if f in _plugin_files]
        required_0_fields = [f for f in required_0_fields if f in _plugin_files]

    if energy == "he":
        # Collect ALL BC temperatures — every patch, every temperature key variant
        _all_bc_t: list[float] = []
        for _pbc in bcs.values():
            if not isinstance(_pbc, dict):
                continue
            for _tk in ("temperature", "T", "T_inlet", "T_wall", "inlet_temperature", "wall_temperature"):
                _tv_raw = _pbc.get(_tk)
                if isinstance(_tv_raw, dict):
                    _tv = _tv_raw.get("value") or _tv_raw.get("uniform")
                elif isinstance(_tv_raw, (int, float)):
                    _tv = float(_tv_raw)
                else:
                    _tv = None
                if _tv is not None:
                    try:
                        _all_bc_t.append(float(_tv))
                    except (TypeError, ValueError):
                        pass
        for _top_key in ("inlet_temperature", "wall_temperature", "temperature"):
            _top_v = _safe_float(validated_config.get(_top_key))
            if _top_v is not None:
                _all_bc_t.append(_top_v)
        if inlet_T is not None and inlet_T not in _all_bc_t:
            _all_bc_t.append(inlet_T)
        if wall_T is not None and wall_T not in _all_bc_t:
            _all_bc_t.append(wall_T)
        fv_options_bc_temps = sorted(set(_all_bc_t))

        _min_t = min(_all_bc_t) if _all_bc_t else (inlet_T or 200.0)
        fv_options_t_min = max(1.0, _min_t * 0.5)

        # Hard EOS ceiling for icoPolynomial ρ(T) = a0 + a1·T: ρ→0 at T = a0/|a1|.
        # The LLM must choose max < eos_t_ceiling or density goes negative → SIGFPE.
        # Threshold 30 kg/m³ catches all cryogenic liquids including LHe (~125 kg/m³)
        # and LH2 (~71 kg/m³) which are below the old 200 kg/m³ cutoff.
        if rho is not None and rho > 30.0:
            _T_ref = inlet_T if inlet_T is not None else 300.0
            if _T_ref < 8.0:
                _a1 = -5.0   # liquid helium (He-4): ~125 kg/m³ at 4.2 K
            elif _T_ref < 35.0:
                _a1 = -0.7   # liquid hydrogen: ~71 kg/m³ at 20 K
            elif _T_ref < 100.0:
                _a1 = -4.7   # LN2 / LOX / LAr: ~800–1140 kg/m³
            else:
                _a1 = -0.5   # other dense liquids (water, oil, etc.)
            _a0 = rho - _a1 * _T_ref
            fv_options_eos_t_ceiling = abs(_a0) / abs(_a1)

        # fvOptions limitTemperature is needed only when a wall is genuinely
        # hotter than the inlet fluid.  Adiabatic walls don't need it;
        # isothermal walls (wall_T ≈ inlet_T) don't need it.
        _wall_temps_found: list[float] = []
        if wall_T is not None:
            _wall_temps_found.append(wall_T)
        for _pname in patch_names:
            if patch_type_by_name.get(_pname) != "wall":
                continue
            _pbc = bcs.get(_pname, {}) or {}
            _t_raw = _pbc.get("temperature") or _pbc.get("T")
            _t_val = _bc_value(_t_raw) if isinstance(_t_raw, dict) else _t_raw
            if _t_val is not None:
                try:
                    _wall_temps_found.append(float(_t_val))
                except (TypeError, ValueError):
                    pass

        _wall_heats_fluid = False
        if _wall_temps_found:
            _max_wall_t = max(_wall_temps_found)
            if inlet_T is not None:
                _wall_heats_fluid = _max_wall_t > inlet_T + 10.0
            else:
                _wall_heats_fluid = _max_wall_t > 200.0

        if _wall_heats_fluid:
            _required_system.append("system/fvOptions")

    # ── N3-energy: No 0/h or 0/e file is generated ───────────────────────────
    # The thermo package initialises the energy field from 0/T at startup.
    # energy_field is kept for fvSchemes/fvSolution naming only.
    energy_bc_values: dict[str, float | None] = {}

    # ── VOF initial domain state (multiphase only) ────────────────────────────
    initial_phase_layout = "uniform_gas"
    initial_domain_pressure: float | None = None
    initial_domain_temperature: float | None = None
    if multiphase:
        _cfg_layout = (
            validated_config.get("initial_phase_layout")
            or phys.get("initial_phase_layout")
        )
        if _cfg_layout in (
            "uniform_gas", "uniform_liquid",
            "liquid_region_in_gas", "gas_region_in_liquid",
        ):
            initial_phase_layout = _cfg_layout
        else:
            # Default: gas-filled domain — liquid enters from inlet.
            initial_phase_layout = "uniform_gas"

        _cfg_init_p = _safe_float(
            validated_config.get("initial_domain_pressure")
            or phys.get("initial_domain_pressure")
        )
        initial_domain_pressure = _cfg_init_p if _cfg_init_p is not None else op_p

        _cfg_init_t = _safe_float(
            validated_config.get("initial_domain_temperature")
            or phys.get("initial_domain_temperature")
        )
        initial_domain_temperature = _cfg_init_t if _cfg_init_t is not None else inlet_T

        _needs_set_fields = initial_phase_layout in (
            "liquid_region_in_gas", "gas_region_in_liquid"
        )
        if _needs_set_fields:
            if "system/setFieldsDict" not in _required_system:
                _required_system.append("system/setFieldsDict")
            required_0_fields = [
                f"{fp}.orig" if fp.startswith("0/alpha.") else fp
                for fp in required_0_fields
            ]

    # ── Mesh quality-driven numerics ────────────────────────────────────────
    _check_mesh_raw = mesh.get("check_mesh") or mesh.get("checkMesh")
    _mq = _mesh_quality_decisions(_check_mesh_raw)

    # ── Thermo profile (gas vs cryogenic) ────────────────────────────────────
    _thermo_profile = _select_thermo_profile(
        fluid_name=fluid_name,
        inlet_temperature=inlet_T,
        rho=rho,
        has_heat_transfer=(energy == "he"),
    )
    if compressible:
        _rho_min, _rho_max = _density_bounds_for_profile(
            _thermo_profile, rho, fv_options_eos_t_ceiling, fv_options_bc_temps
        )
        _mach = _estimate_inlet_mach(_thermo_profile, inlet_vel, inlet_T)
        _transonic = _mach > 0.5
        _p_min = max(1e3, op_p * 0.05)
        _p_max = max(op_p * 20.0, 5e5)
    else:
        _rho_min = _rho_max = _p_min = _p_max = None
        _mach = 0.0
        _transonic = False
    logger.info(
        f"[THERMO_PROFILE] {solver}: profile='{_thermo_profile}'  "
        f"fluid='{fluid_name or '?'}'  inlet_T={inlet_T}  "
        f"rho={rho}  heat={energy == 'he'}  Mach≈{_mach:.2f}  transonic={_transonic}"
    )

    spec = CaseSpec(
        # N0
        patch_names=patch_names,
        patch_type_by_name=patch_type_by_name,
        is_2d=is_2d,
        # N1
        solver=solver,
        algorithm=algorithm,
        pressure_field=pressure_field,
        transient=transient,
        compressible=compressible,
        needs_gravity=needs_gravity,
        # N2
        multiphase=multiphase,
        phase_names=phase_names_cfg,
        alpha_fields=alpha_fields,
        # N3
        energy=energy,
        energy_field=energy_field,
        # N4
        turbulence_model=turb_model,
        sim_type=sim_type,
        turbulence_fields=turb_fields,
        # N5
        required_0_fields=required_0_fields,
        required_constant_files=required_const,
        required_system_files=_required_system,
        # Time
        end_time=end_time,
        delta_t=delta_t,
        max_co=max_co,
        write_interval=write_interval,
        func_write_interval=func_write_interval,
        max_delta_t=max_delta_t,
        # Physical
        fluid_name=fluid_name,
        inlet_velocity=inlet_vel,
        inlet_temperature=inlet_T,
        wall_temperature=wall_T,
        operating_pressure=op_p,
        nu=nu,
        rho=rho,
        mu=mu,
        cp=cp,
        prandtl=pr,
        energy_bc_values=energy_bc_values,
        turbulence_initial_values=turbulence_initial_values,
        boundary_conditions=dict(bcs),
        fv_options_t_min=fv_options_t_min,
        fv_options_eos_t_ceiling=fv_options_eos_t_ceiling,
        fv_options_bc_temps=fv_options_bc_temps,
        initial_phase_layout=initial_phase_layout,
        initial_domain_pressure=initial_domain_pressure,
        initial_domain_temperature=initial_domain_temperature,
        mesh_max_non_orthogonality=_mq["mesh_max_non_orthogonality"],
        mesh_max_skewness=_mq["mesh_max_skewness"],
        mesh_max_aspect_ratio=_mq["mesh_max_aspect_ratio"],
        use_simplec=_mq["use_simplec"],
        n_non_ortho_correctors=_mq["n_non_ortho_correctors"],
        mesh_quality_tier=_mq["mesh_quality_tier"],
        thermo_profile=_thermo_profile,
        rho_min=_rho_min,
        rho_max=_rho_max,
        p_min=_p_min,
        p_max=_p_max,
        transonic=_transonic,
        mach_estimate=_mach,
        # Phase 1 typed strategies — not populated yet (Phase 2 wires them in).
        thermo_strategy=None,
        pressure_solver_strategy=None,
        compressible_bounds_strategy=None,
        inlet_turbulence_strategy=[],
        # Resolved turbulence — solver-aware, single source of truth.
        turbulence_spec=turbulence_spec,
    )

    logger.info(
        f"[CASE_SPEC] solver={solver}  algorithm={algorithm}  "
        f"pressure={pressure_field}  energy={energy_field}  "
        f"turbulence={turb_model}  patches={patch_names}  "
        f"0/fields={[f.split('/', 1)[1] for f in required_0_fields]}  "
        f"const={required_const}  "
        f"init_layout={initial_phase_layout}  "
        f"init_p={initial_domain_pressure}  init_T={initial_domain_temperature}  "
        f"mesh_tier={_mq['mesh_quality_tier']}  "
        f"use_simplec={_mq['use_simplec']}  "
        f"n_non_ortho={_mq['n_non_ortho_correctors']}"
    )
    return spec
