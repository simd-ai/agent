# simd_agent/normalizer.py
"""
Configuration normalizer for backward-compatible parsing.

Converts various input formats (legacy, camelCase, partial configs) to the
canonical SimulationConfigV1 schema.
"""

import logging
import re
from typing import Any

from simd_agent.models import (
    BoundaryConditionV1,
    BoundaryType,
    CheckMeshInfoV1,
    ConfigValidationResult,
    FlowRegime,
    FluidV1,
    GeometryV1,
    MeshInfoV1,
    MeshPatchV1,
    MissingFieldInfo,
    Operation,
    PhysicsV1,
    PressureBCV1,
    SimulationConfigV1,
    SolverV1,
    TemperatureBCV1,
    TimeScheme,
    VelocityBCV1,
)

logger = logging.getLogger(__name__)


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    # Handle consecutive capitals (e.g., 'XMLParser' -> 'xml_parser')
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def deep_convert_keys(obj: Any, converter: callable) -> Any:
    """Recursively convert dictionary keys using the given converter function."""
    if isinstance(obj, dict):
        return {converter(k): deep_convert_keys(v, converter) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_convert_keys(item, converter) for item in obj]
    return obj


def normalize_config(
    raw_config: dict[str, Any],
    user_requirements: str = "",
) -> SimulationConfigV1:
    """
    Normalize a raw simulation config to the canonical V1 schema.
    
    Handles:
    - camelCase -> snake_case conversion
    - Legacy formats (mesh as string, mesh: {mesh_id}, etc.)
    - Partial configs with defaults
    - Boundary condition normalization
    
    Args:
        raw_config: Raw config from frontend (any format)
        user_requirements: Optional user requirements for context
        
    Returns:
        Normalized SimulationConfigV1
    """
    if not raw_config:
        return SimulationConfigV1()
    
    # First, convert all keys to snake_case
    config = deep_convert_keys(raw_config, camel_to_snake)
    
    # IMPORTANT: Restore original boundary_conditions keys.
    # deep_convert_keys converts "frontAndBack" → "front_and_back", but these
    # are OpenFOAM patch names (identifiers), not API field names.
    raw_bcs = raw_config.get("boundary_conditions") or raw_config.get("boundaryConditions") or {}
    if isinstance(raw_bcs, dict) and raw_bcs:
        converted_bcs = config.get("boundary_conditions", {})
        if isinstance(converted_bcs, dict):
            restored_bcs = {}
            orig_keys = list(raw_bcs.keys())
            conv_keys = list(converted_bcs.keys())
            for orig_key, conv_key in zip(orig_keys, conv_keys):
                restored_bcs[orig_key] = converted_bcs[conv_key]
            config["boundary_conditions"] = restored_bcs
    
    logger.debug(f"Normalizing config with keys: {list(config.keys())}")
    
    # Normalize mesh
    mesh = _normalize_mesh(config)
    
    # Normalize physics
    physics = _normalize_physics(config)
    
    # Normalize solver
    solver = _normalize_solver(config)
    
    # Normalize fluid
    fluid = _normalize_fluid(config)
    
    # Normalize geometry
    geometry = _normalize_geometry(config)
    
    # Normalize boundary conditions
    boundary_conditions = _normalize_boundary_conditions(config, mesh)
    
    # Normalize KPI targets
    kpi_targets = _normalize_kpi_targets(config)
    
    # Extract legacy fields
    inlet = config.get("inlet")
    outlet = config.get("outlet")
    case_type = config.get("case_type")
    
    return SimulationConfigV1(
        mesh=mesh,
        physics=physics,
        solver=solver,
        fluid=fluid,
        geometry=geometry,
        boundary_conditions=boundary_conditions,
        kpi_targets=kpi_targets,
        case_type=case_type,
        inlet=inlet,
        outlet=outlet,
    )


def _normalize_mesh(config: dict[str, Any]) -> MeshInfoV1 | None:
    """Normalize mesh configuration."""
    mesh_data = config.get("mesh")
    
    if mesh_data is None:
        return None
    
    # Handle string mesh_id (legacy)
    if isinstance(mesh_data, str):
        return MeshInfoV1(mesh_id=mesh_data)
    
    if not isinstance(mesh_data, dict):
        logger.warning(f"Invalid mesh format: {type(mesh_data)}")
        return None
    
    # Extract mesh_id from various locations
    mesh_id = (
        mesh_data.get("mesh_id") or
        mesh_data.get("meshId") or
        mesh_data.get("id") or
        ""
    )
    
    if not mesh_id:
        logger.warning("No mesh_id found in mesh config")
        return None
    
    # Normalize patches
    patches = []
    raw_patches = mesh_data.get("patches", [])
    
    for p in raw_patches:
        if isinstance(p, dict):
            p_name = p.get("name", "unknown")
            # Accept "patch_type", "patchType", or "type" for the patch type
            p_type = (
                p.get("patch_type") or p.get("patchType") or
                p.get("type", "patch")
            )
            
            # FORCE constraint patch types based on well-known names
            # frontAndBack is ALWAYS empty in 2D OpenFOAM simulations
            name_lower = p_name.lower().replace("_", "")
            if name_lower in ("frontandback", "frontback", "defaultfaces"):
                p_type = "empty"
            
            patches.append(MeshPatchV1(
                name=p_name,
                type=p_type,
                n_faces=p.get("n_faces") or p.get("nFaces") or p.get("n_cells") or p.get("nCells") or 0,
            ))
        elif isinstance(p, str):
            patches.append(MeshPatchV1(name=p, type="patch"))
    
    # Normalize checkMesh
    check_mesh = None
    check_mesh_data = mesh_data.get("check_mesh") or mesh_data.get("checkMesh")
    
    if check_mesh_data and isinstance(check_mesh_data, dict):
        check_mesh = CheckMeshInfoV1(
            cells=check_mesh_data.get("cells", 1),
            faces=check_mesh_data.get("faces", 1),
            points=check_mesh_data.get("points", 1),
            bounding_box=check_mesh_data.get("bounding_box") or check_mesh_data.get("boundingBox"),
            characteristic_length=check_mesh_data.get("characteristic_length") or check_mesh_data.get("characteristicLength"),
            max_aspect_ratio=check_mesh_data.get("max_aspect_ratio") or check_mesh_data.get("maxAspectRatio"),
            max_skewness=check_mesh_data.get("max_skewness") or check_mesh_data.get("maxSkewness"),
            max_non_orthogonality=check_mesh_data.get("max_non_orthogonality") or check_mesh_data.get("maxNonOrthogonality"),
            avg_non_orthogonality=check_mesh_data.get("avg_non_orthogonality") or check_mesh_data.get("avgNonOrthogonality"),
            n_severe_non_ortho=check_mesh_data.get("n_severe_non_ortho") or check_mesh_data.get("nSevereNonOrtho"),
            mesh_ok=check_mesh_data.get("mesh_ok") if check_mesh_data.get("mesh_ok") is not None else check_mesh_data.get("meshOk"),
        )
    
    return MeshInfoV1(
        mesh_id=mesh_id,
        file_name=mesh_data.get("file_name") or mesh_data.get("fileName"),
        patches=patches,
        check_mesh=check_mesh,
    )


def _normalize_physics(config: dict[str, Any]) -> PhysicsV1:
    """Normalize physics configuration."""
    physics_data = config.get("physics", {})
    
    if not isinstance(physics_data, dict):
        physics_data = {}
    
    # Extract flow regime
    flow_regime_str = (
        physics_data.get("flow_regime") or
        physics_data.get("flowRegime") or
        config.get("flow_regime") or
        config.get("flowRegime") or
        "turbulent"
    )
    
    try:
        flow_regime = FlowRegime(flow_regime_str.lower())
    except ValueError:
        flow_regime = FlowRegime.TURBULENT
    
    # Extract time scheme
    time_scheme_str = (
        physics_data.get("time_scheme") or
        physics_data.get("timeScheme") or
        config.get("time_scheme") or
        config.get("timeScheme") or
        config.get("time_stepping") or
        "steady"
    )
    
    try:
        time_scheme = TimeScheme(time_scheme_str.lower())
    except ValueError:
        time_scheme = TimeScheme.STEADY
    
    # Extract compressibility
    compressibility_str = (
        physics_data.get("compressibility") or
        config.get("compressibility") or
        "incompressible"
    )
    
    # Extract heat transfer
    heat_transfer = (
        physics_data.get("heat_transfer") or
        physics_data.get("heatTransfer") or
        physics_data.get("enable_heat_transfer") or
        physics_data.get("enableHeatTransfer") or
        config.get("heat_transfer") or
        config.get("enableHeatTransfer") or
        False
    )
    
    # Extract turbulence model
    turbulence_model = (
        physics_data.get("turbulence_model") or
        physics_data.get("turbulenceModel") or
        config.get("turbulence_model") or
        config.get("turbulenceModel")
    )
    
    return PhysicsV1(
        flow_regime=flow_regime,
        time_scheme=time_scheme,
        compressibility=compressibility_str,
        heat_transfer=bool(heat_transfer),
        turbulence_model=turbulence_model,
    )


def _normalize_solver(config: dict[str, Any]) -> SolverV1:
    """Normalize solver configuration."""
    solver_data = config.get("solver", {})
    
    # Handle string solver (legacy)
    if isinstance(solver_data, str):
        return SolverV1(type=solver_data)
    
    if not isinstance(solver_data, dict):
        solver_data = {}
    
    return SolverV1(
        type=(
            solver_data.get("type") or
            solver_data.get("name") or
            config.get("solver_type") or
            "simpleFoam"
        ),
        max_iterations=(
            solver_data.get("max_iterations") or
            solver_data.get("maxIterations") or
            config.get("max_iterations") or
            config.get("maxIterations") or
            1000
        ),
        convergence_criteria=(
            solver_data.get("convergence_criteria") or
            solver_data.get("convergenceCriteria") or
            config.get("convergence_criteria") or
            config.get("convergenceCriteria") or
            1e-6
        ),
        end_time=(
            solver_data.get("end_time") or
            solver_data.get("endTime") or
            config.get("end_time") or
            config.get("endTime")
        ),
        delta_t=(
            solver_data.get("delta_t") or
            solver_data.get("deltaT") or
            config.get("delta_t") or
            config.get("deltaT")
        ),
        write_interval=(
            solver_data.get("write_interval") or
            solver_data.get("writeInterval") or
            config.get("write_interval") or
            config.get("writeInterval") or
            100
        ),
    )


def _normalize_fluid(config: dict[str, Any]) -> FluidV1:
    """Normalize fluid configuration."""
    fluid_data = config.get("fluid", {})
    
    if not isinstance(fluid_data, dict):
        fluid_data = {}
    
    # Extract kinematic viscosity from multiple possible locations
    kinematic_viscosity = (
        fluid_data.get("kinematic_viscosity") or
        fluid_data.get("kinematicViscosity") or
        fluid_data.get("viscosity") or
        config.get("kinematic_viscosity") or
        config.get("viscosity") or
        config.get("nu") or
        1.5e-5  # Default for air
    )
    
    # Extract density
    density = (
        fluid_data.get("density") or
        fluid_data.get("rho") or
        config.get("density") or
        config.get("rho") or
        1.225  # Default for air
    )
    
    return FluidV1(
        name=fluid_data.get("name", "air"),
        density=float(density),
        kinematic_viscosity=float(kinematic_viscosity),
        dynamic_viscosity=fluid_data.get("dynamic_viscosity") or fluid_data.get("dynamicViscosity"),
        specific_heat=fluid_data.get("specific_heat") or fluid_data.get("specificHeat") or fluid_data.get("cp"),
        thermal_conductivity=fluid_data.get("thermal_conductivity") or fluid_data.get("thermalConductivity"),
        prandtl_number=fluid_data.get("prandtl_number") or fluid_data.get("prandtlNumber") or fluid_data.get("pr"),
    )


def _normalize_geometry(config: dict[str, Any]) -> GeometryV1 | None:
    """Normalize geometry configuration."""
    geometry_data = config.get("geometry", {})
    
    if not isinstance(geometry_data, dict) or not geometry_data:
        return None
    
    return GeometryV1(
        type=geometry_data.get("type"),
        diameter=geometry_data.get("diameter"),
        length=geometry_data.get("length"),
        width=geometry_data.get("width"),
        height=geometry_data.get("height"),
        radius=geometry_data.get("radius"),
        chord=geometry_data.get("chord"),
    )


def _normalize_boundary_conditions(
    config: dict[str, Any],
    mesh: MeshInfoV1 | None,
) -> dict[str, BoundaryConditionV1]:
    """Normalize boundary conditions."""
    bcs: dict[str, BoundaryConditionV1] = {}
    
    # Try new format first: boundary_conditions dict
    bc_data = (
        config.get("boundary_conditions") or
        config.get("boundaryConditions") or
        {}
    )
    
    if isinstance(bc_data, dict):
        for patch_name, bc_config in bc_data.items():
            if isinstance(bc_config, dict):
                bcs[patch_name] = _normalize_single_bc(patch_name, bc_config)
    
    # Also check for legacy inlet/outlet/walls at top level
    if "inlet" in config and "inlet" not in bcs:
        inlet_config = config["inlet"]
        if isinstance(inlet_config, dict):
            bcs["inlet"] = _normalize_single_bc("inlet", {
                "patch_type": "inlet",
                **inlet_config
            })
    
    if "outlet" in config and "outlet" not in bcs:
        outlet_config = config["outlet"]
        if isinstance(outlet_config, dict):
            bcs["outlet"] = _normalize_single_bc("outlet", {
                "patch_type": "outlet",
                **outlet_config
            })
    
    if "walls" in config and "walls" not in bcs:
        walls_config = config["walls"]
        if isinstance(walls_config, dict):
            bcs["walls"] = _normalize_single_bc("walls", {
                "patch_type": "wall",
                **walls_config
            })
    
    # If we have mesh patches but no BCs for them, infer from patch names
    if mesh:
        for patch in mesh.patches:
            if patch.name not in bcs:
                inferred_type = _infer_patch_type(patch.name, patch.type)
                if inferred_type:
                    bcs[patch.name] = BoundaryConditionV1(patch_type=inferred_type)
    
    return bcs


def _normalize_single_bc(
    patch_name: str,
    bc_config: dict[str, Any],
) -> BoundaryConditionV1:
    """Normalize a single boundary condition."""
    # Determine patch type
    patch_type_str = (
        bc_config.get("patch_type") or
        bc_config.get("patchType") or
        bc_config.get("type") or
        bc_config.get("suggested_type") or
        bc_config.get("suggestedType") or
        _infer_patch_type(patch_name, "") or
        "wall"
    )
    
    try:
        patch_type = BoundaryType(patch_type_str.lower())
    except ValueError:
        patch_type = patch_type_str
    
    # Normalize velocity BC
    velocity = None
    vel_data = bc_config.get("velocity")
    if vel_data:
        if isinstance(vel_data, dict):
            _vel_type = vel_data.get("type", "fixedValue")
            _entries = vel_data.get("entries") or {}

            def _scalar_or_none(v: Any) -> float | None:
                """Return v as float only if it is a scalar number, not a list."""
                return float(v) if isinstance(v, (int, float)) else None

            _mfr = (
                _scalar_or_none(vel_data.get("massFlowRate"))
                or _scalar_or_none(vel_data.get("mass_flow_rate"))
                or _scalar_or_none(_entries.get("massFlowRate"))
            )
            _vfr = (
                _scalar_or_none(vel_data.get("volumetricFlowRate"))
                or _scalar_or_none(vel_data.get("volumetric_flow_rate"))
                or _scalar_or_none(_entries.get("volumetricFlowRate"))
            )
            # For flow-rate inlets, a plain scalar "value" is the flow rate, not a velocity
            if _vel_type == "flowRateInletVelocity" and _mfr is None and _vfr is None:
                _raw = vel_data.get("value")
                if isinstance(_raw, (int, float)):
                    _mfr = float(_raw)
            # Optional flowRateInletVelocity fields
            _rho_field = vel_data.get("rho") or vel_data.get("rho_field")
            _rho_inlet = vel_data.get("rhoInlet") or vel_data.get("rho_inlet")
            _extrap = vel_data.get("extrapolateProfile") or vel_data.get("extrapolate_profile")
            velocity = VelocityBCV1(
                type=_vel_type,
                value=vel_data.get("value"),
                magnitude=vel_data.get("magnitude"),
                direction=vel_data.get("direction"),
                mass_flow_rate=float(_mfr) if _mfr is not None else None,
                volumetric_flow_rate=float(_vfr) if _vfr is not None else None,
                rho_field=str(_rho_field) if _rho_field is not None else None,
                rho_inlet=float(_rho_inlet) if _rho_inlet is not None else None,
                extrapolate_profile=bool(_extrap) if _extrap is not None else None,
            )
        elif isinstance(vel_data, (int, float)):
            velocity = VelocityBCV1(type="fixedValue", value=float(vel_data))
        elif isinstance(vel_data, list):
            velocity = VelocityBCV1(type="fixedValue", value=vel_data)
    
    # Normalize pressure BC
    pressure = None
    press_data = bc_config.get("pressure")
    if press_data:
        if isinstance(press_data, dict):
            pressure = PressureBCV1(
                type=press_data.get("type", "fixedValue"),
                value=press_data.get("value"),
            )
        elif isinstance(press_data, (int, float)):
            pressure = PressureBCV1(type="fixedValue", value=float(press_data))
    
    # Normalize temperature BC
    temperature = None
    temp_data = bc_config.get("temperature")
    if temp_data:
        if isinstance(temp_data, dict):
            temperature = TemperatureBCV1(
                type=temp_data.get("type", "fixedValue"),
                value=temp_data.get("value"),
            )
        elif isinstance(temp_data, (int, float)):
            temperature = TemperatureBCV1(type="fixedValue", value=float(temp_data))
    
    return BoundaryConditionV1(
        patch_type=patch_type,
        velocity=velocity,
        pressure=pressure,
        temperature=temperature,
    )


def _infer_patch_type(patch_name: str, openfoam_type: str) -> BoundaryType | None:
    """Infer boundary type from patch name and OpenFOAM type."""
    name_lower = patch_name.lower()
    
    # Check OpenFOAM type first
    if openfoam_type.lower() == "wall":
        return BoundaryType.WALL
    if openfoam_type.lower() == "empty":
        return BoundaryType.EMPTY
    if openfoam_type.lower() in ("symmetry", "symmetryplane"):
        return BoundaryType.SYMMETRY
    if openfoam_type.lower() in ("cyclic", "cyclicami"):
        return BoundaryType.PERIODIC
    
    # Infer from name
    if any(x in name_lower for x in ["inlet", "inflow", "in"]):
        return BoundaryType.INLET
    if any(x in name_lower for x in ["outlet", "outflow", "out", "exit"]):
        return BoundaryType.OUTLET
    if any(x in name_lower for x in ["wall", "walls", "surface"]):
        return BoundaryType.WALL
    if any(x in name_lower for x in ["sym", "symmetry"]):
        return BoundaryType.SYMMETRY
    if any(x in name_lower for x in ["front", "back", "empty"]):
        return BoundaryType.EMPTY
    
    return None


def _normalize_kpi_targets(config: dict[str, Any]) -> list:
    """Normalize KPI targets."""
    from simd_agent.models import KPITargetV1
    
    targets = []
    kpi_data = config.get("kpi_targets") or config.get("kpiTargets") or {}
    
    if isinstance(kpi_data, list):
        for item in kpi_data:
            if isinstance(item, dict) and "name" in item:
                targets.append(KPITargetV1(
                    name=item["name"],
                    value=item.get("value", 0),
                    unit=item.get("unit", ""),
                    tolerance=item.get("tolerance"),
                ))
    elif isinstance(kpi_data, dict):
        # Handle structured format
        for key in ["pressure_drop", "pressureDrop", "flow_rate", "flowRate",
                    "temperature", "velocity"]:
            if key in kpi_data:
                val = kpi_data[key]
                if isinstance(val, dict):
                    targets.append(KPITargetV1(
                        name=camel_to_snake(key),
                        value=val.get("value", 0),
                        unit=val.get("unit", ""),
                    ))
        
        # Handle custom KPIs
        for custom in kpi_data.get("custom", []):
            if isinstance(custom, dict):
                targets.append(KPITargetV1(
                    name=custom.get("name", "custom"),
                    value=custom.get("value", 0),
                    unit=custom.get("unit", ""),
                ))
    
    return targets


def validate_config_for_operation(
    config: SimulationConfigV1,
    operation: Operation,
    user_requirements: str = "",
) -> ConfigValidationResult:
    """
    Validate a normalized config for a specific operation.
    
    Args:
        config: Normalized simulation config
        operation: The operation to validate for
        user_requirements: Optional user requirements for context
        
    Returns:
        ConfigValidationResult with validation details
    """
    missing_fields: list[MissingFieldInfo] = []
    warnings: list[str] = []
    errors: list[str] = []
    
    # Always check mesh
    if not config.mesh or not config.mesh.mesh_id:
        missing_fields.append(MissingFieldInfo(
            field="mesh.mesh_id",
            description="Mesh identifier is required",
            required_for="all",
            suggested_value=None,
        ))
    
    # For codegen, we need complete boundary conditions
    if operation == Operation.CFD_CODEGEN_RUN:
        # Check for inlet
        has_inlet = any(bc.is_inlet() for bc in config.boundary_conditions.values())
        if not has_inlet:
            missing_fields.append(MissingFieldInfo(
                field="boundary_conditions.inlet",
                description="At least one inlet boundary condition is required",
                required_for="codegen",
                suggested_value={"patch_type": "inlet", "velocity": {"type": "fixedValue", "value": [1, 0, 0]}},
            ))
        
        # Check inlet has velocity
        for name, bc in config.boundary_conditions.items():
            if bc.is_inlet():
                if not bc.velocity or bc.velocity.get_magnitude() is None:
                    missing_fields.append(MissingFieldInfo(
                        field=f"boundary_conditions.{name}.velocity",
                        description=f"Inlet '{name}' requires velocity value",
                        required_for="codegen",
                        suggested_value={"type": "fixedValue", "value": [1, 0, 0]},
                    ))
        
        # Check for outlet
        has_outlet = any(bc.is_outlet() for bc in config.boundary_conditions.values())
        if not has_outlet:
            missing_fields.append(MissingFieldInfo(
                field="boundary_conditions.outlet",
                description="At least one outlet boundary condition is required",
                required_for="codegen",
                suggested_value={"patch_type": "outlet", "pressure": {"type": "fixedValue", "value": 0}},
            ))
        
        # Check mesh patches have BCs
        if config.mesh:
            patches_without_bc = config.get_patches_without_bc()
            # Filter out empty/symmetry patches which don't always need explicit BCs
            critical_patches = [
                p for p in patches_without_bc
                if not any(x in p.lower() for x in ["empty", "front", "back"])
            ]
            if critical_patches:
                warnings.append(
                    f"Mesh patches without explicit boundary conditions: {critical_patches}"
                )
    
    # Validate units sanity
    if config.fluid.kinematic_viscosity <= 0:
        errors.append("Kinematic viscosity must be positive")
    if config.fluid.density <= 0:
        errors.append("Density must be positive")
    
    # Check for velocity sanity
    inlet_vel = config.get_inlet_velocity_magnitude()
    if inlet_vel is not None and inlet_vel > 340:  # > speed of sound
        warnings.append(f"Inlet velocity {inlet_vel:.1f} m/s is supersonic; consider compressible solver")
    
    is_complete = len(missing_fields) == 0 and len(errors) == 0
    is_valid = len(errors) == 0
    
    return ConfigValidationResult(
        is_valid=is_valid,
        is_complete=is_complete,
        normalized_config=config if is_valid else None,
        missing_fields=missing_fields,
        warnings=warnings,
        errors=errors,
    )


def config_to_dict(config: SimulationConfigV1) -> dict[str, Any]:
    """Convert SimulationConfigV1 to a plain dictionary for storage/serialization."""
    return config.model_dump(exclude_none=True)
