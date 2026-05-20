# simd_agent/config_normalizer.py
"""
Configuration normalizer for backward-compatible parsing of simulation configs.

Handles:
- Legacy formats (mesh as string, mesh.mesh_id, etc.)
- camelCase to snake_case conversion
- Coercion of common field variations
- Produces canonical SimulationConfigV1
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
    KPITargetV1,
    MeshInfoV1,
    MeshPatchV1,
    MissingFieldInfo,
    PhysicsV1,
    PressureBCV1,
    SimulationConfigV1,
    SolverV1,
    TemperatureBCV1,
    TimeScheme,
    TurbulenceBCV1,
    VelocityBCV1,
)

logger = logging.getLogger(__name__)


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    # Handle consecutive uppercase letters (e.g., "XMLParser" -> "xml_parser")
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def deep_convert_keys(obj: Any, converter: callable) -> Any:
    """Recursively convert all dictionary keys using the converter function."""
    if isinstance(obj, dict):
        return {converter(k): deep_convert_keys(v, converter) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_convert_keys(item, converter) for item in obj]
    else:
        return obj


def detect_format(config: dict[str, Any]) -> str:
    """
    Detect the format of the incoming configuration.
    
    Returns:
        One of: "v1_full", "v1_partial", "legacy_minimal", "legacy_camel", "unknown"
    """
    if not config:
        return "empty"
    
    keys = set(config.keys())
    
    # Check for V1 full format (has nested objects)
    v1_keys = {"mesh", "physics", "solver", "fluid", "boundary_conditions"}
    if len(keys & v1_keys) >= 3:
        # Check if mesh is a proper object
        mesh = config.get("mesh")
        if isinstance(mesh, dict) and ("mesh_id" in mesh or "meshId" in mesh):
            return "v1_full"
        return "v1_partial"
    
    # Check for camelCase keys
    camel_keys = {"boundaryConditions", "flowRegime", "timeScheme", "meshId", 
                  "checkMesh", "turbulenceModel", "kpiTargets"}
    if keys & camel_keys:
        return "legacy_camel"
    
    # Check for minimal legacy format
    if "mesh" in keys and isinstance(config.get("mesh"), str):
        return "legacy_minimal"
    
    # Check for mesh_id or meshId at top level
    if "mesh_id" in keys or "meshId" in keys:
        return "legacy_minimal"
    
    return "unknown"


def normalize_mesh(raw: Any) -> MeshInfoV1 | None:
    """
    Normalize mesh information from various input formats.
    
    Accepts:
    - String (mesh_id only)
    - {"mesh_id": "..."} or {"meshId": "..."}
    - Full MeshInfoV1-like object
    """
    if raw is None:
        return None
    
    if isinstance(raw, str):
        # Legacy: mesh is just a string ID
        return MeshInfoV1(mesh_id=raw)
    
    if not isinstance(raw, dict):
        logger.warning(f"Unexpected mesh type: {type(raw)}")
        return None
    
    # Convert to snake_case
    mesh_data = deep_convert_keys(raw, camel_to_snake)
    
    # Extract mesh_id
    mesh_id = mesh_data.get("mesh_id") or mesh_data.get("id")
    if not mesh_id:
        logger.warning("Mesh data missing mesh_id")
        return None
    
    # Parse patches
    patches = []
    # Use ORIGINAL patches (before deep_convert_keys) to preserve patch names
    raw_patches_orig = raw.get("patches", [])
    raw_patches = mesh_data.get("patches", [])
    for i, p in enumerate(raw_patches):
        if isinstance(p, dict):
            # Get the original patch name (before camel_to_snake conversion)
            orig_p = raw_patches_orig[i] if i < len(raw_patches_orig) and isinstance(raw_patches_orig[i], dict) else p
            p_name = orig_p.get("name", p.get("name", "unknown"))
            
            # Accept "patch_type", "patchType", or "type" for the patch type
            p_type = (
                p.get("patch_type") or p.get("type", "patch")
            )
            
            # FORCE constraint patch types based on well-known names
            # frontAndBack is ALWAYS empty in 2D OpenFOAM simulations
            name_lower = p_name.lower().replace("_", "")
            if name_lower in ("frontandback", "frontback", "defaultfaces"):
                p_type = "empty"
            
            patches.append(MeshPatchV1(
                name=p_name,
                type=p_type,
                n_faces=p.get("n_faces") or p.get("n_cells") or 0,
            ))
    
    # Parse checkMesh
    check_mesh = None
    raw_check = mesh_data.get("check_mesh")
    if isinstance(raw_check, dict):
        try:
            check_mesh = CheckMeshInfoV1(
                cells=raw_check.get("cells", 1),
                faces=raw_check.get("faces", 1),
                points=raw_check.get("points", 1),
                bounding_box=raw_check.get("bounding_box"),
                characteristic_length=raw_check.get("characteristic_length"),
                max_aspect_ratio=raw_check.get("max_aspect_ratio"),
                max_skewness=raw_check.get("max_skewness"),
                max_non_orthogonality=raw_check.get("max_non_orthogonality"),
                avg_non_orthogonality=raw_check.get("avg_non_orthogonality"),
                n_severe_non_ortho=raw_check.get("n_severe_non_ortho"),
                mesh_ok=raw_check.get("mesh_ok"),
            )
        except Exception as e:
            logger.warning(f"Failed to parse checkMesh: {e}")
    
    return MeshInfoV1(
        mesh_id=mesh_id,
        file_name=mesh_data.get("file_name"),
        patches=patches,
        check_mesh=check_mesh,
    )


def normalize_physics(raw: dict[str, Any], top_level: dict[str, Any]) -> PhysicsV1:
    """
    Normalize physics settings from various input formats.
    
    Looks for physics fields in both the raw physics object and top-level config.
    
    NOTE: flow_regime defaults to None (not TURBULENT) so it can be auto-detected
    from Reynolds number during linting if not explicitly specified.
    """
    # Merge top-level fields into physics
    data = {}

    # Top-level overrides
    for key in ["flow_regime", "flowRegime", "time_scheme", "timeScheme",
                "compressibility", "heat_transfer", "heatTransfer",
                "enable_heat_transfer", "enableHeatTransfer",
                "turbulence_model", "turbulenceModel"]:
        snake_key = camel_to_snake(key)
        if key in top_level:
            data[snake_key] = top_level[key]

    # Precheck shape — turbulence is a NESTED sub-object, not a flat field.
    # The precheck emits ``{"turbulence": {"model": "kOmegaSST", ...}}``.
    # Without this lookup, the linter sees ``turbulence_model = None``,
    # defaults to laminar, and the case ships with ``simulationType laminar``
    # (the regression that crashed rhoSimpleFoam with SIGFPE in iteration 64).
    _turb_obj = top_level.get("turbulence")
    if isinstance(_turb_obj, dict):
        _nested_model = (
            _turb_obj.get("model")
            or _turb_obj.get("RASModel")
            or _turb_obj.get("turbulenceModel")
        )
        if _nested_model and not data.get("turbulence_model"):
            data["turbulence_model"] = _nested_model

    # Physics object (if provided)
    if raw:
        physics_data = deep_convert_keys(raw, camel_to_snake)
        data.update(physics_data)

    # Parse flow_regime - None by default, auto-detected from Reynolds
    flow_regime: FlowRegime | None = None
    raw_regime = data.get("flow_regime")
    if raw_regime:
        try:
            flow_regime = FlowRegime(raw_regime.lower() if isinstance(raw_regime, str) else raw_regime)
        except (ValueError, AttributeError):
            pass  # Invalid regime -> will be auto-detected

    # Soft inference: a non-laminar turbulence model implies turbulent flow.
    # When the user provides an explicit model (e.g. kOmegaSST) but the flow
    # regime is missing, we trust the model.  Prevents the linter from
    # auto-detecting laminar from a tiny / missing Reynolds number when the
    # case clearly intends RAS / LES.
    _t_model = data.get("turbulence_model")
    if (
        flow_regime is None
        and isinstance(_t_model, str)
        and _t_model
        and _t_model.lower() not in ("laminar", "none")
    ):
        flow_regime = FlowRegime.TURBULENT
    
    # Parse time_scheme
    time_scheme = TimeScheme.STEADY
    raw_time = data.get("time_scheme")
    if raw_time:
        try:
            time_scheme = TimeScheme(raw_time.lower() if isinstance(raw_time, str) else raw_time)
        except (ValueError, AttributeError):
            pass
    
    # Heat transfer
    heat_transfer = (
        data.get("heat_transfer") or 
        data.get("enable_heat_transfer") or 
        False
    )
    
    return PhysicsV1(
        flow_regime=flow_regime,  # None by default - will be auto-detected
        time_scheme=time_scheme,
        compressibility=data.get("compressibility", "incompressible"),
        heat_transfer=bool(heat_transfer),
        turbulence_model=data.get("turbulence_model"),
    )


def normalize_solver(raw: dict[str, Any] | str | None, top_level: dict[str, Any]) -> SolverV1:
    """Normalize solver settings.
    
    Handles:
    - String solver name: {"solver": "pimpleFoam"}
    - Dict with type: {"solver": {"type": "simpleFoam"}}
    - Dict without type: {"solver": {"max_iterations": 1000}} -> defaults to simpleFoam
    """
    data = {}
    
    # Top-level fields (excluding "solver" which needs special handling)
    for key in ["max_iterations", "maxIterations", 
                "convergence_criteria", "convergenceCriteria",
                "end_time", "endTime", "delta_t", "deltaT",
                "write_interval", "writeInterval"]:
        snake_key = camel_to_snake(key)
        if key in top_level:
            data[snake_key] = top_level[key]
    
    # Handle solver field specially - can be string or dict
    top_solver = top_level.get("solver")
    if isinstance(top_solver, str):
        # Direct solver name: {"solver": "pimpleFoam"}
        data["type"] = top_solver
    elif isinstance(top_solver, dict):
        # Solver object: {"solver": {"type": "simpleFoam", "max_iterations": 1000}}
        solver_data = deep_convert_keys(top_solver, camel_to_snake)
        data.update(solver_data)
    
    # Handle raw parameter (same logic)
    if isinstance(raw, str):
        data["type"] = raw
    elif isinstance(raw, dict):
        solver_data = deep_convert_keys(raw, camel_to_snake)
        data.update(solver_data)
    
    # Get solver type - must be a string, default to simpleFoam
    solver_type = data.get("type")
    if not isinstance(solver_type, str) or not solver_type:
        solver_type = "simpleFoam"
    
    return SolverV1(
        type=solver_type,
        max_iterations=int(data.get("max_iterations", 1000)),
        convergence_criteria=float(data.get("convergence_criteria", 1e-6)),
        end_time=data.get("end_time"),
        delta_t=data.get("delta_t"),
        write_interval=int(data.get("write_interval", 100)),
    )


def normalize_fluid(raw: dict[str, Any] | None, top_level: dict[str, Any]) -> FluidV1:
    """Normalize fluid properties."""
    data = {}
    
    # Top-level fields (legacy format)
    for key in ["viscosity", "kinematic_viscosity", "kinematicViscosity",
                "density", "nu", "rho"]:
        snake_key = camel_to_snake(key)
        if key in top_level:
            data[snake_key] = top_level[key]
    
    if raw:
        fluid_data = deep_convert_keys(raw, camel_to_snake)
        data.update(fluid_data)
    
    # Handle viscosity aliases
    kinematic_viscosity = (
        data.get("kinematic_viscosity") or
        data.get("viscosity") or
        data.get("nu") or
        1.5e-5  # Default air viscosity
    )
    
    density = (
        data.get("density") or
        data.get("rho") or
        1.225  # Default air density
    )
    
    return FluidV1(
        name=data.get("name", "air"),
        density=float(density),
        kinematic_viscosity=float(kinematic_viscosity),
        dynamic_viscosity=data.get("dynamic_viscosity"),
        specific_heat=data.get("specific_heat"),
        thermal_conductivity=data.get("thermal_conductivity"),
        prandtl_number=data.get("prandtl_number"),
    )


def normalize_geometry(raw: dict[str, Any] | None, top_level: dict[str, Any]) -> GeometryV1 | None:
    """Normalize geometry settings."""
    data = {}
    
    # Top-level geometry field
    if "geometry" in top_level and isinstance(top_level["geometry"], dict):
        data.update(deep_convert_keys(top_level["geometry"], camel_to_snake))
    
    if raw:
        data.update(deep_convert_keys(raw, camel_to_snake))
    
    if not data:
        return None
    
    return GeometryV1(
        type=data.get("type"),
        diameter=data.get("diameter"),
        length=data.get("length"),
        width=data.get("width"),
        height=data.get("height"),
        radius=data.get("radius"),
        chord=data.get("chord"),
    )


def infer_patch_type(patch_name: str) -> BoundaryType:
    """Infer boundary type from patch name."""
    name_lower = patch_name.lower()
    
    # Check outlet FIRST to avoid "out" matching in "outlet" when checking for "in" later
    # Also use word boundaries where possible
    if any(x in name_lower for x in ["outlet", "outflow", "exit"]):
        return BoundaryType.OUTLET
    if name_lower.endswith("_out") or name_lower == "out":
        return BoundaryType.OUTLET
    
    # Now check inlet (after outlet to avoid "out" false positive)
    if any(x in name_lower for x in ["inlet", "inflow"]):
        return BoundaryType.INLET
    if name_lower.endswith("_in") or name_lower == "in":
        return BoundaryType.INLET
    
    if any(x in name_lower for x in ["wall", "surface"]):
        return BoundaryType.WALL
    if any(x in name_lower for x in ["sym", "symmetry"]):
        return BoundaryType.SYMMETRY
    if any(x in name_lower for x in ["periodic", "cyclic"]):
        return BoundaryType.PERIODIC
    if any(x in name_lower for x in ["empty", "front", "back"]):
        return BoundaryType.EMPTY
    
    # Default to wall
    return BoundaryType.WALL


def normalize_boundary_condition(raw: dict[str, Any], patch_name: str) -> BoundaryConditionV1:
    """Normalize a single boundary condition.
    
    Handles two formats:
    1. Semantic: {"patch_type": "wall", "velocity": {"type": "noSlip"}, "pressure": {...}}
    2. Per-field: {"U": {"type": "empty"}, "p": {"type": "empty"}, ...}
    """
    data = deep_convert_keys(raw, camel_to_snake)
    
    # Detect per-field format (keys are field names like U, p, T, k, omega, nut)
    field_names = {"u", "p", "t", "k", "omega", "nut", "epsilon"}
    data_keys_lower = {k.lower() for k in data.keys()}
    if data_keys_lower & field_names and not data.get("patch_type") and not data.get("velocity"):
        # Per-field format — infer patch type from the field BCs
        # If all fields have type "empty", it's an empty patch
        all_types = set()
        for k, v in data.items():
            if isinstance(v, dict) and "type" in v:
                all_types.add(v["type"].lower())
        
        if all_types == {"empty"}:
            return BoundaryConditionV1(patch_type=BoundaryType.EMPTY)
        if all_types == {"symmetry"}:
            return BoundaryConditionV1(patch_type=BoundaryType.SYMMETRY)
        
        # Mixed types — try to extract velocity/pressure/temperature from field names
        vel_data = data.get("u") or data.get("U")
        pres_data = data.get("p")
        temp_data = data.get("t") or data.get("T")
        
        # Rebuild in semantic format and recurse
        semantic = {}
        if vel_data and isinstance(vel_data, dict):
            semantic["velocity"] = vel_data
        if pres_data and isinstance(pres_data, dict):
            semantic["pressure"] = pres_data
        if temp_data and isinstance(temp_data, dict):
            semantic["temperature"] = temp_data
        
        # Extract turbulence fields
        for turb_key in ["k", "epsilon", "omega", "nut"]:
            turb_data = data.get(turb_key)
            if isinstance(turb_data, dict):
                semantic[turb_key] = turb_data
        
        if semantic:
            return normalize_boundary_condition(semantic, patch_name)
    
    # Determine patch type
    patch_type_raw = data.get("patch_type") or data.get("type") or data.get("suggested_type")
    if patch_type_raw:
        try:
            patch_type = BoundaryType(patch_type_raw.lower())
        except (ValueError, AttributeError):
            patch_type = infer_patch_type(patch_name)
    else:
        patch_type = infer_patch_type(patch_name)
    
    # Parse velocity BC
    velocity = None
    raw_vel = data.get("velocity")
    if raw_vel:
        if isinstance(raw_vel, dict):
            vel_data = deep_convert_keys(raw_vel, camel_to_snake)
            velocity = VelocityBCV1(
                type=vel_data.get("type", "fixedValue"),
                value=vel_data.get("value"),
                magnitude=vel_data.get("magnitude"),
                direction=vel_data.get("direction"),
            )
        elif isinstance(raw_vel, (list, int, float)):
            velocity = VelocityBCV1(type="fixedValue", value=raw_vel)
    
    # Parse pressure BC
    pressure = None
    raw_pres = data.get("pressure")
    if raw_pres:
        if isinstance(raw_pres, dict):
            pres_data = deep_convert_keys(raw_pres, camel_to_snake)
            pressure = PressureBCV1(
                type=pres_data.get("type", "fixedValue"),
                value=pres_data.get("value"),
            )
        elif isinstance(raw_pres, (int, float)):
            pressure = PressureBCV1(type="fixedValue", value=float(raw_pres))
    
    # Parse temperature BC
    temperature = None
    raw_temp = data.get("temperature")
    if raw_temp:
        if isinstance(raw_temp, dict):
            temp_data = deep_convert_keys(raw_temp, camel_to_snake)
            temperature = TemperatureBCV1(
                type=temp_data.get("type", "fixedValue"),
                value=temp_data.get("value"),
            )
        elif isinstance(raw_temp, (int, float)):
            temperature = TemperatureBCV1(type="fixedValue", value=float(raw_temp))
    
    # Parse turbulence BC
    turbulence = None
    if any(k in data for k in ["k", "epsilon", "omega", "nut"]):
        turbulence = TurbulenceBCV1(
            k=data.get("k"),
            epsilon=data.get("epsilon"),
            omega=data.get("omega"),
            nut=data.get("nut"),
        )
    
    return BoundaryConditionV1(
        patch_type=patch_type,
        velocity=velocity,
        pressure=pressure,
        temperature=temperature,
        turbulence=turbulence,
    )


def normalize_boundary_conditions(
    raw: dict[str, Any] | None,
    top_level: dict[str, Any],
    mesh: MeshInfoV1 | None = None,
) -> dict[str, BoundaryConditionV1]:
    """
    Normalize boundary conditions from various formats.
    
    Handles:
    - {"boundary_conditions": {...}} 
    - {"boundaryConditions": {...}}
    - Legacy {"inlet": {...}, "outlet": {...}}
    """
    bcs: dict[str, BoundaryConditionV1] = {}
    
    # New format: boundary_conditions or boundaryConditions
    bc_data = raw or top_level.get("boundary_conditions") or top_level.get("boundaryConditions") or {}
    
    if bc_data and isinstance(bc_data, dict):
        for patch_name, bc_raw in bc_data.items():
            if isinstance(bc_raw, dict):
                bcs[patch_name] = normalize_boundary_condition(bc_raw, patch_name)
    
    # Legacy format: inlet/outlet/walls at top level
    for legacy_key in ["inlet", "outlet", "walls", "wall"]:
        if legacy_key in top_level and isinstance(top_level[legacy_key], dict):
            patch_name = legacy_key if legacy_key != "walls" else "wall"
            if patch_name not in bcs:
                bcs[patch_name] = normalize_boundary_condition(
                    top_level[legacy_key], 
                    patch_name
                )
    
    # If we have mesh patches but no BCs, create stub entries
    if mesh and mesh.patches:
        for patch in mesh.patches:
            if patch.name not in bcs:
                # Create a stub BC based on patch name inference
                patch_type = infer_patch_type(patch.name)
                bcs[patch.name] = BoundaryConditionV1(patch_type=patch_type)
    
    return bcs


def normalize_kpi_targets(raw: Any) -> list[KPITargetV1]:
    """Normalize KPI targets from various formats."""
    if not raw:
        return []
    
    targets = []
    
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                item_data = deep_convert_keys(item, camel_to_snake)
                targets.append(KPITargetV1(
                    name=item_data.get("name", "unknown"),
                    value=float(item_data.get("value", 0)),
                    unit=item_data.get("unit", ""),
                    tolerance=item_data.get("tolerance"),
                ))
    elif isinstance(raw, dict):
        # Object format: {pressureDrop: {...}, flowRate: {...}}
        raw_data = deep_convert_keys(raw, camel_to_snake)
        for name, val in raw_data.items():
            if name == "custom" and isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        targets.append(KPITargetV1(
                            name=item.get("name", "custom"),
                            value=float(item.get("value", 0)),
                            unit=item.get("unit", ""),
                        ))
            elif isinstance(val, dict) and "value" in val:
                targets.append(KPITargetV1(
                    name=name,
                    value=float(val.get("value", 0)),
                    unit=val.get("unit", ""),
                ))
    
    return targets


def normalize_config(raw_config: dict[str, Any]) -> tuple[SimulationConfigV1, str, list[str]]:
    """
    Normalize a raw simulation configuration to canonical V1 format.
    
    Args:
        raw_config: Raw configuration dictionary from frontend
        
    Returns:
        Tuple of (normalized_config, detected_format, transformations_applied)
    """
    transformations = []
    
    # Detect format
    detected_format = detect_format(raw_config)
    transformations.append(f"detected_format={detected_format}")
    
    # Convert camelCase to snake_case at top level
    config = deep_convert_keys(raw_config, camel_to_snake)
    if raw_config != config:
        transformations.append("converted_camel_to_snake")
    
    # IMPORTANT: Restore original boundary_conditions keys.
    # deep_convert_keys converts "frontAndBack" → "front_and_back", but these
    # are OpenFOAM patch names (identifiers), not API field names.
    # We must preserve the original patch names exactly.
    raw_bcs = raw_config.get("boundary_conditions") or raw_config.get("boundaryConditions") or {}
    if isinstance(raw_bcs, dict) and raw_bcs:
        # Replace the snake_case-converted BC dict with original keys
        converted_bcs = config.get("boundary_conditions", {})
        if isinstance(converted_bcs, dict):
            restored_bcs = {}
            # Map snake_case keys back to original keys
            orig_keys = list(raw_bcs.keys())
            conv_keys = list(converted_bcs.keys())
            for orig_key, conv_key in zip(orig_keys, conv_keys):
                restored_bcs[orig_key] = converted_bcs[conv_key]
            config["boundary_conditions"] = restored_bcs
            if restored_bcs != converted_bcs:
                transformations.append("restored_original_patch_names")
    
    # Normalize mesh
    mesh = normalize_mesh(config.get("mesh"))
    if mesh:
        transformations.append("normalized_mesh")
    
    # Normalize physics
    physics = normalize_physics(config.get("physics"), config)
    transformations.append("normalized_physics")
    
    # Normalize solver
    solver = normalize_solver(config.get("solver"), config)
    transformations.append("normalized_solver")
    
    # Normalize fluid
    fluid = normalize_fluid(config.get("fluid"), config)
    transformations.append("normalized_fluid")
    
    # Normalize geometry
    geometry = normalize_geometry(config.get("geometry"), config)
    if geometry:
        transformations.append("normalized_geometry")
    
    # Normalize boundary conditions
    bcs = normalize_boundary_conditions(
        config.get("boundary_conditions"),
        config,
        mesh,
    )
    if bcs:
        transformations.append(f"normalized_boundary_conditions({len(bcs)} patches)")
    
    # Normalize KPI targets
    kpi_targets = normalize_kpi_targets(
        config.get("kpi_targets") or config.get("kpi")
    )
    if kpi_targets:
        transformations.append(f"normalized_kpi_targets({len(kpi_targets)} targets)")
    
    # Build normalized config
    normalized = SimulationConfigV1(
        mesh=mesh,
        physics=physics,
        solver=solver,
        fluid=fluid,
        geometry=geometry,
        boundary_conditions=bcs,
        kpi_targets=kpi_targets,
        case_type=config.get("case_type"),
        inlet=config.get("inlet"),
        outlet=config.get("outlet"),
    )
    
    return normalized, detected_format, transformations


def validate_config_completeness(
    config: SimulationConfigV1,
    operation: str = "CFD_CODEGEN_RUN",
) -> ConfigValidationResult:
    """
    Validate that configuration has all required fields for the operation.
    
    Args:
        config: Normalized configuration
        operation: The operation to validate for
        
    Returns:
        ConfigValidationResult with validation status and missing fields
    """
    missing_fields: list[MissingFieldInfo] = []
    warnings: list[str] = []
    errors: list[str] = []
    
    # --- Mesh validation ---
    if not config.mesh:
        missing_fields.append(MissingFieldInfo(
            field="mesh",
            description="Mesh information is required",
            required_for=operation,
        ))
    elif not config.mesh.mesh_id:
        missing_fields.append(MissingFieldInfo(
            field="mesh.mesh_id",
            description="Mesh identifier is required",
            required_for=operation,
        ))
    
    # --- Boundary conditions validation ---
    has_inlet = any(bc.is_inlet() for bc in config.boundary_conditions.values())
    has_outlet = any(bc.is_outlet() for bc in config.boundary_conditions.values())

    # "Closed-domain" cases legitimately have no inlets and no outlets — the
    # flow is driven by something other than a pressure gradient: buoyancy
    # (gravity + ΔT), a moving wall (lid-driven cavity), an MRF rotating
    # zone, an fvOptions momentum source.  These are all valid CFD setups.
    #
    # The inlet/outlet check is a heuristic to catch the common mistake of
    # forgetting to tag a patch.  Skip it when the user has *explicitly*
    # accounted for every mesh patch (every patch has a BC and none of them
    # are tagged inlet/outlet) — meaning they really did mean "closed
    # domain", not "I forgot something".
    _CLOSED_OK = {"wall", "symmetry", "empty", "wedge", "cyclic"}

    def _patch_kind(bc) -> str:
        pt = getattr(bc, "patch_type", "")
        # BoundaryType is a str-Enum, so .value gives the canonical name;
        # fall back to str() for plain-string inputs from the normaliser.
        return getattr(pt, "value", str(pt)).lower()

    is_closed_domain_case = (
        not has_inlet and not has_outlet
        and len(config.boundary_conditions) > 0
        and all(_patch_kind(bc) in _CLOSED_OK
                for bc in config.boundary_conditions.values())
    )

    if not has_inlet and not is_closed_domain_case:
        missing_fields.append(MissingFieldInfo(
            field="boundary_conditions.inlet",
            description="At least one inlet boundary condition is required",
            required_for=operation,
            suggested_value={
                "patch_type": "inlet",
                "velocity": {"type": "fixedValue", "value": [1, 0, 0]}
            },
        ))

    if not has_outlet and not is_closed_domain_case:
        missing_fields.append(MissingFieldInfo(
            field="boundary_conditions.outlet",
            description="At least one outlet boundary condition is required",
            required_for=operation,
            suggested_value={
                "patch_type": "outlet",
                "pressure": {"type": "fixedValue", "value": 0}
            },
        ))
    
    # Check inlet has velocity
    for name, bc in config.boundary_conditions.items():
        if bc.is_inlet():
            if not bc.velocity or bc.velocity.get_magnitude() is None:
                missing_fields.append(MissingFieldInfo(
                    field=f"boundary_conditions.{name}.velocity",
                    description=f"Inlet '{name}' requires velocity specification",
                    required_for=operation,
                    suggested_value={"type": "fixedValue", "value": [1, 0, 0]},
                ))
    
    # --- Check mesh patches have BCs ---
    if config.mesh and config.mesh.patches:
        mesh_patch_names = {p.name for p in config.mesh.patches}
        bc_patch_names = set(config.boundary_conditions.keys())
        
        uncovered = mesh_patch_names - bc_patch_names
        if uncovered:
            for patch_name in uncovered:
                # Skip common auto-generated patches
                if patch_name.lower() in ["defaultfaces", "frontandback"]:
                    continue
                missing_fields.append(MissingFieldInfo(
                    field=f"boundary_conditions.{patch_name}",
                    description=f"Mesh patch '{patch_name}' has no boundary condition",
                    required_for=operation,
                ))
    
    # --- Validate for CFD_CODEGEN_RUN specifically ---
    is_codegen = operation == "CFD_CODEGEN_RUN"
    
    # For codegen, we need more complete info
    if is_codegen:
        # Need characteristic length for Reynolds
        if not config.get_characteristic_length():
            warnings.append("Cannot calculate Reynolds number: no characteristic length available")
        
        # Need velocity for Reynolds
        if not config.get_inlet_velocity_magnitude():
            warnings.append("Cannot calculate Reynolds number: no inlet velocity specified")
    
    # Determine validity
    has_errors = len([f for f in missing_fields if f.required_for == operation]) > 0
    is_complete = len(missing_fields) == 0
    
    # For lint, we're more lenient
    is_valid = not has_errors if is_codegen else True
    
    return ConfigValidationResult(
        is_valid=is_valid,
        is_complete=is_complete,
        normalized_config=config,
        missing_fields=missing_fields,
        warnings=warnings,
        errors=errors,
    )


def get_config_summary(config: SimulationConfigV1) -> dict[str, Any]:
    """Get a summary of the configuration for logging/events."""
    bc_summary = {}
    for name, bc in config.boundary_conditions.items():
        bc_info = {"type": str(bc.patch_type.value if isinstance(bc.patch_type, BoundaryType) else bc.patch_type)}
        if bc.velocity:
            bc_info["has_velocity"] = True
            bc_info["velocity_magnitude"] = bc.velocity.get_magnitude()
        if bc.pressure:
            bc_info["has_pressure"] = True
        if bc.temperature:
            bc_info["has_temperature"] = True
        bc_summary[name] = bc_info
    
    # Handle flow_regime being None (auto-detected later)
    flow_regime_value = None
    if config.physics.flow_regime:
        flow_regime_value = config.physics.flow_regime.value
    
    return {
        "has_mesh": config.mesh is not None,
        "mesh_id": config.mesh.mesh_id if config.mesh else None,
        "mesh_patches": config.mesh.get_patch_names() if config.mesh else [],
        "physics": {
            "flow_regime": flow_regime_value,
            "time_scheme": config.physics.time_scheme.value,
            "heat_transfer": config.physics.heat_transfer,
            "turbulence_model": config.physics.turbulence_model,
        },
        "solver": config.solver.type,
        "fluid": config.fluid.name,
        "boundary_conditions": bc_summary,
        "num_kpi_targets": len(config.kpi_targets),
    }
