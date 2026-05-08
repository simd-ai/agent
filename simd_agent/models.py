# simd_agent/models.py
"""Pydantic models for WebSocket protocol and domain objects."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# --- Enums ---

class Operation(str, Enum):
    """Supported operations."""
    CFD_LINT = "CFD_LINT"
    CFD_CODEGEN_RUN = "CFD_CODEGEN_RUN"
    CFD_RESUBMIT = "CFD_RESUBMIT"


class EventLevel(str, Enum):
    """Event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class RunStatus(str, Enum):
    """Overall run status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NOT_CLEAR = "not_clear"
    CONFIG_INCOMPLETE = "config_incomplete"


class FlowRegime(str, Enum):
    """CFD flow regime classification."""
    LAMINAR = "laminar"
    TRANSITIONAL = "transitional"
    TURBULENT = "turbulent"


class TimeScheme(str, Enum):
    """Time discretization scheme."""
    STEADY = "steady"
    TRANSIENT = "transient"


class Compressibility(str, Enum):
    """Compressibility type."""
    INCOMPRESSIBLE = "incompressible"
    COMPRESSIBLE = "compressible"


class BoundaryType(str, Enum):
    """Boundary condition type classification."""
    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    SYMMETRY = "symmetry"
    PERIODIC = "periodic"
    EMPTY = "empty"
    WEDGE = "wedge"



# =============================================================================
# SIMULATION CONFIG V1 - Canonical Schema
# =============================================================================

class CheckMeshInfoV1(BaseModel):
    """Mesh quality information from OpenFOAM checkMesh."""
    cells: int = Field(..., gt=0, description="Total number of cells")
    faces: int = Field(..., gt=0, description="Total number of faces")
    points: int = Field(..., gt=0, description="Total number of points")
    bounding_box: dict[str, list[float]] | None = Field(
        default=None,
        description="Bounding box as {min: [x,y,z], max: [x,y,z]}"
    )
    characteristic_length: float | None = Field(
        default=None,
        gt=0,
        description="Characteristic length scale (m)"
    )
    # ── Quality metrics from OpenFOAM checkMesh ──────────────────────────
    max_non_orthogonality: float | None = Field(default=None, description="Max face non-orthogonality (degrees)")
    avg_non_orthogonality: float | None = Field(default=None, description="Average face non-orthogonality (degrees)")
    max_aspect_ratio: float | None = Field(default=None, description="Max cell aspect ratio")
    max_skewness: float | None = Field(default=None, description="Max cell skewness")
    n_severe_non_ortho: int | None = Field(default=None, description="Faces with non-orthogonality > 70 degrees")
    mesh_ok: bool | None = Field(default=None, description="True if checkMesh reports no errors")

    model_config = ConfigDict(populate_by_name=True)


class MeshPatchV1(BaseModel):
    """Mesh patch/boundary definition."""
    name: str = Field(..., min_length=1, description="Patch name")
    type: str = Field(..., description="OpenFOAM patch type (patch, wall, empty, etc.)")
    n_faces: int = Field(default=0, ge=0, description="Number of faces in patch")
    
    model_config = ConfigDict(populate_by_name=True)


class MeshInfoV1(BaseModel):
    """Complete mesh information."""
    mesh_id: str = Field(..., min_length=1, description="Mesh identifier/reference")
    file_name: str | None = Field(default=None, description="Original file name")
    patches: list[MeshPatchV1] = Field(default_factory=list, description="Mesh patches/boundaries")
    check_mesh: CheckMeshInfoV1 | None = Field(default=None, description="checkMesh output")
    
    model_config = ConfigDict(populate_by_name=True)
    
    def get_patch_names(self) -> list[str]:
        """Get list of patch names."""
        return [p.name for p in self.patches]
    


class PhysicsV1(BaseModel):
    """Physics settings for simulation."""
    # flow_regime defaults to None - will be auto-detected from Reynolds number
    flow_regime: FlowRegime | None = Field(default=None, description="Flow regime (auto-detected from Reynolds if not specified)")
    time_scheme: TimeScheme = Field(default=TimeScheme.STEADY, description="Time scheme")
    compressibility: Compressibility = Field(
        default=Compressibility.INCOMPRESSIBLE,
        description="Compressibility treatment"
    )
    heat_transfer: bool = Field(default=False, description="Enable heat transfer")
    gravity: bool = Field(default=False, description="Enable gravity / buoyancy effects")
    multiphase: bool = Field(default=False, description="Multiphase simulation")
    phases: list[str] = Field(default_factory=list, description="Phase names for multiphase (e.g. ['water','air'])")
    turbulence_model: str | None = Field(
        default=None,
        description="Turbulence model (kEpsilon, kOmegaSST, laminar, etc.)"
    )
    
    model_config = ConfigDict(populate_by_name=True)


class SolverV1(BaseModel):
    """Solver settings.

    The frontend sends camelCase keys (endTime, deltaT, maxIterations) while the
    rest of the codebase uses snake_case.  Each field has a camelCase alias;
    populate_by_name=True means the Python attribute name (snake_case) also
    works for validation (useful for internal construction and tests).
    """
    type: str = Field(default="simpleFoam", description="OpenFOAM solver application")
    max_iterations: int = Field(
        default=1000, gt=0, alias="maxIterations",
        description="Maximum iterations",
    )
    convergence_criteria: float = Field(
        default=1e-6, gt=0, alias="convergenceCriteria",
        description="Convergence residual",
    )
    end_time: float | None = Field(
        default=None, gt=0, alias="endTime",
        description="End time — physical seconds for transient, iteration count for steady",
    )
    delta_t: float | None = Field(
        default=None, gt=0, alias="deltaT",
        description="Time step (transient)",
    )
    write_interval: int = Field(
        default=100, gt=0, alias="writeInterval",
        description="Write interval",
    )

    model_config = ConfigDict(populate_by_name=True)


class FluidV1(BaseModel):
    """Fluid properties.
    
    Note: Validation of positive values is done in the linting phase,
    not during model construction, to allow lint issues to be reported.
    """
    name: str = Field(default="air", description="Fluid name (air, water, custom)")
    density: float = Field(default=1.225, description="Density (kg/m³)")
    kinematic_viscosity: float = Field(
        default=1.5e-5,
        description="Kinematic viscosity (m²/s)"
    )
    dynamic_viscosity: float | None = Field(
        default=None,
        description="Dynamic viscosity (Pa·s)"
    )
    specific_heat: float | None = Field(default=None, description="Specific heat Cp (J/kg·K)")
    thermal_conductivity: float | None = Field(
        default=None,
        description="Thermal conductivity (W/m·K)"
    )
    thermal_diffusivity: float | None = Field(
        default=None,
        description="Thermal diffusivity α = k/(ρ·Cp) (m²/s)"
    )
    prandtl_number: float | None = Field(
        default=None, alias="prandtl",
        description="Prandtl number (also accepted as 'prandtl')",
    )
    temperature: float | None = Field(
        default=None,
        description="Reference / bulk temperature of the fluid (K or °C as provided)"
    )

    model_config = ConfigDict(populate_by_name=True)


class TurbulenceConfigV1(BaseModel):
    """Top-level turbulence configuration block.

    Carries the turbulence model choice AND the pre-computed initial field
    values (k, omega, epsilon, nut) so the backend never has to guess them.
    """
    model: str = Field(default="kOmegaSST", description="Turbulence model name")
    intensity: float | None = Field(default=None, description="Turbulence intensity I [%]")
    length_scale: float | None = Field(
        default=None, alias="lengthScale",
        description="Turbulence length scale L [m]",
    )
    hydraulic_diameter: float | None = Field(
        default=None, alias="hydraulicDiameter",
        description="Hydraulic diameter Dh [m]",
    )
    k: float | None = Field(default=None, description="Turbulent kinetic energy k [m²/s²]")
    omega: float | None = Field(default=None, description="Specific dissipation rate ω [1/s]")
    epsilon: float | None = Field(default=None, description="Turbulent dissipation rate ε [m²/s³]")
    nut: float | None = Field(default=None, description="Turbulent kinematic viscosity νt [m²/s]")
    wall_functions: bool = Field(
        default=True, alias="wallFunctions",
        description="Use wall functions for turbulence BCs",
    )

    model_config = ConfigDict(populate_by_name=True)


class GeometryV1(BaseModel):
    """Geometry description.
    
    Note: Validation of positive values is done in the linting phase,
    not during model construction, to allow lint issues to be reported.
    """
    type: str | None = Field(default=None, description="Geometry type (pipe, airfoil, etc.)")
    diameter: float | None = Field(default=None, description="Diameter (m)")
    length: float | None = Field(default=None, description="Length (m)")
    width: float | None = Field(default=None, description="Width (m)")
    height: float | None = Field(default=None, description="Height (m)")
    radius: float | None = Field(default=None, description="Radius (m)")
    chord: float | None = Field(default=None, description="Chord length (m)")
    
    model_config = ConfigDict(populate_by_name=True)
    
    def get_characteristic_length(self) -> float | None:
        """Get characteristic length for Reynolds number calculation."""
        if self.diameter:
            return self.diameter
        if self.radius:
            return self.radius * 2
        if self.chord:
            return self.chord
        if self.length:
            return self.length
        return None


class VelocityBCV1(BaseModel):
    """Velocity boundary condition."""
    type: str = Field(default="fixedValue", description="BC type")
    value: list[float] | float | None = Field(
        default=None,
        description="Velocity value [Ux, Uy, Uz] or magnitude"
    )
    magnitude: float | None = Field(default=None, description="Velocity magnitude (m/s)")
    direction: list[float] | None = Field(default=None, description="Flow direction [x, y, z]")
    # flowRateInletVelocity — Option-1 (mutually exclusive with volumetric_flow_rate)
    mass_flow_rate: float | None = Field(
        default=None, alias="massFlowRate",
        description="Mass flow rate [kg/s] for flowRateInletVelocity"
    )
    # flowRateInletVelocity — Option-2
    volumetric_flow_rate: float | None = Field(
        default=None, alias="volumetricFlowRate",
        description="Volumetric flow rate [m³/s] for flowRateInletVelocity"
    )
    # flowRateInletVelocity — optional: name of density field (default 'rho')
    rho_field: str | None = Field(
        default=None, alias="rho",
        description="Name of density field used to convert mass→volumetric flow rate"
    )
    # flowRateInletVelocity — optional: density initialisation value when the density
    # field is not yet available (e.g. iteration 0).  Required for compressible solvers.
    rho_inlet: float | None = Field(
        default=None, alias="rhoInlet",
        description="Density initialisation value [kg/m³] for compressible flowRateInletVelocity"
    )
    # flowRateInletVelocity — optional: extrapolate velocity profile from interior cells
    extrapolate_profile: bool | None = Field(
        default=None, alias="extrapolateProfile",
        description="Extrapolate velocity profile from interior (default false = plug flow)"
    )

    model_config = ConfigDict(populate_by_name=True)

    def is_flow_rate_inlet(self) -> bool:
        return self.type == "flowRateInletVelocity"

    def get_flow_rate(self) -> tuple[str, float] | None:
        """Return (key_name, value) for flow-rate inlets, or None."""
        if self.mass_flow_rate is not None:
            return ("massFlowRate", self.mass_flow_rate)
        if self.volumetric_flow_rate is not None:
            return ("volumetricFlowRate", self.volumetric_flow_rate)
        if self.is_flow_rate_inlet() and isinstance(self.value, (int, float)):
            return ("massFlowRate", float(self.value))
        return None

    def get_velocity_vector(self) -> list[float] | None:
        """Get velocity as [Ux, Uy, Uz] vector."""
        if isinstance(self.value, list) and len(self.value) == 3:
            return self.value
        if self.magnitude is not None and self.direction:
            import math
            d = self.direction
            norm = math.sqrt(sum(x**2 for x in d))
            if norm > 0:
                return [self.magnitude * x / norm for x in d]
        # For flow-rate inlets the scalar value is the flow rate, NOT a velocity.
        if self.is_flow_rate_inlet():
            return None
        if isinstance(self.value, (int, float)):
            return [float(self.value), 0.0, 0.0]
        return None
    
    def get_magnitude(self) -> float | None:
        """Get velocity magnitude."""
        if self.magnitude is not None:
            return self.magnitude
        if isinstance(self.value, (int, float)):
            return abs(float(self.value))
        if isinstance(self.value, list):
            import math
            return math.sqrt(sum(x**2 for x in self.value))
        return None


class PressureBCV1(BaseModel):
    """Pressure boundary condition."""
    type: str = Field(default="fixedValue", description="BC type")
    value: float | None = Field(default=None, description="Pressure value (Pa)")
    
    model_config = ConfigDict(populate_by_name=True)


class TemperatureBCV1(BaseModel):
    """Temperature boundary condition."""
    type: str = Field(default="fixedValue", description="BC type")
    value: float | None = Field(default=None, description="Temperature value (K)")
    
    model_config = ConfigDict(populate_by_name=True)


class TurbulenceBCV1(BaseModel):
    """Turbulence boundary condition."""
    k: dict[str, Any] | None = Field(default=None, description="Turbulent kinetic energy BC")
    epsilon: dict[str, Any] | None = Field(default=None, description="Turbulent dissipation BC")
    omega: dict[str, Any] | None = Field(default=None, description="Specific dissipation BC")
    nut: dict[str, Any] | None = Field(default=None, description="Turbulent viscosity BC")
    
    model_config = ConfigDict(populate_by_name=True)


class BoundaryConditionV1(BaseModel):
    """Complete boundary condition for a single patch."""
    patch_type: BoundaryType | str = Field(
        default=BoundaryType.WALL,
        description="Boundary type classification"
    )
    velocity: VelocityBCV1 | None = Field(default=None, description="Velocity BC")
    pressure: PressureBCV1 | None = Field(default=None, description="Pressure BC")
    temperature: TemperatureBCV1 | None = Field(default=None, description="Temperature BC")
    turbulence: TurbulenceBCV1 | None = Field(default=None, description="Turbulence BCs (nested)")

    # Turbulence fields at top level (frontend sends them here directly)
    k: dict[str, Any] | None = Field(default=None, description="k BC")
    epsilon: dict[str, Any] | None = Field(default=None, description="epsilon BC")
    omega: dict[str, Any] | None = Field(default=None, description="omega BC")
    nut: dict[str, Any] | None = Field(default=None, description="nut BC")
    alphat: dict[str, Any] | None = Field(default=None, description="alphat BC")
    nuTilda: dict[str, Any] | None = Field(default=None, description="nuTilda BC (Spalart-Allmaras)")

    model_config = ConfigDict(populate_by_name=True)
    
    def is_inlet(self) -> bool:
        """Check if this is an inlet-type BC."""
        if isinstance(self.patch_type, BoundaryType):
            return self.patch_type == BoundaryType.INLET
        return str(self.patch_type).lower() == "inlet"
    
    def is_outlet(self) -> bool:
        """Check if this is an outlet-type BC."""
        if isinstance(self.patch_type, BoundaryType):
            return self.patch_type == BoundaryType.OUTLET
        return str(self.patch_type).lower() == "outlet"
    
    def is_wall(self) -> bool:
        """Check if this is a wall-type BC."""
        if isinstance(self.patch_type, BoundaryType):
            return self.patch_type == BoundaryType.WALL
        return str(self.patch_type).lower() == "wall"


class KPITargetV1(BaseModel):
    """A KPI target value."""
    name: str = Field(..., description="KPI name")
    value: float = Field(..., description="Target value")
    unit: str = Field(default="", description="Unit of measurement")
    tolerance: float | None = Field(default=None, description="Acceptable tolerance")
    
    model_config = ConfigDict(populate_by_name=True)


class SimulationConfigV1(BaseModel):
    """
    Canonical V1 schema for simulation configuration.
    
    This is the normalized form used internally. The normalizer converts
    various input formats (legacy, camelCase, etc.) to this schema.
    """
    # Core components (all optional for backward compat, validated at runtime)
    mesh: MeshInfoV1 | None = Field(default=None, description="Mesh information")
    physics: PhysicsV1 = Field(default_factory=PhysicsV1, description="Physics settings")
    solver: SolverV1 = Field(default_factory=SolverV1, description="Solver settings")
    fluid: FluidV1 = Field(default_factory=FluidV1, description="Fluid properties")
    turbulence: TurbulenceConfigV1 | None = Field(
        default=None,
        description="Turbulence model + pre-computed initial field values (k, ω, ε, νt)"
    )
    geometry: GeometryV1 | None = Field(default=None, description="Geometry description")
    
    # Boundary conditions keyed by patch name
    boundary_conditions: dict[str, BoundaryConditionV1] = Field(
        default_factory=dict,
        description="Boundary conditions per patch"
    )
    
    # Optional KPI targets
    kpi_targets: list[KPITargetV1] = Field(
        default_factory=list,
        description="KPI targets to achieve"
    )
    
    # Legacy/passthrough fields for backward compatibility
    case_type: str | None = Field(default=None, description="Detected case type")
    inlet: dict[str, Any] | None = Field(default=None, description="Legacy inlet config")
    outlet: dict[str, Any] | None = Field(default=None, description="Legacy outlet config")
    
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    
    def get_inlet_velocity_magnitude(self) -> float | None:
        """Get inlet velocity magnitude from boundary conditions or legacy inlet."""
        # Check boundary conditions first
        for name, bc in self.boundary_conditions.items():
            if bc.is_inlet() and bc.velocity:
                mag = bc.velocity.get_magnitude()
                if mag is not None:
                    return mag
        
        # Fall back to legacy inlet
        if self.inlet:
            vel = self.inlet.get("velocity")
            if isinstance(vel, (int, float)):
                return abs(float(vel))
            if isinstance(vel, list):
                import math
                return math.sqrt(sum(x**2 for x in vel))
        
        return None
    
    def get_characteristic_length(self) -> float | None:
        """Get characteristic length for Reynolds number."""
        # Try geometry first
        if self.geometry:
            length = self.geometry.get_characteristic_length()
            if length:
                return length
        
        # Try mesh bounding box
        if self.mesh and self.mesh.check_mesh:
            if self.mesh.check_mesh.characteristic_length:
                return self.mesh.check_mesh.characteristic_length
            if self.mesh.check_mesh.bounding_box:
                bb = self.mesh.check_mesh.bounding_box
                if "min" in bb and "max" in bb:
                    # Use max dimension
                    dims = [bb["max"][i] - bb["min"][i] for i in range(3)]
                    return max(dims)
        
        return None
    
    def get_kinematic_viscosity(self) -> float:
        """Get kinematic viscosity."""
        return self.fluid.kinematic_viscosity
    
    def get_patches_without_bc(self) -> list[str]:
        """Get list of mesh patches that don't have boundary conditions."""
        if not self.mesh:
            return []
        
        mesh_patches = set(p.name for p in self.mesh.patches)
        bc_patches = set(self.boundary_conditions.keys())
        
        return list(mesh_patches - bc_patches)


class MissingFieldInfo(BaseModel):
    """Information about a missing required field."""
    field: str = Field(..., description="Field path (e.g., 'boundary_conditions.inlet')")
    description: str = Field(..., description="What this field should contain")
    required_for: str = Field(
        default="codegen",
        description="Which operation requires this field"
    )
    suggested_value: Any = Field(default=None, description="Suggested default value")


class ConfigValidationResult(BaseModel):
    """Result of configuration validation."""
    is_valid: bool = Field(..., description="Whether config is valid for the operation")
    is_complete: bool = Field(..., description="Whether all required fields are present")
    normalized_config: SimulationConfigV1 | None = Field(
        default=None,
        description="Normalized configuration"
    )
    missing_fields: list[MissingFieldInfo] = Field(
        default_factory=list,
        description="List of missing required fields"
    )
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings")
    errors: list[str] = Field(default_factory=list, description="Fatal errors")


# --- Client -> Server ---

class Constraints(BaseModel):
    """Constraints for the run."""
    max_retries: int = Field(default=7, ge=1, le=10)
    solver_preference: str | None = None
    mesh_preference: str | None = None
    timeout_seconds: int = Field(default=21600, ge=30, le=21600)


class Metadata(BaseModel):
    """Optional metadata for tracking."""
    user_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    parent_run_id: str | None = None  # For CFD_RESUBMIT: the run whose files to reuse
    refine_strategy: str | None = None  # Mesh refinement: "wall", "global", or None
    tags: list[str] = Field(default_factory=list)


class StartRequest(BaseModel):
    """Initial WebSocket message from client to start a run."""
    op: Operation
    provider: str = Field(default="gemini")
    prompt_pack: str = Field(default="simd")
    user_requirements: str = Field(
        ...,
        min_length=1,
        description="Natural language description of what the user wants to simulate",
    )
    simulation_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Partial or complete simulation configuration",
    )
    constraints: Constraints = Field(default_factory=Constraints)
    metadata: Metadata = Field(default_factory=Metadata)


# --- Server -> Client ---

class AgentEvent(BaseModel):
    """Event streamed from server to client via WebSocket."""
    run_id: UUID
    seq: int = Field(ge=0, description="Monotonically increasing sequence number")
    ts: datetime = Field(default_factory=datetime.utcnow)
    level: EventLevel = EventLevel.INFO
    type: str = Field(
        ...,
        description="Event type identifier",
    )
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    
    def to_ws_message(self) -> dict[str, Any]:
        """Serialize for WebSocket transmission."""
        return {
            "run_id": str(self.run_id),
            "seq": self.seq,
            "ts": self.ts.isoformat(),
            "level": self.level.value,
            "type": self.type,
            "message": self.message,
            "payload": self.payload,
        }


# --- Event Types (as constants for type safety) ---

class EventTypes:
    """Standard event type identifiers."""
    # Lifecycle
    RUN_STARTED = "run_started"
    RUN_SUCCEEDED = "run_succeeded"
    RUN_FAILED = "run_failed"
    # Emitted when the user explicitly cancels a validation or run.
    # The frontend should restore the "Validate Setup" / "Run Simulation" button
    # and reset the events panel so the user can start again cleanly.
    RUN_CANCELLED = "run_cancelled"
    SIMULATION_NOT_CLEAR = "simulation_not_clear"
    
    # Config validation
    CONFIG_RECEIVED = "config_received"
    CONFIG_INCOMPLETE = "config_incomplete"
    CONFIG_NORMALIZED = "config_normalized"
    # Emitted after lint + solver selection — carries the final split config the
    # frontend should persist to the `simulation_config` Neon table so that the
    # chat service can read cfd_physics / cfd_solver / cfd_fluid / cfd_turbulence.
    SIMULATION_CONFIG_READY = "simulation_config_ready"
    
    # Linting
    LINT_STARTED = "lint_started"
    LINT_RESULT = "lint_result"
    
    # Planning
    PLANNING_STARTED = "planning_started"
    PLANNING_COMPLETE = "planning_complete"
    SUBAGENT_STARTED = "subagent_started"
    SUBAGENT_UPDATE = "subagent_update"
    SUBAGENT_COMPLETE = "subagent_complete"
    
    # Codegen
    CODEGEN_STARTED = "codegen_started"
    CODEGEN_ITERATION = "codegen_iteration"
    CODEGEN_COMPLETE = "codegen_complete"
    # Per-file streaming events
    FILE_GENERATING = "file_generating"   # emitted when an individual file LLM call starts
    FILE_GENERATED = "file_generated"     # emitted when an individual file is ready (with content)
    
    # Simulation Server
    SIM_SUBMITTED = "sim_submitted"
    SIM_EXTRACT_STARTED = "sim_extract_started"
    SIM_EXTRACT_COMPLETE = "sim_extract_complete"
    SIM_MESH_CONVERSION_STARTED = "sim_mesh_conversion_started"
    SIM_MESH_CONVERSION_COMPLETE = "sim_mesh_conversion_complete"
    SIM_MESH_CONVERSION_FAILED = "sim_mesh_conversion_failed"
    SIM_BLOCKMESH_STARTED = "sim_blockmesh_started"
    SIM_BLOCKMESH_COMPLETE = "sim_blockmesh_complete"
    SIM_CHECKMESH_STARTED = "sim_checkmesh_started"
    SIM_CHECKMESH_COMPLETE = "sim_checkmesh_complete"
    SIM_RUN_STARTED = "sim_run_started"
    SIM_RUN_PROGRESS = "sim_run_progress"
    SIM_RUN_LOG = "sim_run_log"
    SIM_RUN_SUCCEEDED = "sim_run_succeeded"
    SIM_RUN_FAILED = "sim_run_failed"
    SIM_ARTIFACTS_READY = "sim_artifacts_ready"
    # MPI parallel decompose / reconstruct
    SIM_DECOMPOSE_STARTED = "sim_decompose_started"
    SIM_DECOMPOSE_COMPLETE = "sim_decompose_complete"
    SIM_DECOMPOSE_FAILED = "sim_decompose_failed"
    SIM_RECONSTRUCT_STARTED = "sim_reconstruct_started"
    SIM_RECONSTRUCT_COMPLETE = "sim_reconstruct_complete"
    SIM_RECONSTRUCT_FAILED = "sim_reconstruct_failed"
    
    # Code verification (super-model quality gate)
    SOLVER_SELECTION_STARTED = "solver_selection_started"
    SOLVER_SELECTED = "solver_selected"
    CODEGEN_VERIFICATION_STARTED = "codegen_verification_started"
    CODEGEN_VERIFICATION_COMPLETE = "codegen_verification_complete"

    # LLM thinking indicator (frontend shows shimmer while LLM is working)
    THINKING_STARTED = "thinking_started"
    THINKING_COMPLETE = "thinking_complete"

    # Self-healing
    DIAGNOSING = "diagnosing"
    ERROR_SUMMARY = "error_summary"
    RETRYING = "retrying"
    SIM_PROGRESS_RESET = "sim_progress_reset"  # emitted before retry; tells frontend to clear residuals
    
    # Final — always carries op ("CFD_LINT" | "CFD_CODEGEN_RUN") so the frontend
    # can route the result to the correct section (validation vs simulation).
    FINAL = "final"


# --- Linting Models ---

class ApplyChange(BaseModel):
    """A recommended change to the simulation config."""
    path: str = Field(..., description="Dot-path or key to the config field")
    value: Any = Field(..., description="Recommended value")
    reason: str = Field(..., description="Why this change is recommended")
    severity: Literal["info", "warning", "error"] = "info"


class LintIssue(BaseModel):
    """A validation issue found during linting."""
    code: str = Field(..., description="Issue code (e.g., 'INVALID_UNITS')")
    path: str | None = Field(None, description="Config path where issue was found")
    message: str
    severity: Literal["warning", "error"]


class LintResult(BaseModel):
    """Result of CFD linting."""
    validated_config: dict[str, Any]
    normalized_config: SimulationConfigV1 | None = Field(
        default=None,
        description="Fully normalized V1 config"
    )
    apply_changes: list[ApplyChange] = Field(default_factory=list)
    issues: list[LintIssue] = Field(default_factory=list)
    missing_fields: list[MissingFieldInfo] = Field(
        default_factory=list,
        description="Required fields that are missing"
    )
    detected_case_type: str | None = None
    detected_regime: FlowRegime | None = None
    selected_solver: str | None = None
    reynolds_number: float | None = None
    is_complete: bool = Field(
        default=False,
        description="Whether config is complete for codegen"
    )


# --- Planning Models ---

class WorkItem(BaseModel):
    """A unit of work for parallel sub-agents."""
    id: str
    task: str = Field(..., description="Task identifier (e.g., 'choose_solver')")
    description: str
    priority: int = Field(default=1, ge=1, le=10)
    dependencies: list[str] = Field(default_factory=list)


class SubAgentResult(BaseModel):
    """Result from a sub-agent task."""
    work_item_id: str
    task: str
    result: dict[str, Any]
    duration_ms: int


class PlanningResult(BaseModel):
    """Result of the planning phase."""
    work_items: list[WorkItem]
    case_type: str
    regime: FlowRegime | None = None
    solver: str
    turbulence_model: str | None = None
    mesh_strategy: str
    sub_results: list[SubAgentResult] = Field(default_factory=list)


# --- Error Summary Models ---

class ErrorSummary(BaseModel):
    """Summary of simulation execution error."""
    root_cause: str
    actionable_changes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of changes to apply to fix the error",
    )
    affected_files: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# --- Final Result Models ---

class FinalResult(BaseModel):
    """Final result payload for the 'final' event."""
    status: RunStatus
    validated_config: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    retries: int = 0
    summary: str = ""
    case_type: str | None = None
    solver: str | None = None
    error: str | None = None


# --- Database Row Models ---

class RunRow(BaseModel):
    """Database row for a run."""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    op: Operation
    status: RunStatus = RunStatus.PENDING
    provider: str
    prompt_pack: str
    user_requirements: str
    simulation_config: dict[str, Any]
    validated_config: dict[str, Any] | None = None
    attempts: int = 0
    result: dict[str, Any] | None = None


class EventRow(BaseModel):
    """Database row for an event."""
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    seq: int
    ts: datetime
    level: EventLevel
    type: str
    message: str
    payload: dict[str, Any]
