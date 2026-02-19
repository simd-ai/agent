# simd_agent/precheck.py
"""Precheck service for analyzing user prompts and extracting simulation specifications.

Uses Google GenAI with structured output (response_schema) for reliable JSON parsing.
"""

import json
import logging
import math
import re
from typing import Any, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from simd_agent.settings import get_settings

load_dotenv()

logger = logging.getLogger(__name__)


# --- Request Models ---

class MeshPatch(BaseModel):
    """Mesh patch information."""
    name: str
    type: str  # e.g., "wall", "patch", "empty"
    n_cells: int = Field(alias="nCells", default=0)

    class Config:
        populate_by_name = True


class CheckMeshInfo(BaseModel):
    """checkMesh output information."""
    cells: int
    faces: int
    points: int
    bounding_box: dict[str, list[float]] | None = Field(alias="boundingBox", default=None)
    characteristic_length: float | None = Field(alias="characteristicLength", default=None)

    class Config:
        populate_by_name = True


class MeshInfo(BaseModel):
    """Uploaded mesh information."""
    mesh_id: str = Field(alias="meshId")
    file_name: str = Field(alias="fileName")
    patches: list[MeshPatch]
    check_mesh: CheckMeshInfo = Field(alias="checkMesh")

    class Config:
        populate_by_name = True


class PrecheckRequest(BaseModel):
    """Request for precheck analysis."""
    prompt: str = Field(..., min_length=1, description="Natural language simulation description")
    has_mesh: bool = Field(default=False, alias="hasMesh")
    mesh_info: MeshInfo | None = Field(default=None, alias="meshInfo")
    # Legacy support for old format
    mesh: MeshInfo | None = None
    previous_config: dict[str, Any] | None = Field(default=None, alias="previousConfig")

    class Config:
        populate_by_name = True

    def get_mesh(self) -> MeshInfo | None:
        """Get mesh info from either field."""
        return self.mesh_info or self.mesh


# --- Response Models ---

class SolverSettings(BaseModel):
    """Solver configuration."""
    algorithm: Literal["SIMPLE", "PIMPLE", "PISO"] = "SIMPLE"
    max_iterations: int = Field(default=2000, alias="maxIterations")
    convergence_criteria: float = Field(default=1e-6, alias="convergenceCriteria")
    end_time: float | None = Field(default=None, alias="endTime")  # For transient
    delta_t: float | None = Field(default=None, alias="deltaT")  # For transient
    write_interval: float | None = Field(default=None, alias="writeInterval")

    class Config:
        populate_by_name = True


class FluidProperties(BaseModel):
    """Fluid/material properties."""
    preset_id: str = Field(default="air", alias="presetId")  # water, air, custom
    name: str = "Air"
    rho: float = 1.225  # density [kg/m³]
    mu: float = 1.81e-5  # dynamic viscosity [Pa·s]
    Cp: float = 1006.0  # specific heat [J/(kg·K)]
    k: float = 0.0257  # thermal conductivity [W/(m·K)]
    temperature: float = 293.15  # reference temp [K]

    class Config:
        populate_by_name = True


class TurbulenceSettings(BaseModel):
    """Turbulence model parameters."""
    model: Literal["kEpsilon", "kOmegaSST", "spalartAllmaras", "laminar"] = "kOmegaSST"
    turbulence_intensity: float = Field(default=5.0, alias="turbulenceIntensity")  # I [%]
    turbulence_length_scale: float = Field(default=0.01, alias="turbulenceLengthScale")  # L [m]
    hydraulic_diameter: float = Field(default=0.1, alias="hydraulicDiameter")  # Dh [m]
    wall_functions: bool = Field(default=True, alias="wallFunctions")

    class Config:
        populate_by_name = True


class FieldBC(BaseModel):
    """Single field boundary condition (OpenFOAM-style)."""
    type: str  # fixedValue, zeroGradient, noSlip, etc.
    value: float | list[float] | None = None


class PatchBoundaryCondition(BaseModel):
    """Per-patch, per-field boundary conditions."""
    patch_class: Literal["inlet", "outlet", "wall", "symmetry", "periodic", "empty"] = Field(
        alias="patchClass"
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    # Velocity field
    U: FieldBC | None = None
    # Pressure field
    p: FieldBC | None = None
    # Temperature (if heat transfer enabled)
    T: FieldBC | None = None
    # Turbulence fields (k-epsilon)
    k: FieldBC | None = None
    epsilon: FieldBC | None = None
    # Turbulence fields (k-omega)
    omega: FieldBC | None = None
    # Turbulent viscosity
    nut: FieldBC | None = None

    class Config:
        populate_by_name = True


class SuggestedConfig(BaseModel):
    """Suggested simulation configuration - complete setup for frontend."""
    # Core physics
    case_type: str = Field(default="internal_flow", alias="caseType")  # internal_pipe_flow, external_aero, etc.
    flow_regime: Literal["laminar", "turbulent"] = Field(alias="flowRegime")
    time_scheme: Literal["steady", "transient"] = Field(alias="timeScheme")
    compressibility: Literal["incompressible", "compressible"] = "incompressible"
    enable_heat_transfer: bool = Field(default=False, alias="enableHeatTransfer")
    gravity: bool = False

    # Solver settings
    solver: SolverSettings

    # Fluid properties
    fluid: FluidProperties

    # Turbulence settings
    turbulence: TurbulenceSettings

    # Per-patch boundary conditions (OpenFOAM-style)
    boundary_conditions: dict[str, PatchBoundaryCondition] = Field(
        default_factory=dict, alias="boundaryConditions"
    )

    class Config:
        populate_by_name = True


# Legacy boundary hint (still supported for backward compat)
class VelocityBC(BaseModel):
    """Velocity boundary condition."""
    type: str
    value: list[float] | None = None
    magnitude: float | None = None


class PressureBC(BaseModel):
    """Pressure boundary condition."""
    type: str
    value: float | None = None


class TemperatureBC(BaseModel):
    """Temperature boundary condition."""
    type: str
    value: float | None = None


class BoundaryHint(BaseModel):
    """Suggested boundary condition for a patch (legacy format)."""
    suggested_type: Literal["inlet", "outlet", "wall", "symmetry", "periodic"] = Field(
        alias="suggestedType"
    )
    velocity: VelocityBC | None = None
    pressure: PressureBC | None = None
    temperature: TemperatureBC | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""

    class Config:
        populate_by_name = True


class KPIValue(BaseModel):
    """A KPI target value."""
    value: float
    unit: str


class CustomKPI(BaseModel):
    """Custom KPI target."""
    name: str
    value: float
    unit: str


class KPITargets(BaseModel):
    """KPI targets extracted from prompt."""
    pressure_drop: KPIValue | None = Field(default=None, alias="pressureDrop")
    flow_rate: KPIValue | None = Field(default=None, alias="flowRate")
    temperature: KPIValue | None = None
    velocity: KPIValue | None = None
    custom: list[CustomKPI] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class Interpretation(BaseModel):
    """LLM's understanding of the prompt."""
    summary: str
    simulation_type: str = Field(alias="simulationType")  # e.g., "Internal pipe flow"
    key_physics: list[str] = Field(default_factory=list, alias="keyPhysics")
    assumptions: list[str] = Field(default_factory=list)
    clarifications: list[str] | None = None

    class Config:
        populate_by_name = True


class ConfidenceScores(BaseModel):
    """Confidence scores for various aspects."""
    overall: float = Field(default=0.5, ge=0.0, le=1.0)
    flow_regime: float = Field(default=0.5, ge=0.0, le=1.0, alias="flowRegime")
    boundary_conditions: float = Field(default=0.5, ge=0.0, le=1.0, alias="boundaryConditions")
    physics_settings: float = Field(default=0.5, ge=0.0, le=1.0, alias="physicsSettings")

    class Config:
        populate_by_name = True


class PrecheckResponse(BaseModel):
    """Response from precheck analysis."""
    success: bool
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # Top-level confidence
    message: str = ""  # Summary message for frontend
    suggested_config: SuggestedConfig = Field(alias="suggestedConfig")
    # Legacy fields (still returned for backward compat)
    boundary_hints: dict[str, BoundaryHint] | None = Field(default=None, alias="boundaryHints")
    kpi_targets: KPITargets | None = Field(default=None, alias="kpiTargets")
    interpretation: Interpretation
    confidence_scores: ConfidenceScores = Field(alias="confidenceScores")
    next_step: int = Field(default=1, ge=1, le=3, alias="nextStep")
    should_show_mesh_viewer: bool = Field(default=False, alias="shouldShowMeshViewer")
    warnings: list[str] | None = None
    errors: list[str] | None = None

    class Config:
        populate_by_name = True


# --- Fluid Presets ---

FLUID_PRESETS: dict[str, FluidProperties] = {
    "air": FluidProperties(
        preset_id="air",
        name="Air",
        rho=1.225,
        mu=1.81e-5,
        Cp=1006.0,
        k=0.0257,
        temperature=293.15,
    ),
    "water": FluidProperties(
        preset_id="water",
        name="Water",
        rho=998.2,
        mu=1.002e-3,
        Cp=4182.0,
        k=0.598,
        temperature=293.15,
    ),
    "oil": FluidProperties(
        preset_id="oil",
        name="Oil (SAE 30)",
        rho=880.0,
        mu=0.29,
        Cp=1900.0,
        k=0.145,
        temperature=293.15,
    ),
    "ln2": FluidProperties(
        preset_id="ln2",
        name="Liquid Nitrogen (LN2)",
        rho=808.0,  # kg/m³ at 77K
        mu=1.58e-4,  # Pa·s at 77K
        Cp=2042.0,  # J/(kg·K)
        k=0.140,  # W/(m·K) thermal conductivity
        temperature=77.0,  # K (boiling point at 1 atm)
    ),
}


# --- Precheck Service ---

# Model to use for precheck analysis (Gemini 3)
PRECHECK_MODEL = "gemini-3-flash-preview"


# --- Function/Tool Definition for Structured Output ---
# Using function calling instead of response_schema to handle dict[str, ...] types

PRECHECK_TOOL_SCHEMA = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="submit_cfd_configuration",
            description="Submit the analyzed CFD simulation configuration based on user requirements",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "message": types.Schema(
                        type="STRING",
                        description="Summary message of the detected simulation",
                    ),
                    "case_type": types.Schema(
                        type="STRING",
                        description="Type of simulation",
                        enum=["internal_pipe_flow", "external_aero", "heat_exchanger", "mixing", "general"],
                    ),
                    "flow_regime": types.Schema(
                        type="STRING",
                        enum=["laminar", "turbulent"],
                    ),
                    "time_scheme": types.Schema(
                        type="STRING",
                        enum=["steady", "transient"],
                    ),
                    "compressibility": types.Schema(
                        type="STRING",
                        enum=["incompressible", "compressible"],
                    ),
                    "enable_heat_transfer": types.Schema(type="BOOLEAN"),
                    "gravity": types.Schema(type="BOOLEAN"),
                    "solver_algorithm": types.Schema(
                        type="STRING",
                        enum=["SIMPLE", "PIMPLE", "PISO"],
                    ),
                    "solver_max_iterations": types.Schema(type="INTEGER"),
                    "solver_convergence_criteria": types.Schema(type="NUMBER"),
                    "solver_end_time": types.Schema(type="NUMBER", nullable=True),
                    "solver_delta_t": types.Schema(type="NUMBER", nullable=True),
                    "fluid_preset_id": types.Schema(
                        type="STRING",
                        enum=["air", "water", "oil", "ln2", "custom"],
                    ),
                    "fluid_name": types.Schema(type="STRING"),
                    "fluid_rho": types.Schema(type="NUMBER", description="Density kg/m³"),
                    "fluid_mu": types.Schema(type="NUMBER", description="Dynamic viscosity Pa·s"),
                    "fluid_Cp": types.Schema(type="NUMBER", description="Specific heat J/(kg·K)"),
                    "fluid_k": types.Schema(type="NUMBER", description="Thermal conductivity W/(m·K)"),
                    "fluid_temperature": types.Schema(type="NUMBER", description="Reference temperature K"),
                    "turbulence_model": types.Schema(
                        type="STRING",
                        enum=["kEpsilon", "kOmegaSST", "spalartAllmaras", "laminar"],
                    ),
                    "turbulence_intensity": types.Schema(type="NUMBER", description="Percentage (0-100)"),
                    "turbulence_length_scale": types.Schema(type="NUMBER", description="Length scale in meters"),
                    "hydraulic_diameter": types.Schema(type="NUMBER", description="Hydraulic diameter in meters"),
                    "wall_functions": types.Schema(type="BOOLEAN"),
                    "boundary_conditions": types.Schema(
                        type="ARRAY",
                        description="Boundary conditions for each patch",
                        items=types.Schema(
                            type="OBJECT",
                            properties={
                                "patch_name": types.Schema(type="STRING"),
                                "patch_class": types.Schema(
                                    type="STRING",
                                    enum=["inlet", "outlet", "wall", "symmetry", "periodic"],
                                ),
                                "confidence": types.Schema(type="NUMBER"),
                                "U_type": types.Schema(type="STRING"),
                                "U_value": types.Schema(
                                    type="ARRAY",
                                    items=types.Schema(type="NUMBER"),
                                    nullable=True,
                                ),
                                "p_type": types.Schema(type="STRING"),
                                "p_value": types.Schema(type="NUMBER", nullable=True),
                                "T_type": types.Schema(type="STRING", nullable=True),
                                "T_value": types.Schema(type="NUMBER", nullable=True),
                                "k_type": types.Schema(type="STRING", nullable=True),
                                "k_value": types.Schema(type="NUMBER", nullable=True),
                                "omega_type": types.Schema(type="STRING", nullable=True),
                                "omega_value": types.Schema(type="NUMBER", nullable=True),
                                "epsilon_type": types.Schema(type="STRING", nullable=True),
                                "epsilon_value": types.Schema(type="NUMBER", nullable=True),
                                "nut_type": types.Schema(type="STRING", nullable=True),
                                "nut_value": types.Schema(type="NUMBER", nullable=True),
                            },
                            required=["patch_name", "patch_class", "U_type", "p_type"],
                        ),
                    ),
                    "interpretation_summary": types.Schema(type="STRING"),
                    "interpretation_simulation_type": types.Schema(type="STRING"),
                    "interpretation_key_physics": types.Schema(
                        type="ARRAY",
                        items=types.Schema(type="STRING"),
                    ),
                    "interpretation_assumptions": types.Schema(
                        type="ARRAY",
                        items=types.Schema(type="STRING"),
                    ),
                    "confidence_overall": types.Schema(type="NUMBER"),
                    "confidence_flow_regime": types.Schema(type="NUMBER"),
                    "confidence_boundary_conditions": types.Schema(type="NUMBER"),
                    "confidence_physics_settings": types.Schema(type="NUMBER"),
                },
                required=[
                    "message",
                    "case_type",
                    "flow_regime",
                    "time_scheme",
                    "enable_heat_transfer",
                    "fluid_preset_id",
                    "fluid_rho",
                    "fluid_mu",
                    "turbulence_model",
                    "boundary_conditions",
                    "interpretation_summary",
                    "confidence_overall",
                ],
            ),
        )
    ]
)


class PrecheckService:
    """Service for analyzing prompts and extracting simulation specs.
    
    Uses Google GenAI with function calling for reliable structured output.
    """

    def __init__(self):
        self.settings = get_settings()
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        """Lazy-load the Google GenAI client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    async def analyze(self, request: PrecheckRequest) -> PrecheckResponse:
        """Analyze a user prompt and extract simulation specifications.
        
        Uses Gemini with function calling for reliable structured output.
        """
        try:
            # Build the analysis prompt
            analysis_prompt = self._build_analysis_prompt(request)

            # Call Gemini with function calling
            logger.info(f"[Precheck] Calling Gemini model: {PRECHECK_MODEL} with tool calling")

            response = self.client.models.generate_content(
                model=PRECHECK_MODEL,
                contents=analysis_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    tools=[PRECHECK_TOOL_SCHEMA],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY",  # Force function call
                            allowed_function_names=["submit_cfd_configuration"],
                        )
                    ),
                ),
            )

            # Parse the function call response
            logger.info("[Precheck] Got function call response from Gemini")
            result = self._parse_function_call_response(response, request)
            return result

        except Exception as e:
            logger.exception(f"Precheck analysis failed: {e}")
            return self._create_fallback_response(request, str(e))

    def _parse_function_call_response(
        self, response: types.GenerateContentResponse, request: PrecheckRequest
    ) -> PrecheckResponse:
        """Parse the function call response from Gemini into PrecheckResponse."""
        try:
            # Extract function call from response
            candidate = response.candidates[0]
            part = candidate.content.parts[0]

            if not hasattr(part, 'function_call') or part.function_call is None:
                raise ValueError("No function call in response")

            func_call = part.function_call
            if func_call.name != "submit_cfd_configuration":
                raise ValueError(f"Unexpected function: {func_call.name}")

            # Get the arguments
            args = dict(func_call.args)
            logger.debug(f"[Precheck] Function args: {list(args.keys())}")

            # Build solver settings
            solver = SolverSettings(
                algorithm=args.get("solver_algorithm", "SIMPLE"),
                max_iterations=args.get("solver_max_iterations", 2000),
                convergence_criteria=args.get("solver_convergence_criteria", 1e-6),
                end_time=args.get("solver_end_time"),
                delta_t=args.get("solver_delta_t"),
                write_interval=args.get("solver_write_interval"),
            )

            # Build fluid properties
            preset_id = args.get("fluid_preset_id", "air")
            fluid = FluidProperties(
                preset_id=preset_id,
                name=args.get("fluid_name", preset_id.upper()),
                rho=args.get("fluid_rho", 1.225),
                mu=args.get("fluid_mu", 1.81e-5),
                Cp=args.get("fluid_Cp", 1006.0),
                k=args.get("fluid_k", 0.0257),
                temperature=args.get("fluid_temperature", 293.15),
            )

            # Build turbulence settings
            flow_regime = args.get("flow_regime", "turbulent")
            turbulence = TurbulenceSettings(
                model=args.get("turbulence_model", "kOmegaSST"),
                turbulence_intensity=args.get("turbulence_intensity", 5.0),
                turbulence_length_scale=args.get("turbulence_length_scale", 0.01),
                hydraulic_diameter=args.get("hydraulic_diameter", 0.1),
                wall_functions=args.get("wall_functions", True),
            )

            # Build boundary conditions from array
            boundary_conditions = {}
            bc_array = args.get("boundary_conditions", [])
            for bc in bc_array:
                patch_name = bc.get("patch_name", "unknown")
                boundary_conditions[patch_name] = PatchBoundaryCondition(
                    patch_class=bc.get("patch_class", "wall"),
                    confidence=bc.get("confidence", 0.8),
                    U=FieldBC(type=bc.get("U_type", "fixedValue"), value=bc.get("U_value")),
                    p=FieldBC(type=bc.get("p_type", "zeroGradient"), value=bc.get("p_value")),
                    T=FieldBC(type=bc["T_type"], value=bc.get("T_value")) if bc.get("T_type") else None,
                    k=FieldBC(type=bc["k_type"], value=bc.get("k_value")) if bc.get("k_type") else None,
                    epsilon=FieldBC(type=bc["epsilon_type"], value=bc.get("epsilon_value")) if bc.get("epsilon_type") else None,
                    omega=FieldBC(type=bc["omega_type"], value=bc.get("omega_value")) if bc.get("omega_type") else None,
                    nut=FieldBC(type=bc["nut_type"], value=bc.get("nut_value")) if bc.get("nut_type") else None,
                )

            # Build suggested config
            suggested_config = SuggestedConfig(
                case_type=args.get("case_type", "internal_pipe_flow"),
                flow_regime=flow_regime,
                time_scheme=args.get("time_scheme", "steady"),
                compressibility=args.get("compressibility", "incompressible"),
                enable_heat_transfer=args.get("enable_heat_transfer", False),
                gravity=args.get("gravity", False),
                solver=solver,
                fluid=fluid,
                turbulence=turbulence,
                boundary_conditions=boundary_conditions,
            )

            # Build legacy boundary hints
            boundary_hints = self._build_boundary_hints(boundary_conditions)

            # Build interpretation
            interpretation = Interpretation(
                summary=args.get("interpretation_summary", "CFD simulation analysis"),
                simulation_type=args.get("interpretation_simulation_type", "General CFD"),
                key_physics=args.get("interpretation_key_physics", []),
                assumptions=args.get("interpretation_assumptions", []),
                clarifications=args.get("interpretation_clarifications"),
            )

            # Build confidence scores
            confidence_scores = ConfidenceScores(
                overall=args.get("confidence_overall", 0.8),
                flow_regime=args.get("confidence_flow_regime", 0.8),
                boundary_conditions=args.get("confidence_boundary_conditions", 0.7),
                physics_settings=args.get("confidence_physics_settings", 0.8),
            )

            # Determine next step
            mesh = request.get_mesh()
            next_step = 2 if mesh else 1
            should_show_mesh = mesh is not None

            return PrecheckResponse(
                success=True,
                confidence=confidence_scores.overall,
                message=args.get("message", f"Detected {flow_regime} {suggested_config.case_type} simulation"),
                suggested_config=suggested_config,
                boundary_hints=boundary_hints,
                kpi_targets=None,  # KPI targets not in function schema for simplicity
                interpretation=interpretation,
                confidence_scores=confidence_scores,
                next_step=next_step,
                should_show_mesh_viewer=should_show_mesh,
            )

        except Exception as e:
            logger.warning(f"Failed to parse function call response: {e}")
            return self._create_fallback_response(request, f"Parse error: {e}")
    
    def _build_analysis_prompt(self, request: PrecheckRequest) -> str:
        """Build the prompt for LLM analysis."""
        mesh = request.get_mesh()

        prompt_parts = [
            "Analyze the following CFD simulation request and return a COMPLETE configuration.",
            "",
            f"User's description: {request.prompt}",
            "",
        ]
        
        # Add mesh info if available
        if mesh:
            prompt_parts.extend([
                "Uploaded mesh information:",
                f"  - File: {mesh.file_name}",
                f"  - Mesh ID: {mesh.mesh_id}",
                f"  - Cells: {mesh.check_mesh.cells:,}",
                f"  - Faces: {mesh.check_mesh.faces:,}",
                f"  - Points: {mesh.check_mesh.points:,}",
                "",
                "Mesh patches (boundaries):",
            ])
            for patch in mesh.patches:
                prompt_parts.append(f"  - {patch.name} (type: {patch.type}, cells: {patch.n_cells})")
            prompt_parts.append("")
        
        # Add previous config if refining
        if request.previous_config:
            prompt_parts.extend([
                "Previous configuration (user is refining):",
                json.dumps(request.previous_config, indent=2),
                "",
            ])
        
        prompt_parts.extend([
            "IMPORTANT: Return a COMPLETE suggested configuration with CALCULATED NUMERICAL VALUES.",
            "DO NOT use 'calculated' type with null values - always compute the actual numbers!",
            "",
            "Include ALL of the following:",
            "1. Physics: case_type, flow_regime, time_scheme, compressibility, enable_heat_transfer, gravity",
            "2. Solver: algorithm (SIMPLE/PIMPLE/PISO), max_iterations, convergence_criteria, and for transient: end_time, delta_t, write_interval",
            "3. Fluid: preset_id (air/water/oil/ln2/custom), name, rho, mu, Cp, k, temperature (in Kelvin)",
            "4. Turbulence: model, turbulence_intensity (%), turbulence_length_scale, hydraulic_diameter, wall_functions",
            "5. Boundary conditions: For EACH patch, provide OpenFOAM-style BCs with ACTUAL VALUES for U, p, T (if heat transfer), k, omega, epsilon, nut",
            "",
            "=== FLUID PROPERTIES (use these or calculate custom values) ===",
            "Air:   rho=1.225 kg/m³, mu=1.81e-5 Pa·s, Cp=1006 J/(kg·K), k=0.0257 W/(m·K), T=293.15K",
            "Water: rho=998.2 kg/m³, mu=1.002e-3 Pa·s, Cp=4182 J/(kg·K), k=0.598 W/(m·K), T=293.15K",
            "LN2:   rho=808 kg/m³, mu=1.58e-4 Pa·s, Cp=2042 J/(kg·K), k=0.140 W/(m·K), T=77K",
            "",
            "=== VELOCITY CALCULATIONS (ALWAYS CALCULATE!) ===",
            "From mass flow rate: U = m_dot / (rho * A)",
            "  where A = π * (D/2)² for circular pipe",
            "  Example: m_dot=0.089 kg/s, rho=808 kg/m³, D=0.025m → A=π*(0.0125)²=4.91e-4 m² → U=0.089/(808*4.91e-4)=0.224 m/s",
            "",
            "=== TURBULENCE CALCULATIONS (ALWAYS CALCULATE!) ===",
            "Constants: Cmu = 0.09",
            "Turbulence intensity I = 0.05 (5% default), or calculate: I = 0.16 * Re^(-1/8)",
            "Length scale L = 0.07 * D_h (hydraulic diameter)",
            "",
            "k (turbulent kinetic energy) = 1.5 * (U * I)²",
            "  Example: U=0.224 m/s, I=0.05 → k = 1.5 * (0.224 * 0.05)² = 1.88e-4 m²/s²",
            "",
            "epsilon (dissipation rate) = Cmu^0.75 * k^1.5 / L",
            "  Example: k=1.88e-4, L=0.00175m → epsilon = 0.09^0.75 * (1.88e-4)^1.5 / 0.00175 = 8.4e-5 m²/s³",
            "",
            "omega (specific dissipation) = sqrt(k) / (Cmu^0.25 * L)",
            "  Example: k=1.88e-4, L=0.00175m → omega = sqrt(1.88e-4) / (0.09^0.25 * 0.00175) = 14.3 1/s",
            "",
            "nut (turbulent viscosity at inlet) = k / omega (or Cmu * k² / epsilon)",
            "",
            "=== BOUNDARY CONDITION PATTERNS ===",
            "INLET:",
            "  U: type=fixedValue, value=[Ux, Uy, Uz] (calculated velocity vector)",
            "  p: type=zeroGradient",
            "  T: type=fixedValue, value=<inlet_temp_in_K> (if heat transfer enabled)",
            "  k: type=fixedValue, value=<calculated_k>",
            "  omega: type=fixedValue, value=<calculated_omega>",
            "  epsilon: type=fixedValue, value=<calculated_epsilon>",
            "  nut: type=calculated, value=0",
            "",
            "OUTLET:",
            "  U: type=zeroGradient or inletOutlet",
            "  p: type=fixedValue, value=0 (gauge pressure)",
            "  T: type=zeroGradient (if heat transfer enabled)",
            "  k: type=zeroGradient",
            "  omega: type=zeroGradient",
            "  epsilon: type=zeroGradient",
            "  nut: type=calculated, value=0",
            "",
            "WALL (with heat transfer):",
            "  U: type=noSlip",
            "  p: type=zeroGradient",
            "  T: type=fixedValue, value=<wall_temp_in_K> (e.g., 400K for heated wall)",
            "  k: type=kqRWallFunction, value=0",
            "  omega: type=omegaWallFunction, value=0",
            "  epsilon: type=epsilonWallFunction, value=0",
            "  nut: type=nutkWallFunction, value=0",
            "",
            "RESPOND WITH ONLY VALID JSON, NO MARKDOWN CODE BLOCKS.",
            "",
            "JSON Schema:",
            """{
  "message": "Summary of detected simulation",
  "suggested_config": {
    "case_type": "internal_pipe_flow" | "external_aero" | "heat_exchanger" | "mixing" | "general",
    "flow_regime": "laminar" | "turbulent",
    "time_scheme": "steady" | "transient",
    "compressibility": "incompressible" | "compressible",
    "enable_heat_transfer": boolean,
    "gravity": boolean,
    "solver": {
      "algorithm": "SIMPLE" | "PIMPLE" | "PISO",
      "max_iterations": number,
      "convergence_criteria": number,
      "end_time": number | null,
      "delta_t": number | null,
      "write_interval": number | null
    },
    "fluid": {
      "preset_id": "air" | "water" | "oil" | "ln2" | "custom",
      "name": string,
      "rho": number,
      "mu": number,
      "Cp": number,
      "k": number,
      "temperature": number
    },
    "turbulence": {
      "model": "kEpsilon" | "kOmegaSST" | "spalartAllmaras" | "laminar",
      "turbulence_intensity": number,
      "turbulence_length_scale": number,
      "hydraulic_diameter": number,
      "wall_functions": boolean
    },
    "boundary_conditions": {
    "<patchName>": {
        "patch_class": "inlet" | "outlet" | "wall" | "symmetry" | "periodic",
      "confidence": number,
        "U": { "type": string, "value": [x, y, z] },
        "p": { "type": string, "value": number },
        "T": { "type": string, "value": number } | null,
        "k": { "type": string, "value": number },
        "omega": { "type": string, "value": number },
        "epsilon": { "type": string, "value": number },
        "nut": { "type": string, "value": number }
      }
    }
  },
  "kpi_targets": {
    "pressure_drop": { "value": number, "unit": string } | null,
    "flow_rate": { "value": number, "unit": string } | null,
    "velocity": { "value": number, "unit": string } | null,
    "custom": []
  },
  "interpretation": {
    "summary": string,
    "simulation_type": string,
    "key_physics": [string],
    "assumptions": [string],
    "clarifications": [string] | null
  },
  "confidence_scores": {
    "overall": number,
    "flow_regime": number,
    "boundary_conditions": number,
    "physics_settings": number
  }
}""",
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_field_bc(self, data: dict | None) -> FieldBC | None:
        """Parse a field boundary condition."""
        if not data:
            return None
        return FieldBC(
            type=data.get("type", "fixedValue"),
            value=data.get("value"),
        )
    
    def _parse_kpi_value(self, data: dict | None) -> KPIValue | None:
        """Parse KPI value."""
        if not data:
            return None
        return KPIValue(
            value=data.get("value", 0),
            unit=data.get("unit", ""),
        )
    
    def _build_boundary_hints(
        self, boundary_conditions: dict[str, PatchBoundaryCondition]
    ) -> dict[str, BoundaryHint]:
        """Build legacy boundary hints from new boundary conditions."""
        hints = {}
        for patch_name, bc in boundary_conditions.items():
            velocity = None
            if bc.U:
                velocity = VelocityBC(
                    type=bc.U.type,
                    value=bc.U.value if isinstance(bc.U.value, list) else None,
                    magnitude=bc.U.value if isinstance(bc.U.value, (int, float)) else None,
                )

            pressure = None
            if bc.p:
                pressure = PressureBC(
                    type=bc.p.type,
                    value=bc.p.value if isinstance(bc.p.value, (int, float)) else None,
                )

            temperature = None
            if bc.T:
                temperature = TemperatureBC(
                    type=bc.T.type,
                    value=bc.T.value if isinstance(bc.T.value, (int, float)) else None,
                )

            hints[patch_name] = BoundaryHint(
                suggested_type=bc.patch_class,
                velocity=velocity,
                pressure=pressure,
                temperature=temperature,
                confidence=bc.confidence,
                reasoning=f"Classified as {bc.patch_class}",
            )

        return hints
    
    def _create_fallback_response(self, request: PrecheckRequest, error: str) -> PrecheckResponse:
        """Create a fallback response when LLM fails using heuristics."""
        prompt_lower = request.prompt.lower()
        mesh = request.get_mesh()
        
        # Detect flow regime
        is_turbulent = any(word in prompt_lower for word in [
            "turbulent", "industrial", "high speed", "fast", "re >", "reynolds"
        ])
        is_laminar = any(word in prompt_lower for word in [
            "laminar", "slow", "creeping", "viscous", "low speed"
        ])
        flow_regime = "laminar" if is_laminar and not is_turbulent else "turbulent"
        
        # Detect time scheme
        is_transient = any(word in prompt_lower for word in [
            "transient", "unsteady", "time", "pulsating", "oscillating"
        ])
        time_scheme = "transient" if is_transient else "steady"
        
        # Detect fluid type
        uses_ln2 = any(word in prompt_lower for word in ["ln2", "liquid nitrogen", "nitrogen", "cryogenic"])
        uses_water = any(word in prompt_lower for word in ["water", "liquid", "hydraulic"])
        if uses_ln2:
            fluid = FLUID_PRESETS["ln2"]
        elif uses_water:
            fluid = FLUID_PRESETS["water"]
        else:
            fluid = FLUID_PRESETS["air"]

        # Extract temperatures from prompt (e.g., "77K", "400 K", "293.15 kelvin")
        inlet_temp = fluid.temperature  # Default to fluid's reference temp
        wall_temp = 300.0  # Default wall temp
        # Match numbers followed by K or kelvin (with optional space)
        temp_matches = re.findall(r"(\d+(?:\.\d+)?)\s*k(?:elvin)?(?:\b|$)", prompt_lower, re.IGNORECASE)
        if temp_matches:
            temps = [float(t) for t in temp_matches]
            # Assume smaller temp is inlet (cryogenic), larger is wall (heated)
            if len(temps) >= 2:
                inlet_temp = min(temps)
                wall_temp = max(temps)
            else:
                # Single temp - decide based on context
                if "wall" in prompt_lower or "heated" in prompt_lower:
                    wall_temp = temps[0]
                else:
                    inlet_temp = temps[0]

        # Detect heat transfer - either explicit keywords or different temperatures detected
        has_heat = any(word in prompt_lower for word in [
            "heat", "thermal", "temperature", "cooling", "heating", "hot", "cold"
        ])
        # Also enable heat transfer if we have multiple temperatures or wall temp differs from inlet
        if len(temp_matches) >= 2 or (temp_matches and abs(wall_temp - inlet_temp) > 50):
            has_heat = True

        # Extract hydraulic diameter / pipe ID
        diam_match = re.search(r"(?:id|diameter|d)\s*[=:]\s*(\d+(?:\.\d+)?)\s*mm", prompt_lower)
        hydraulic_diameter = float(diam_match.group(1)) / 1000.0 if diam_match else 0.025  # Default 25mm

        # Extract mass flow rate
        mass_flow_match = re.search(r"(\d+(?:\.\d+)?)\s*g/s(?:ec)?", prompt_lower)
        mass_flow_rate = float(mass_flow_match.group(1)) / 1000.0 if mass_flow_match else None  # kg/s

        # Note: Pressure extracted but OpenFOAM typically uses gauge pressure (0 at outlet)
        # pressure_match = re.search(r"(\d+(?:\.\d+)?)\s*bar", prompt_lower)
        # inlet_pressure = float(pressure_match.group(1)) * 1e5 if pressure_match else 101325  # Pa
        
        # Detect simulation type
        is_internal = any(word in prompt_lower for word in [
            "pipe", "duct", "channel", "internal", "tube"
        ])
        is_external = any(word in prompt_lower for word in [
            "external", "wind", "aerodynamic", "vehicle", "airfoil", "wing"
        ])
        case_type = "internal_pipe_flow" if is_internal else "external_aero" if is_external else "general"
        sim_type = "Internal pipe flow" if is_internal else "External aerodynamics" if is_external else "General CFD"

        # Extract velocity if mentioned, or calculate from mass flow rate
        velocity_match = re.search(r"(\d+(?:\.\d+)?)\s*m/s", prompt_lower)
        if velocity_match:
            inlet_velocity = float(velocity_match.group(1))
        elif mass_flow_rate is not None:
            # Calculate from mass flow rate: U = m_dot / (rho * A)
            area = math.pi * (hydraulic_diameter / 2) ** 2
            inlet_velocity = mass_flow_rate / (fluid.rho * area)
        else:
            inlet_velocity = 1.0

        # Calculate turbulence quantities
        turb_intensity = 0.05  # 5% turbulence intensity
        length_scale = 0.07 * hydraulic_diameter  # length scale = 0.07 * Dh
        Cmu = 0.09
        U_mag = inlet_velocity
        k_inlet = 1.5 * (U_mag * turb_intensity) ** 2
        omega_inlet = math.sqrt(k_inlet) / (Cmu ** 0.25 * length_scale)
        epsilon_inlet = Cmu ** 0.75 * k_inlet ** 1.5 / length_scale

        # Build boundary conditions from mesh patches
        boundary_conditions = {}
        if mesh:
            for patch in mesh.patches:
                patch_lower = patch.name.lower()
                if any(x in patch_lower for x in ["inlet", "inflow", "in"]):
                    bc_class = "inlet"
                    boundary_conditions[patch.name] = PatchBoundaryCondition(
                        patch_class=bc_class,
                        confidence=0.7,
                        U=FieldBC(type="fixedValue", value=[inlet_velocity, 0, 0]),
                        p=FieldBC(type="zeroGradient"),
                        T=FieldBC(type="fixedValue", value=inlet_temp) if has_heat else None,
                        k=FieldBC(type="fixedValue", value=k_inlet) if flow_regime == "turbulent" else None,
                        omega=FieldBC(type="fixedValue", value=omega_inlet) if flow_regime == "turbulent" else None,
                        epsilon=FieldBC(type="fixedValue", value=epsilon_inlet) if flow_regime == "turbulent" else None,
                        nut=FieldBC(type="calculated", value=0) if flow_regime == "turbulent" else None,
                    )
                elif any(x in patch_lower for x in ["outlet", "outflow", "out", "exit"]):
                    bc_class = "outlet"
                    boundary_conditions[patch.name] = PatchBoundaryCondition(
                        patch_class=bc_class,
                        confidence=0.7,
                        U=FieldBC(type="zeroGradient"),
                        p=FieldBC(type="fixedValue", value=0),
                        T=FieldBC(type="zeroGradient") if has_heat else None,
                        k=FieldBC(type="zeroGradient") if flow_regime == "turbulent" else None,
                        omega=FieldBC(type="zeroGradient") if flow_regime == "turbulent" else None,
                        epsilon=FieldBC(type="zeroGradient") if flow_regime == "turbulent" else None,
                        nut=FieldBC(type="calculated", value=0) if flow_regime == "turbulent" else None,
                    )
                elif any(x in patch_lower for x in ["wall", "surface"]):
                    bc_class = "wall"
                    boundary_conditions[patch.name] = PatchBoundaryCondition(
                        patch_class=bc_class,
                        confidence=0.9,
                        U=FieldBC(type="noSlip"),
                        p=FieldBC(type="zeroGradient"),
                        T=FieldBC(type="fixedValue", value=wall_temp) if has_heat else None,
                        k=FieldBC(type="kqRWallFunction", value=0) if flow_regime == "turbulent" else None,
                        omega=FieldBC(type="omegaWallFunction", value=0) if flow_regime == "turbulent" else None,
                        epsilon=FieldBC(type="epsilonWallFunction", value=0) if flow_regime == "turbulent" else None,
                        nut=FieldBC(type="nutkWallFunction", value=0) if flow_regime == "turbulent" else None,
                    )
                elif any(x in patch_lower for x in ["sym", "symmetry"]):
                    bc_class = "symmetry"
                    boundary_conditions[patch.name] = PatchBoundaryCondition(
                        patch_class=bc_class,
                        confidence=0.9,
                        U=FieldBC(type="symmetry"),
                        p=FieldBC(type="symmetry"),
                        T=FieldBC(type="symmetry") if has_heat else None,
                        k=FieldBC(type="symmetry") if flow_regime == "turbulent" else None,
                        omega=FieldBC(type="symmetry") if flow_regime == "turbulent" else None,
                        epsilon=FieldBC(type="symmetry") if flow_regime == "turbulent" else None,
                        nut=FieldBC(type="symmetry") if flow_regime == "turbulent" else None,
                    )
                else:
                    # Default to wall
                    bc_class = "wall"
                    boundary_conditions[patch.name] = PatchBoundaryCondition(
                        patch_class=bc_class,
                        confidence=0.5,
                        U=FieldBC(type="noSlip"),
                        p=FieldBC(type="zeroGradient"),
                        T=FieldBC(type="fixedValue", value=wall_temp) if has_heat else None,
                        k=FieldBC(type="kqRWallFunction", value=0) if flow_regime == "turbulent" else None,
                        omega=FieldBC(type="omegaWallFunction", value=0) if flow_regime == "turbulent" else None,
                        nut=FieldBC(type="nutkWallFunction", value=0) if flow_regime == "turbulent" else None,
                    )

        # Build solver settings
        solver = SolverSettings(
            algorithm="SIMPLE" if time_scheme == "steady" else "PIMPLE",
            max_iterations=2000 if time_scheme == "steady" else 50,
            convergence_criteria=1e-6,
            end_time=1.0 if is_transient else None,
            delta_t=0.001 if is_transient else None,
            write_interval=0.1 if is_transient else None,
        )

        # Build turbulence settings
        turbulence = TurbulenceSettings(
            model="kOmegaSST" if flow_regime == "turbulent" else "laminar",
            turbulence_intensity=turb_intensity * 100,  # Convert to percentage
            turbulence_length_scale=length_scale,
            hydraulic_diameter=hydraulic_diameter,
            wall_functions=True,
        )

        suggested_config = SuggestedConfig(
            case_type=case_type,
            flow_regime=flow_regime,
            time_scheme=time_scheme,
            compressibility="incompressible",
            enable_heat_transfer=has_heat,
            gravity=False,
            solver=solver,
            fluid=fluid,
            turbulence=turbulence,
            boundary_conditions=boundary_conditions,
        )

        boundary_hints = self._build_boundary_hints(boundary_conditions)

        return PrecheckResponse(
            success=False,
            confidence=0.4,
            message=f"Fallback: Detected {flow_regime} {case_type} (LLM unavailable)",
            suggested_config=suggested_config,
            boundary_hints=boundary_hints,
            interpretation=Interpretation(
                summary=f"Fallback analysis (LLM error: {error})",
                simulation_type=sim_type,
                key_physics=["turbulence"] if flow_regime == "turbulent" else [],
                assumptions=["Using heuristic-based defaults due to LLM error"],
                clarifications=["Please review and adjust settings manually"],
            ),
            confidence_scores=ConfidenceScores(
                overall=0.4,
                flow_regime=0.5,
                boundary_conditions=0.4,
                physics_settings=0.5,
            ),
            next_step=1,
            should_show_mesh_viewer=mesh is not None,
            warnings=[f"LLM analysis failed, using heuristics: {error}"],
        )


# Singleton instance
_precheck_service: PrecheckService | None = None


def get_precheck_service() -> PrecheckService:
    """Get or create the precheck service singleton."""
    global _precheck_service
    if _precheck_service is None:
        _precheck_service = PrecheckService()
    return _precheck_service
