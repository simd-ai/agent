# simd_agent/precheck_models.py
"""Pydantic models, fluid presets, and the tool schema for the precheck service."""

from typing import Any, Literal

from google.genai import types
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class MeshPatch(BaseModel):
    name: str
    type: str  # "wall" | "patch" | "empty" | "symmetry" …
    n_cells: int = Field(alias="nCells", default=0)

    class Config:
        populate_by_name = True


class CheckMeshInfo(BaseModel):
    cells: int
    faces: int
    points: int
    bounding_box: dict[str, list[float]] | None = Field(alias="boundingBox", default=None)
    characteristic_length: float | None = Field(alias="characteristicLength", default=None)

    class Config:
        populate_by_name = True


class MeshInfo(BaseModel):
    mesh_id: str = Field(alias="meshId")
    file_name: str = Field(alias="fileName")
    patches: list[MeshPatch]
    check_mesh: CheckMeshInfo = Field(alias="checkMesh")

    class Config:
        populate_by_name = True


class PrecheckRequest(BaseModel):
    prompt: str = Field(default="", alias="prompt")
    has_mesh: bool = Field(default=False, alias="hasMesh")
    mesh_info: MeshInfo | None = Field(default=None, alias="meshInfo")
    mesh: MeshInfo | None = None  # legacy alias
    previous_config: dict[str, Any] | None = Field(default=None, alias="previousConfig")

    class Config:
        populate_by_name = True

    def validate_prompt(self) -> str | None:
        """Return a human-friendly error string if the request cannot proceed, else None."""
        if not self.prompt or not self.prompt.strip():
            return (
                "Please describe your simulation before running the analysis — "
                "e.g. what fluid, geometry, inlet velocity, and goals you have."
            )
        if not self.get_mesh():
            return (
                "Please upload a mesh file before running the analysis. "
                "The mesh defines your geometry and boundary patches "
                "(inlet, outlet, walls, etc.) that the solver needs."
            )
        return None

    def get_mesh(self) -> MeshInfo | None:
        return self.mesh_info or self.mesh


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SolverSettings(BaseModel):
    algorithm: Literal["SIMPLE", "PIMPLE", "PISO"] = "SIMPLE"
    max_iterations: int = Field(default=2000, alias="maxIterations")
    convergence_criteria: float = Field(default=1e-6, alias="convergenceCriteria")
    end_time: float | None = Field(default=None, alias="endTime")
    delta_t: float | None = Field(default=None, alias="deltaT")
    write_interval: float | None = Field(default=None, alias="writeInterval")

    class Config:
        populate_by_name = True


class FluidProperties(BaseModel):
    preset_id: str = Field(default="air", alias="presetId")
    name: str = "Air"
    rho: float = 1.225        # kg/m³
    mu: float = 1.81e-5       # Pa·s
    Cp: float = 1006.0        # J/(kg·K)
    k: float = 0.0257         # W/(m·K)
    temperature: float = 293.15  # K

    class Config:
        populate_by_name = True


class TurbulenceSettings(BaseModel):
    model: Literal["kEpsilon", "kOmegaSST", "spalartAllmaras", "laminar"] = "kOmegaSST"
    turbulence_intensity: float = Field(default=5.0, alias="turbulenceIntensity")   # %
    turbulence_length_scale: float = Field(default=0.01, alias="turbulenceLengthScale")  # m
    hydraulic_diameter: float = Field(default=0.1, alias="hydraulicDiameter")  # m
    wall_functions: bool = Field(default=True, alias="wallFunctions")

    class Config:
        populate_by_name = True


class FieldBC(BaseModel):
    type: str
    value: float | list[float] | None = None


class PatchBoundaryCondition(BaseModel):
    patch_class: Literal["inlet", "outlet", "wall", "symmetry", "periodic", "empty"] = Field(
        alias="patchClass"
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    U: FieldBC | None = None
    p: FieldBC | None = None
    T: FieldBC | None = None
    k: FieldBC | None = None
    epsilon: FieldBC | None = None
    omega: FieldBC | None = None
    nut: FieldBC | None = None

    class Config:
        populate_by_name = True


class SuggestedConfig(BaseModel):
    case_type: str = Field(default="internal_flow", alias="caseType")
    flow_regime: Literal["laminar", "turbulent"] = Field(alias="flowRegime")
    time_scheme: Literal["steady", "transient"] = Field(alias="timeScheme")
    compressibility: Literal["incompressible", "compressible"] = "incompressible"
    enable_heat_transfer: bool = Field(default=False, alias="enableHeatTransfer")
    gravity: bool = False
    solver: SolverSettings
    fluid: FluidProperties
    turbulence: TurbulenceSettings
    boundary_conditions: dict[str, PatchBoundaryCondition] = Field(
        default_factory=dict, alias="boundaryConditions"
    )

    class Config:
        populate_by_name = True


# Legacy hint models (kept for backward compat)

class VelocityBC(BaseModel):
    type: str
    value: list[float] | None = None
    magnitude: float | None = None


class PressureBC(BaseModel):
    type: str
    value: float | None = None


class TemperatureBC(BaseModel):
    type: str
    value: float | None = None


class BoundaryHint(BaseModel):
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
    value: float
    unit: str


class CustomKPI(BaseModel):
    name: str
    value: float
    unit: str


class KPITargets(BaseModel):
    pressure_drop: KPIValue | None = Field(default=None, alias="pressureDrop")
    flow_rate: KPIValue | None = Field(default=None, alias="flowRate")
    temperature: KPIValue | None = None
    velocity: KPIValue | None = None
    custom: list[CustomKPI] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class Interpretation(BaseModel):
    summary: str
    simulation_type: str = Field(alias="simulationType")
    key_physics: list[str] = Field(default_factory=list, alias="keyPhysics")
    assumptions: list[str] = Field(default_factory=list)
    clarifications: list[str] | None = None

    class Config:
        populate_by_name = True


class ConfidenceScores(BaseModel):
    overall: float = Field(default=0.5, ge=0.0, le=1.0)
    flow_regime: float = Field(default=0.5, ge=0.0, le=1.0, alias="flowRegime")
    boundary_conditions: float = Field(default=0.5, ge=0.0, le=1.0, alias="boundaryConditions")
    physics_settings: float = Field(default=0.5, ge=0.0, le=1.0, alias="physicsSettings")

    class Config:
        populate_by_name = True


class PrecheckResponse(BaseModel):
    success: bool
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    message: str = ""
    suggested_config: SuggestedConfig = Field(alias="suggestedConfig")
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


# ---------------------------------------------------------------------------
# Fluid presets
# ---------------------------------------------------------------------------

FLUID_PRESETS: dict[str, FluidProperties] = {
    "air": FluidProperties(
        preset_id="air", name="Air",
        rho=1.225, mu=1.81e-5, Cp=1006.0, k=0.0257, temperature=293.15,
    ),
    "water": FluidProperties(
        preset_id="water", name="Water",
        rho=998.2, mu=1.002e-3, Cp=4182.0, k=0.598, temperature=293.15,
    ),
    "oil": FluidProperties(
        preset_id="oil", name="Oil (SAE 30)",
        rho=880.0, mu=0.29, Cp=1900.0, k=0.145, temperature=293.15,
    ),
    "ln2": FluidProperties(
        preset_id="ln2", name="Liquid Nitrogen (LN2)",
        rho=808.0, mu=1.58e-4, Cp=2042.0, k=0.140, temperature=77.0,
    ),
    "lng": FluidProperties(
        preset_id="lng", name="Liquefied Natural Gas (LNG)",
        rho=450.0,    # kg/m³ at ~111 K, 1 atm
        mu=1.2e-4,    # Pa·s at 111 K
        Cp=3500.0,    # J/(kg·K)
        k=0.185,      # W/(m·K)
        temperature=111.0,  # K (boiling point at 1 atm)
    ),
    "helium": FluidProperties(
        preset_id="helium", name="Helium (gas)",
        rho=0.164,    # kg/m³ at 293 K, 1 atm
        mu=1.96e-5,   # Pa·s at 293 K
        Cp=5193.0,    # J/(kg·K)
        k=0.152,      # W/(m·K)
        temperature=293.15,
    ),
}

# Keywords that map to cryogenic presets (used by both LLM prompt and fallback)
CRYOGENIC_KEYWORDS: tuple[str, ...] = (
    "ln2", "liquid nitrogen",
    "lh2", "liquid hydrogen",
    "lox", "liquid oxygen",
    "lng", "liquefied natural gas", "liquid natural gas",
    "helium", "lhe", "liquid helium",
    "cryogenic", "cryo",
)


# ---------------------------------------------------------------------------
# Gemini tool schema
# ---------------------------------------------------------------------------

PRECHECK_MODEL = "gemini-3-flash-preview"   # gemini-3-flash-preview -  gemini-3.1-pro-preview
REVIEW_MODEL   = "gemini-3-flash-preview"   # deep review + corrections

PRECHECK_TOOL_SCHEMA = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="submit_cfd_configuration",
            description="Submit the analyzed CFD simulation configuration",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "message": types.Schema(type="STRING"),
                    "case_type": types.Schema(
                        type="STRING",
                        enum=["internal_pipe_flow", "external_aero", "heat_exchanger", "mixing", "general"],
                    ),
                    "flow_regime": types.Schema(type="STRING", enum=["laminar", "turbulent"]),
                    "time_scheme": types.Schema(type="STRING", enum=["steady", "transient"]),
                    "compressibility": types.Schema(type="STRING", enum=["incompressible", "compressible"]),
                    "enable_heat_transfer": types.Schema(type="BOOLEAN"),
                    "gravity": types.Schema(type="BOOLEAN"),
                    "solver_algorithm": types.Schema(type="STRING", enum=["SIMPLE", "PIMPLE", "PISO"]),
                    "solver_max_iterations": types.Schema(type="INTEGER"),
                    "solver_convergence_criteria": types.Schema(type="NUMBER"),
                    "solver_end_time": types.Schema(type="NUMBER", nullable=True),
                    "solver_delta_t": types.Schema(type="NUMBER", nullable=True),
                    "fluid_preset_id": types.Schema(
                        type="STRING",
                        enum=["air", "water", "oil", "ln2", "lng", "helium", "custom"],
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
                    "turbulence_intensity": types.Schema(type="NUMBER"),
                    "turbulence_length_scale": types.Schema(type="NUMBER"),
                    "hydraulic_diameter": types.Schema(type="NUMBER"),
                    "wall_functions": types.Schema(type="BOOLEAN"),
                    "boundary_conditions": types.Schema(
                        type="ARRAY",
                        items=types.Schema(
                            type="OBJECT",
                            properties={
                                "patch_name": types.Schema(type="STRING"),
                                "patch_class": types.Schema(
                                    type="STRING",
                                    enum=["inlet", "outlet", "wall", "symmetry", "periodic", "empty"],
                                ),
                                "confidence": types.Schema(type="NUMBER"),
                                "U_type": types.Schema(type="STRING"),
                                "U_value": types.Schema(type="ARRAY", items=types.Schema(type="NUMBER"), nullable=True),
                                "p_type": types.Schema(type="STRING"),
                                "p_value": types.Schema(type="NUMBER", nullable=True),
                                "T_type": types.Schema(
                                    type="STRING",
                                    description=(
                                        "REQUIRED for every patch when enable_heat_transfer=true. "
                                        "inlet → 'fixedValue' (fluid temperature). "
                                        "outlet → 'zeroGradient'. "
                                        "wall → 'fixedValue' (wall temperature). "
                                        "empty/frontAndBack → 'empty'. "
                                        "Use null only when enable_heat_transfer=false."
                                    ),
                                    nullable=True,
                                ),
                                "T_value": types.Schema(
                                    type="NUMBER",
                                    description=(
                                        "Temperature in K. "
                                        "inlet: fluid temperature (e.g. 77 for LN2) — NOT the wall temp. "
                                        "wall: wall temperature stated by user — NOT the fluid temp. "
                                        "outlet/empty: null."
                                    ),
                                    nullable=True,
                                ),
                                "k_type": types.Schema(type="STRING", nullable=True),
                                "k_value": types.Schema(type="NUMBER", nullable=True),
                                "omega_type": types.Schema(type="STRING", nullable=True),
                                "omega_value": types.Schema(type="NUMBER", nullable=True),
                                "epsilon_type": types.Schema(type="STRING", nullable=True),
                                "epsilon_value": types.Schema(type="NUMBER", nullable=True),
                                "nut_type": types.Schema(type="STRING", nullable=True),
                                "nut_value": types.Schema(type="NUMBER", nullable=True),
                            },
                            required=["patch_name", "patch_class", "U_type", "p_type", "T_type"],
                        ),
                    ),
                    "interpretation_summary": types.Schema(type="STRING"),
                    "interpretation_simulation_type": types.Schema(type="STRING"),
                    "interpretation_key_physics": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                    "interpretation_assumptions": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                    "confidence_overall": types.Schema(type="NUMBER"),
                    "confidence_flow_regime": types.Schema(type="NUMBER"),
                    "confidence_boundary_conditions": types.Schema(type="NUMBER"),
                    "confidence_physics_settings": types.Schema(type="NUMBER"),
                },
                required=[
                    "message", "case_type", "flow_regime", "time_scheme",
                    "enable_heat_transfer", "fluid_preset_id", "fluid_rho", "fluid_mu",
                    "turbulence_model", "boundary_conditions",
                    "interpretation_summary", "confidence_overall",
                ],
            ),
        )
    ]
)


# ---------------------------------------------------------------------------
# Review tool schema  (second LLM pass — spec review + correction)
# ---------------------------------------------------------------------------

REVIEW_TOOL_SCHEMA = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="submit_review",
            description=(
                "Submit the structured review of the CFD configuration. "
                "Each item explains one check with math notation. "
                "CRITICAL: every numeric value in corrected_boundary_conditions MUST be "
                "copied verbatim from the formula you computed in the corresponding detail "
                "field — do NOT re-derive or re-round at fill time."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "items": types.Schema(
                        type="ARRAY",
                        description=(
                            "One item per check group. For boundary conditions produce one "
                            "item per patch, listing every field verified."
                        ),
                        items=types.Schema(
                            type="OBJECT",
                            properties={
                                "patch": types.Schema(type="STRING", nullable=True,
                                    description="Patch name, or null for global checks"),
                                "field": types.Schema(type="STRING", nullable=True,
                                    description="Field name: U, p, T, k, omega, epsilon, turbulence, fluid, fvOptions, summary…"),
                                "status": types.Schema(type="STRING",
                                    enum=["ok", "corrected", "warning", "error"],
                                    description=(
                                        "Almost always 'ok'. "
                                        "Use 'warning' for genuine physics concerns. "
                                        "Use 'error' only for unresolvable problems. "
                                        "Never use 'corrected' — silently fix values in "
                                        "corrected_boundary_conditions instead."
                                    )),
                                "label": types.Schema(type="STRING",
                                    description="Short, natural title. Use backticks for identifiers."),
                                "detail": types.Schema(type="STRING",
                                    description=(
                                        "Natural-language confirmation of the final correct value. "
                                        "Never say 'was X, corrected to Y' or 'fixed' or 'error'. "
                                        "Use LaTeX math ($...$) inline, display math ($$...$$) on "
                                        "its own line, and backticks for identifiers."
                                    )),
                            },
                            required=["status", "label", "detail"],
                        ),
                    ),
                    "corrections_made": types.Schema(type="BOOLEAN"),
                    "corrected_boundary_conditions": types.Schema(
                        type="ARRAY",
                        nullable=True,
                        description=(
                            "Provide only if corrections_made=true. "
                            "Include ALL patches (not just changed ones). "
                            "CRITICAL: numeric values here must be the EXACT same numbers "
                            "computed and stated in the detail fields above. Do NOT recompute."
                        ),
                        items=types.Schema(
                            type="OBJECT",
                            properties={
                                "patch_name":   types.Schema(type="STRING"),
                                "patch_class":  types.Schema(type="STRING",
                                    enum=["inlet", "outlet", "wall", "symmetry", "periodic", "empty"]),
                                "confidence":   types.Schema(type="NUMBER", nullable=True,
                                    description="Confidence 0-1 for this patch classification"),
                                "U_type":       types.Schema(type="STRING",  nullable=True),
                                "U_value":      types.Schema(type="ARRAY", nullable=True,
                                    description="[Ux, Uy, Uz] — copy exact vector from check #2",
                                    items=types.Schema(type="NUMBER")),
                                "p_type":       types.Schema(type="STRING",  nullable=True),
                                "p_value":      types.Schema(type="NUMBER",  nullable=True,
                                    description="Pressure value in Pa — copy from user-stated operating pressure"),
                                "T_type":       types.Schema(type="STRING",  nullable=True),
                                "T_value":      types.Schema(type="NUMBER",  nullable=True,
                                    description=(
                                        "Temperature in K. "
                                        "inlet: fluid temperature (NOT wall temp). "
                                        "wall: wall temperature stated by user (NOT fluid temp). "
                                        "outlet/empty: null (zeroGradient or empty)."
                                    )),
                                "k_type":       types.Schema(type="STRING",  nullable=True),
                                "k_value":      types.Schema(type="NUMBER",  nullable=True,
                                    description="k in m²/s² — copy exact value from check #3 formula result"),
                                "omega_type":   types.Schema(type="STRING",  nullable=True),
                                "omega_value":  types.Schema(type="NUMBER",  nullable=True,
                                    description="omega in s⁻¹ — copy exact value from check #3 formula result"),
                                "epsilon_type": types.Schema(type="STRING",  nullable=True),
                                "epsilon_value":types.Schema(type="NUMBER",  nullable=True,
                                    description="epsilon in m²/s³ — copy exact value from check #3 formula result"),
                                "nut_type":     types.Schema(type="STRING",  nullable=True),
                                "nut_value":    types.Schema(type="NUMBER",  nullable=True),
                            },
                            required=["patch_name", "patch_class"],
                        ),
                    ),
                },
                required=["items", "corrections_made"],
            ),
        )
    ]
)
