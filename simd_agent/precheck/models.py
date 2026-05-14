# simd_agent/precheck_models.py
"""Pydantic models, fluid presets, and the tool schema for the precheck service."""

import pathlib as _pathlib
from typing import Any, Literal

from simd_agent.llm.gemini.provider import genai_types as types
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class MeshPatch(BaseModel):
    name: str
    type: str  # "wall" | "patch" | "empty" | "symmetry" …
    n_cells: int = Field(alias="nCells", default=0)

    model_config = ConfigDict(populate_by_name=True)


class CheckMeshInfo(BaseModel):
    cells: int
    faces: int
    points: int
    bounding_box: dict[str, list[float]] | None = Field(alias="boundingBox", default=None)
    characteristic_length: float | None = Field(alias="characteristicLength", default=None)

    model_config = ConfigDict(populate_by_name=True)


class MeshInfo(BaseModel):
    mesh_id: str = Field(alias="meshId")
    file_name: str = Field(alias="fileName")
    patches: list[MeshPatch]
    check_mesh: CheckMeshInfo = Field(alias="checkMesh")

    model_config = ConfigDict(populate_by_name=True)


class PrecheckRequest(BaseModel):
    prompt: str = Field(default="", alias="prompt")
    has_mesh: bool = Field(default=False, alias="hasMesh")
    mesh_info: MeshInfo | None = Field(default=None, alias="meshInfo")
    mesh: MeshInfo | None = None  # legacy alias
    previous_config: dict[str, Any] | None = Field(default=None, alias="previousConfig")

    # ── Conversation support ──
    history: list[dict[str, str]] | None = Field(
        default=None,
        description="Chat history [{role, content}] for multi-turn conversation",
    )
    conversation_summary: str | None = Field(
        default=None,
        alias="conversationSummary",
        description="Summary of prior conversation when history was truncated",
    )
    confirm_analysis: bool = Field(
        default=False,
        alias="confirmAnalysis",
        description="User confirmed to start the full analysis pipeline",
    )
    simulation_context: dict[str, Any] | None = Field(
        default=None,
        alias="simulationContext",
        description="Current simulation state (physics, fluid, patches, precheckSummary) — "
                    "used to avoid re-triggering analysis when already completed",
    )

    model_config = ConfigDict(populate_by_name=True)

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
    # algorithm is derived from the selected openfoam_solver name.
    # Temporal fields (end_time, delta_t, write_interval) are extracted
    # from the prompt ("run for 2s", etc.).
    algorithm: Literal["SIMPLE", "PIMPLE", "PISO"] | None = None
    max_iterations: int | None = Field(default=None, alias="maxIterations")
    convergence_criteria: float | None = Field(default=None, alias="convergenceCriteria")
    end_time: float | None = Field(default=None, alias="endTime")
    delta_t: float | None = Field(default=None, alias="deltaT")
    write_interval: float | None = Field(default=None, alias="writeInterval")

    model_config = ConfigDict(populate_by_name=True)


class FluidProperties(BaseModel):
    preset_id: str = Field(default="air", alias="presetId")
    name: str = "Air"
    rho: float = 1.225        # kg/m³
    mu: float = 1.81e-5       # Pa·s
    Cp: float = 1006.0        # J/(kg·K)
    k: float = 0.0257         # W/(m·K)
    temperature: float = 293.15  # K

    model_config = ConfigDict(populate_by_name=True)


class TurbulenceSettings(BaseModel):
    model: Literal["kEpsilon", "kOmegaSST", "spalartAllmaras", "laminar"] = "kOmegaSST"
    # None when model == "laminar" — frontend should hide these fields in that case
    turbulence_intensity: float | None = Field(default=5.0, alias="turbulenceIntensity")   # %
    turbulence_length_scale: float | None = Field(default=0.01, alias="turbulenceLengthScale")  # m
    hydraulic_diameter: float | None = Field(default=0.1, alias="hydraulicDiameter")  # m
    wall_functions: bool = Field(default=True, alias="wallFunctions")
    # Pre-computed inlet turbulence values (None for laminar)
    k: float | None = None          # m²/s²
    omega: float | None = None      # s⁻¹
    epsilon: float | None = None    # m²/s³
    nut: float | None = None        # m²/s

    model_config = ConfigDict(populate_by_name=True)


class FieldBC(BaseModel):
    type: str
    value: float | list[float] | None = None
    entries: dict[str, Any] | None = None


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
    # Per-patch turbulence intensity (fraction, e.g. 0.05 for 5%).
    # Used by the inlet-turbulence validator to compute k = 1.5 · (U·I)² for
    # this patch specifically.  None → fall back to the global default.
    # Different inlets can carry different TIs (e.g. turbulent jet at 10%
    # mixing with a settled coflow at 1%) — they are NOT forced to match.
    turbulence_intensity: float | None = Field(
        default=None, alias="turbulenceIntensity"
    )

    model_config = ConfigDict(populate_by_name=True)


class SuggestedConfig(BaseModel):
    case_type: str = Field(default="internal_flow", alias="caseType")
    flow_regime: Literal["laminar", "turbulent"] = Field(alias="flowRegime")
    time_scheme: Literal["steady", "transient"] = Field(alias="timeScheme")
    compressibility: Literal["incompressible", "compressible"] = "incompressible"
    enable_heat_transfer: bool = Field(default=False, alias="enableHeatTransfer")
    gravity: bool = False
    # The actual OpenFOAM solver name (e.g. "rhoPimpleFoam", "interFoam").
    # Read-only info shown in the UI — does not change the run flow, but lets
    # the user see what solver will be used before they submit.
    openfoam_solver: str | None = Field(default=None, alias="openfoamSolver")
    # Set to True when the prompt suggests boiling, cavitation, evaporation, etc.
    # Triggers a warning in the UI that single-phase modelling may be insufficient.
    phase_change_detected: bool = Field(default=False, alias="phaseChangeDetected")
    # Reynolds number computed from inlet velocity, hydraulic diameter, and fluid properties.
    # Displayed read-only in the UI so the user can see the flow regime justification.
    reynolds_number: float | None = Field(default=None, alias="reynoldsNumber")
    solver: SolverSettings
    fluid: FluidProperties
    turbulence: TurbulenceSettings
    boundary_conditions: dict[str, PatchBoundaryCondition] = Field(
        default_factory=dict, alias="boundaryConditions"
    )

    model_config = ConfigDict(populate_by_name=True)


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

    model_config = ConfigDict(populate_by_name=True)


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

    model_config = ConfigDict(populate_by_name=True)


class Interpretation(BaseModel):
    summary: str
    simulation_type: str = Field(alias="simulationType")
    key_physics: list[str] = Field(default_factory=list, alias="keyPhysics")
    assumptions: list[str] = Field(default_factory=list)
    clarifications: list[str] | None = None

    model_config = ConfigDict(populate_by_name=True)


class ConfidenceScores(BaseModel):
    overall: float = Field(default=0.5, ge=0.0, le=1.0)
    flow_regime: float = Field(default=0.5, ge=0.0, le=1.0, alias="flowRegime")
    boundary_conditions: float = Field(default=0.5, ge=0.0, le=1.0, alias="boundaryConditions")
    physics_settings: float = Field(default=0.5, ge=0.0, le=1.0, alias="physicsSettings")

    model_config = ConfigDict(populate_by_name=True)


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

    model_config = ConfigDict(populate_by_name=True)


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
    "lox": FluidProperties(
        preset_id="lox", name="Liquid Oxygen (LOX)",
        rho=1141.0,   # kg/m³ at 90 K, 1 atm
        mu=1.95e-4,   # Pa·s at 90 K
        Cp=1699.0,    # J/(kg·K)
        k=0.152,      # W/(m·K)
        temperature=90.0,
    ),
    "lh2": FluidProperties(
        preset_id="lh2", name="Liquid Hydrogen (LH2)",
        rho=70.8,     # kg/m³ at 20 K, 1 atm
        mu=1.33e-5,   # Pa·s at 20 K
        Cp=9668.0,    # J/(kg·K)
        k=0.099,      # W/(m·K)
        temperature=20.0,
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

PRECHECK_MODEL     = "gemini-3.1-pro-preview"   # gemini-3-flash-preview -  gemini-3.1-pro-preview
REVIEW_MODEL       = "gemini-3-flash-preview"   # deep review + corrections
CONVERSATION_MODEL = "gemini-3-flash-preview"   # lighter model for chat + summarization

# Token threshold: when conversation history exceeds this, summarize with the
# lighter model so the context window stays manageable.
CONVERSATION_TOKEN_LIMIT = 30_000

# Base path for bc_knowledge markdown files (resolved relative to this file)
BC_KNOWLEDGE_DIR: _pathlib.Path = _pathlib.Path(__file__).parent / "bc_knowledge"

# ---------------------------------------------------------------------------
# Conversation tool — signals the LLM has enough info to start analysis
# ---------------------------------------------------------------------------

READY_TOOL_SCHEMA = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="signal_ready_to_analyze",
            description=(
                "Call this ONLY when you have gathered enough information from the "
                "conversation to configure the CFD simulation AND the user has uploaded "
                "a mesh file. Do NOT call if no mesh is uploaded."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "summary": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Concise summary of the simulation the user wants, including: "
                            "fluid type, flow regime, boundary conditions, and any physics "
                            "details gathered from the conversation."
                        ),
                    ),
                },
                required=["summary"],
            ),
        ),
    ],
)

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
                    # Only temporal user-intent parameters are collected here.
                    "solver_end_time": types.Schema(type="NUMBER", nullable=True,
                        description="Physical end time in seconds extracted from the prompt ('run for 2s' → 2.0). Null for steady simulations."),
                    "solver_delta_t": types.Schema(type="NUMBER", nullable=True,
                        description="Time step in seconds extracted from the prompt. Null if not specified."),
                    "solver_max_iterations": types.Schema(type="INTEGER", nullable=True,
                        description="Number of iterations extracted from the prompt ('run for 5000 iterations' → 5000). Null if not specified. Only for steady-state."),
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
                                        "wall → 'fixedValue' only when user specifies a concrete wall temperature; "
                                        "otherwise 'zeroGradient' (adiabatic — no heat flux). "
                                        "NEVER set wall T_type to 'fixedValue' when no numeric temperature is given. "
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
                    # solver_end_time is required so the LLM always states the
                    # requested physical duration (null for steady cases).
                    "solver_end_time",
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


# ---------------------------------------------------------------------------
# Boundary-planner tool schema  (new multi-pass architecture)
# ---------------------------------------------------------------------------

# fieldStrategy sub-schema (reused inside patch items)
_FIELD_STRATEGY_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "status": types.Schema(
            type="STRING",
            enum=["selected", "not_needed", "unknown", "defer_to_solver_default"],
        ),
        "selectionMode": types.Schema(
            type="STRING",
            enum=["direct_user_value", "derived_bc_family", "solver_default", "review_required"],
        ),
        "bcFamily":      types.Schema(type="STRING", nullable=True),
        "uiPrimaryInput": types.Schema(type="STRING", nullable=True),
        "reason":        types.Schema(type="STRING", nullable=True),
    },
    required=["status", "selectionMode"],
)

BOUNDARY_PLAN_TOOL_SCHEMA = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="submit_boundary_plan",
            description=(
                "Submit the per-patch boundary plan. "
                "One item per mesh patch. "
                "Decide roles, physical requirements, field strategies, and retrieval docs. "
                "Also declare the global time_scheme and end_time for the simulation."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "time_scheme": types.Schema(
                        type="STRING",
                        enum=["steady", "transient"],
                        description=(
                            "REQUIRED. Default: 'steady'. "
                            "Use 'steady' when the user gives no indication of time evolution, "
                            "or explicitly requests steady-state. Most simple flow queries "
                            "('simulate water in a pipe', 'airflow at 1 m/s') are steady. "
                            "Use 'transient' ONLY when the user specifies a duration ('for 2s', "
                            "'10 seconds'), uses transient keywords ('unsteady', 'transient', "
                            "'pulsating', 'oscillating', 'vortex shedding'), or asks about "
                            "time evolution ('how the flow develops', 'startup')."
                        ),
                    ),
                    "end_time": types.Schema(
                        type="NUMBER",
                        nullable=True,
                        description=(
                            "Physical simulation end time in seconds. "
                            "When time_scheme='transient': extract from the user request if stated "
                            "(e.g. '2s' → 2.0, '0.5 seconds' → 0.5); if not stated, leave null "
                            "(the system will default to 5.0 s). Null for steady simulations. "
                            "Note: end times above 10 s are automatically capped to 10 s."
                        ),
                    ),
                    "max_iterations": types.Schema(
                        type="INTEGER",
                        nullable=True,
                        description=(
                            "Number of solver iterations for steady-state simulations. "
                            "Extract from the user request if stated (e.g. 'run for 5000 iterations' → 5000). "
                            "Null if not specified (system defaults to 1000). "
                            "Only relevant when time_scheme='steady'."
                        ),
                    ),
                    "patches": types.Schema(
                        type="ARRAY",
                        items=types.Schema(
                            type="OBJECT",
                            properties={
                                "patchName":     types.Schema(type="STRING"),
                                "meshPatchType": types.Schema(type="STRING"),
                                "patchRole": types.Schema(
                                    type="STRING",
                                    enum=["inlet", "outlet", "wall", "frontAndBack", "symmetry", "other"],
                                ),
                                "confidence": types.Schema(type="NUMBER"),
                                # --- detected physical requirements ---
                                "hasMassFlowRate":       types.Schema(type="BOOLEAN"),
                                "massFlowRate":          types.Schema(type="NUMBER", nullable=True),
                                "massFlowRateUnit":      types.Schema(type="STRING", nullable=True),
                                "hasVolumetricFlowRate": types.Schema(type="BOOLEAN"),
                                "volumetricFlowRate":    types.Schema(type="NUMBER", nullable=True),
                                "hasVelocity":           types.Schema(
                                    type="BOOLEAN",
                                    description=(
                                        "True when the user specifies an inlet velocity value "
                                        "(e.g. '30 m/s', 'velocity 5', 'U = 10 m/s'). "
                                        "False when mass flow rate or volumetric flow rate is "
                                        "the primary driver."
                                    ),
                                ),
                                "velocityMagnitude":     types.Schema(
                                    type="NUMBER", nullable=True,
                                    description=(
                                        "Inlet velocity magnitude in m/s extracted from the user "
                                        "prompt (e.g. '30 m/s' → 30.0). Null when velocity is "
                                        "not specified or when mass/vol flow rate is used instead."
                                    ),
                                ),
                                "velocityVector":        types.Schema(
                                    type="ARRAY", nullable=True,
                                    items=types.Schema(type="NUMBER"),
                                    description=(
                                        "Inlet velocity vector [Ux, Uy, Uz] in m/s if the user "
                                        "specifies direction. Null if only magnitude is given "
                                        "(direction will be inferred from mesh patch normal)."
                                    ),
                                ),
                                "hasStaticPressure":     types.Schema(type="BOOLEAN"),
                                "staticPressure":        types.Schema(type="NUMBER", nullable=True),
                                "hasTotalPressure":      types.Schema(type="BOOLEAN"),
                                "totalPressure":         types.Schema(type="NUMBER", nullable=True),
                                "hasStaticTemperature":  types.Schema(
                                    type="BOOLEAN",
                                    description=(
                                        "True ONLY when the user EXPLICITLY requests heat transfer, "
                                        "a specific temperature value, heated/cooled walls, or thermal "
                                        "analysis. False for isothermal or simple flow simulations."
                                    ),
                                ),
                                "staticTemperature":     types.Schema(
                                    type="NUMBER", nullable=True,
                                    description=(
                                        "Temperature in Kelvin. Set ONLY when the user provides a "
                                        "specific numeric temperature. null otherwise."
                                    ),
                                ),
                                "hasTotalTemperature":   types.Schema(type="BOOLEAN"),
                                "totalTemperature":      types.Schema(type="NUMBER", nullable=True),
                                "hasNoSlip":             types.Schema(type="BOOLEAN"),
                                "hasTurbulenceIntensity": types.Schema(type="BOOLEAN"),
                                "turbulenceIntensity":   types.Schema(type="NUMBER", nullable=True),
                                # --- field strategies ---
                                "strategyU":       _FIELD_STRATEGY_SCHEMA,
                                "strategyP":       _FIELD_STRATEGY_SCHEMA,
                                "strategyT":       _FIELD_STRATEGY_SCHEMA,
                                "strategyK":       _FIELD_STRATEGY_SCHEMA,
                                "strategyEpsilon": _FIELD_STRATEGY_SCHEMA,
                                "strategyOmega":   _FIELD_STRATEGY_SCHEMA,
                                # --- retrieval ---
                                "retrievalDocs": types.Schema(
                                    type="ARRAY",
                                    description="Relative paths to bc_knowledge MD files to load for this patch.",
                                    items=types.Schema(type="STRING"),
                                ),
                                "assumptions": types.Schema(
                                    type="ARRAY", items=types.Schema(type="STRING"),
                                ),
                                "blockingIssues": types.Schema(
                                    type="ARRAY", items=types.Schema(type="STRING"),
                                ),
                            },
                            required=["patchName", "meshPatchType", "patchRole", "confidence",
                                      "retrievalDocs"],
                        ),
                    ),
                },
                required=["time_scheme", "patches"],
            ),
        )
    ]
)


# ---------------------------------------------------------------------------
# Patch-spec tool schema  (one call per patch)
# ---------------------------------------------------------------------------

# foamFieldSpec sub-schema
_FOAM_FIELD_SPEC_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "enabled":         types.Schema(type="BOOLEAN"),
        "bcType":          types.Schema(type="STRING"),
        "selectionReason": types.Schema(type="STRING", nullable=True),
        # flat key-value entries (string-encoded numbers/vectors for flexibility)
        "entryMassFlowRate":    types.Schema(type="NUMBER", nullable=True),
        "entryVolumetricFlowRate": types.Schema(type="NUMBER", nullable=True),
        "entryIntensity":       types.Schema(type="NUMBER", nullable=True),
        "entryValue":           types.Schema(type="NUMBER", nullable=True),
        "entryValueVector":     types.Schema(
            type="ARRAY", nullable=True, items=types.Schema(type="NUMBER"),
        ),
        "entryP0":              types.Schema(type="NUMBER", nullable=True),
        "entryT0":              types.Schema(type="NUMBER", nullable=True),
    },
    required=["enabled", "bcType"],
)

# uiInput sub-schema
_UI_INPUT_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "key":      types.Schema(type="STRING"),
        "label":    types.Schema(type="STRING"),
        "kind":     types.Schema(type="STRING", enum=["number", "select", "toggle", "vector", "text"]),
        "required": types.Schema(type="BOOLEAN"),
        "unit":     types.Schema(type="STRING", nullable=True),
        "mapsToField": types.Schema(type="STRING"),
        "mapsToEntry": types.Schema(type="STRING"),
        "helpText": types.Schema(type="STRING", nullable=True),
    },
    required=["key", "label", "kind", "required", "mapsToField", "mapsToEntry"],
)

# uiDerived sub-schema
_UI_DERIVED_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "key":         types.Schema(type="STRING"),
        "label":       types.Schema(type="STRING"),
        "valueSource": types.Schema(type="STRING"),
        "unit":        types.Schema(type="STRING", nullable=True),
        "uiOnly":      types.Schema(type="BOOLEAN"),
    },
    required=["key", "label", "valueSource", "uiOnly"],
)

PATCH_SPEC_TOOL_SCHEMA = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="submit_patch_spec",
            description=(
                "Submit the fully resolved boundary-condition spec for ONE patch. "
                "Include foamFields for all active fields, uiModel inputs and derived display items, "
                "plus any assumptions or warnings."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "patchName": types.Schema(type="STRING"),
                    "patchRole": types.Schema(
                        type="STRING",
                        enum=["inlet", "outlet", "wall", "frontAndBack", "symmetry", "other"],
                    ),
                    "confidence": types.Schema(type="NUMBER"),
                    # foam fields
                    "fieldU":       _FOAM_FIELD_SPEC_SCHEMA,
                    "fieldP":       _FOAM_FIELD_SPEC_SCHEMA,
                    "fieldT":       _FOAM_FIELD_SPEC_SCHEMA,
                    "fieldK":       _FOAM_FIELD_SPEC_SCHEMA,
                    "fieldEpsilon": _FOAM_FIELD_SPEC_SCHEMA,
                    "fieldOmega":   _FOAM_FIELD_SPEC_SCHEMA,
                    "fieldNut":     _FOAM_FIELD_SPEC_SCHEMA,
                    # UI model
                    "uiInputs": types.Schema(
                        type="ARRAY", items=_UI_INPUT_SCHEMA,
                        description="User-editable input fields for this patch.",
                    ),
                    "uiDerived": types.Schema(
                        type="ARRAY", items=_UI_DERIVED_SCHEMA,
                        description="Derived display values (uiOnly=true means not a real BC entry).",
                    ),
                    "assumptions": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                    "warnings":    types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                },
                required=["patchName", "patchRole", "confidence"],
            ),
        )
    ]
)
