# simd_agent/precheck.py
"""Precheck service for analyzing user prompts and extracting simulation specifications."""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from simd_agent.settings import get_settings

load_dotenv()

logger = logging.getLogger(__name__)


# --- Request Models ---

class MeshPatch(BaseModel):
    """Mesh patch information."""
    name: str
    type: str  # e.g., "wall", "patch", "empty"
    nCells: int


class CheckMeshInfo(BaseModel):
    """checkMesh output information."""
    cells: int
    faces: int
    points: int
    boundingBox: dict[str, list[float]] | None = None
    characteristicLength: float | None = None


class MeshInfo(BaseModel):
    """Uploaded mesh information."""
    meshId: str
    fileName: str
    patches: list[MeshPatch]
    checkMesh: CheckMeshInfo


class PrecheckRequest(BaseModel):
    """Request for precheck analysis."""
    prompt: str = Field(..., min_length=1, description="Natural language simulation description")
    mesh: MeshInfo | None = None
    previousConfig: dict[str, Any] | None = None


# --- Response Models ---

class FluidProperties(BaseModel):
    """Fluid properties."""
    name: str  # e.g., "air", "water", "custom"
    density: float | None = None  # kg/m³
    viscosity: float | None = None  # Pa·s
    specificHeat: float | None = None
    thermalConductivity: float | None = None


class SuggestedConfig(BaseModel):
    """Suggested simulation configuration."""
    flowRegime: str = Field(..., pattern="^(laminar|turbulent)$")
    timeScheme: str = Field(..., pattern="^(steady|transient)$")
    compressibility: str = Field(default="incompressible", pattern="^(incompressible|compressible)$")
    enableHeatTransfer: bool = False
    turbulenceModel: str | None = None  # kEpsilon, kOmegaSST, spalartAllmaras
    maxIterations: int = 1000
    convergenceCriteria: float = 1e-6
    endTime: float | None = None  # For transient
    deltaT: float | None = None  # For transient
    fluid: FluidProperties | None = None
    presetId: str | None = None


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
    """Suggested boundary condition for a patch."""
    suggestedType: str = Field(..., pattern="^(inlet|outlet|wall|symmetry|periodic)$")
    velocity: VelocityBC | None = None
    pressure: PressureBC | None = None
    temperature: TemperatureBC | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""


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
    pressureDrop: KPIValue | None = None
    flowRate: KPIValue | None = None
    temperature: KPIValue | None = None
    velocity: KPIValue | None = None
    custom: list[CustomKPI] = Field(default_factory=list)


class Interpretation(BaseModel):
    """LLM's understanding of the prompt."""
    summary: str
    simulationType: str  # e.g., "Internal pipe flow"
    keyPhysics: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    clarifications: list[str] | None = None


class ConfidenceScores(BaseModel):
    """Confidence scores for various aspects."""
    overall: float = Field(default=0.5, ge=0.0, le=1.0)
    flowRegime: float = Field(default=0.5, ge=0.0, le=1.0)
    boundaryConditions: float = Field(default=0.5, ge=0.0, le=1.0)
    physicsSettings: float = Field(default=0.5, ge=0.0, le=1.0)


class PrecheckResponse(BaseModel):
    """Response from precheck analysis."""
    success: bool
    suggestedConfig: SuggestedConfig
    boundaryHints: dict[str, BoundaryHint] | None = None
    kpiTargets: KPITargets | None = None
    interpretation: Interpretation
    confidence: ConfidenceScores
    nextStep: int = Field(default=1, ge=1, le=3)
    shouldShowMeshViewer: bool = False
    warnings: list[str] | None = None
    errors: list[str] | None = None


# --- Precheck Service ---

class PrecheckService:
    """Service for analyzing prompts and extracting simulation specs."""
    
    def __init__(self):
        self.settings = get_settings()
        self._generator = None
    
    @property
    def generator(self):
        """Lazy-load the code generator."""
        if self._generator is None:
            from codegen import CodeGenerator
            self._generator = CodeGenerator(
                provider=self.settings.default_provider,
                prompt_pack="default",
            )
        return self._generator
    
    async def analyze(self, request: PrecheckRequest) -> PrecheckResponse:
        """Analyze a user prompt and extract simulation specifications."""
        try:
            # Build the analysis prompt
            analysis_prompt = self._build_analysis_prompt(request)
            
            # Call the LLM
            from codegen import GenerationContext
            
            # Use "codegen" task type (the only valid one for analysis)
            # The actual precheck instructions are in the requirements
            context = GenerationContext(
                task="codegen",  # Must use valid TaskType
                domain="cfd_analysis",
                requirements=analysis_prompt,
                extra_context={
                    "output_format": "json",
                    "schema": "PrecheckResponse",
                    "task_type": "precheck",  # Our custom indicator
                },
            )
            
            result = self.generator.generate(context)
            
            # Parse the LLM response
            response = self._parse_llm_response(result.final_text, request)
            return response
            
        except Exception as e:
            logger.exception(f"Precheck analysis failed: {e}")
            # Return a fallback response
            return self._create_fallback_response(request, str(e))
    
    def _build_analysis_prompt(self, request: PrecheckRequest) -> str:
        """Build the prompt for LLM analysis."""
        prompt_parts = [
            "Analyze the following CFD simulation request and extract structured parameters.",
            "",
            f"User's description: {request.prompt}",
            "",
        ]
        
        # Add mesh info if available
        if request.mesh:
            prompt_parts.extend([
                "Uploaded mesh information:",
                f"  - File: {request.mesh.fileName}",
                f"  - Cells: {request.mesh.checkMesh.cells:,}",
                f"  - Faces: {request.mesh.checkMesh.faces:,}",
                f"  - Points: {request.mesh.checkMesh.points:,}",
                "",
                "Mesh patches:",
            ])
            for patch in request.mesh.patches:
                prompt_parts.append(f"  - {patch.name} (type: {patch.type}, cells: {patch.nCells})")
            prompt_parts.append("")
        
        # Add previous config if refining
        if request.previousConfig:
            prompt_parts.extend([
                "Previous configuration (user is refining):",
                json.dumps(request.previousConfig, indent=2),
                "",
            ])
        
        prompt_parts.extend([
            "Based on the above, provide a JSON response with:",
            "1. suggestedConfig: Physics and solver settings",
            "2. boundaryHints: Suggested BCs for each mesh patch (if mesh provided)",
            "3. kpiTargets: Any quantitative targets mentioned",
            "4. interpretation: Your understanding of the simulation",
            "5. confidence: How confident you are in each aspect (0-1)",
            "",
            "Use these heuristics:",
            "- 'pipe flow', 'duct', 'channel' → internal flow",
            "- 'wind', 'aerodynamics', 'vehicle', 'external' → external flow",
            "- High Re (>4000), 'industrial', 'turbulent' → turbulent flow, use kEpsilon or kOmegaSST",
            "- Low Re (<2300), 'laminar', 'creeping', 'slow' → laminar flow",
            "- 'heat', 'thermal', 'temperature', 'cooling', 'heating' → enable heat transfer",
            "- 'pulsating', 'oscillating', 'time-varying', 'unsteady' → transient simulation",
            "- 'steady state', 'converged', 'equilibrium' → steady simulation",
            "",
            "For boundary conditions, use patch naming conventions:",
            "- 'inlet', 'inflow', 'in' → inlet BC",
            "- 'outlet', 'outflow', 'out', 'exit' → outlet BC",
            "- 'wall', 'walls', 'surface' → wall BC (no-slip)",
            "- 'symmetry', 'sym' → symmetry BC",
            "- 'periodic', 'cyclic' → periodic BC",
            "",
            "IMPORTANT: Respond with ONLY valid JSON, no markdown code blocks.",
            "",
            "JSON Schema:",
            """{
  "suggestedConfig": {
    "flowRegime": "laminar" | "turbulent",
    "timeScheme": "steady" | "transient",
    "compressibility": "incompressible" | "compressible",
    "enableHeatTransfer": boolean,
    "turbulenceModel": "kEpsilon" | "kOmegaSST" | "spalartAllmaras" | null,
    "maxIterations": number,
    "convergenceCriteria": number,
    "endTime": number | null,
    "deltaT": number | null,
    "fluid": { "name": string, "density": number, "viscosity": number } | null,
    "presetId": string | null
  },
  "boundaryHints": {
    "<patchName>": {
      "suggestedType": "inlet" | "outlet" | "wall" | "symmetry" | "periodic",
      "velocity": { "type": string, "value": [x, y, z], "magnitude": number } | null,
      "pressure": { "type": string, "value": number } | null,
      "temperature": { "type": string, "value": number } | null,
      "confidence": number,
      "reasoning": string
    }
  },
  "kpiTargets": {
    "pressureDrop": { "value": number, "unit": string } | null,
    "flowRate": { "value": number, "unit": string } | null,
    "temperature": { "value": number, "unit": string } | null,
    "velocity": { "value": number, "unit": string } | null,
    "custom": [{ "name": string, "value": number, "unit": string }]
  },
  "interpretation": {
    "summary": string,
    "simulationType": string,
    "keyPhysics": [string],
    "assumptions": [string],
    "clarifications": [string] | null
  },
  "confidence": {
    "overall": number,
    "flowRegime": number,
    "boundaryConditions": number,
    "physicsSettings": number
  }
}""",
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, llm_output: str, request: PrecheckRequest) -> PrecheckResponse:
        """Parse the LLM response into a structured PrecheckResponse."""
        try:
            # Try to extract JSON from the response
            json_str = self._extract_json(llm_output)
            data = json.loads(json_str)
            
            # Build the response
            suggested_config = SuggestedConfig(
                flowRegime=data.get("suggestedConfig", {}).get("flowRegime", "turbulent"),
                timeScheme=data.get("suggestedConfig", {}).get("timeScheme", "steady"),
                compressibility=data.get("suggestedConfig", {}).get("compressibility", "incompressible"),
                enableHeatTransfer=data.get("suggestedConfig", {}).get("enableHeatTransfer", False),
                turbulenceModel=data.get("suggestedConfig", {}).get("turbulenceModel"),
                maxIterations=data.get("suggestedConfig", {}).get("maxIterations", 1000),
                convergenceCriteria=data.get("suggestedConfig", {}).get("convergenceCriteria", 1e-6),
                endTime=data.get("suggestedConfig", {}).get("endTime"),
                deltaT=data.get("suggestedConfig", {}).get("deltaT"),
                fluid=self._parse_fluid(data.get("suggestedConfig", {}).get("fluid")),
                presetId=data.get("suggestedConfig", {}).get("presetId"),
            )
            
            # Parse boundary hints
            boundary_hints = None
            if "boundaryHints" in data and data["boundaryHints"]:
                boundary_hints = {}
                for patch_name, hint_data in data["boundaryHints"].items():
                    boundary_hints[patch_name] = BoundaryHint(
                        suggestedType=hint_data.get("suggestedType", "wall"),
                        velocity=self._parse_velocity_bc(hint_data.get("velocity")),
                        pressure=self._parse_pressure_bc(hint_data.get("pressure")),
                        temperature=self._parse_temperature_bc(hint_data.get("temperature")),
                        confidence=hint_data.get("confidence", 0.5),
                        reasoning=hint_data.get("reasoning", ""),
                    )
            
            # Parse KPI targets
            kpi_targets = None
            if "kpiTargets" in data and data["kpiTargets"]:
                kpi_data = data["kpiTargets"]
                kpi_targets = KPITargets(
                    pressureDrop=self._parse_kpi_value(kpi_data.get("pressureDrop")),
                    flowRate=self._parse_kpi_value(kpi_data.get("flowRate")),
                    temperature=self._parse_kpi_value(kpi_data.get("temperature")),
                    velocity=self._parse_kpi_value(kpi_data.get("velocity")),
                    custom=[
                        CustomKPI(**c) for c in kpi_data.get("custom", [])
                    ],
                )
            
            # Parse interpretation
            interp_data = data.get("interpretation", {})
            interpretation = Interpretation(
                summary=interp_data.get("summary", "CFD simulation analysis"),
                simulationType=interp_data.get("simulationType", "General CFD"),
                keyPhysics=interp_data.get("keyPhysics", []),
                assumptions=interp_data.get("assumptions", []),
                clarifications=interp_data.get("clarifications"),
            )
            
            # Parse confidence
            conf_data = data.get("confidence", {})
            confidence = ConfidenceScores(
                overall=conf_data.get("overall", 0.7),
                flowRegime=conf_data.get("flowRegime", 0.7),
                boundaryConditions=conf_data.get("boundaryConditions", 0.5),
                physicsSettings=conf_data.get("physicsSettings", 0.7),
            )
            
            # Determine next step
            next_step = 1
            should_show_mesh = False
            if request.mesh:
                next_step = 2  # Go to boundary conditions
                should_show_mesh = True
            elif boundary_hints:
                next_step = 2
            
            return PrecheckResponse(
                success=True,
                suggestedConfig=suggested_config,
                boundaryHints=boundary_hints,
                kpiTargets=kpi_targets,
                interpretation=interpretation,
                confidence=confidence,
                nextStep=next_step,
                shouldShowMeshViewer=should_show_mesh,
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return self._create_fallback_response(request, f"JSON parse error: {e}")
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._create_fallback_response(request, str(e))
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        # Remove markdown code blocks if present
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        
        # Try to find JSON object
        brace_start = text.find("{")
        if brace_start != -1:
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(text[brace_start:], brace_start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[brace_start:i + 1]
        
        return text.strip()
    
    def _parse_fluid(self, data: dict | None) -> FluidProperties | None:
        """Parse fluid properties."""
        if not data:
            return None
        return FluidProperties(
            name=data.get("name", "air"),
            density=data.get("density"),
            viscosity=data.get("viscosity"),
            specificHeat=data.get("specificHeat"),
            thermalConductivity=data.get("thermalConductivity"),
        )
    
    def _parse_velocity_bc(self, data: dict | None) -> VelocityBC | None:
        """Parse velocity BC."""
        if not data:
            return None
        return VelocityBC(
            type=data.get("type", "fixedValue"),
            value=data.get("value"),
            magnitude=data.get("magnitude"),
        )
    
    def _parse_pressure_bc(self, data: dict | None) -> PressureBC | None:
        """Parse pressure BC."""
        if not data:
            return None
        return PressureBC(
            type=data.get("type", "fixedValue"),
            value=data.get("value"),
        )
    
    def _parse_temperature_bc(self, data: dict | None) -> TemperatureBC | None:
        """Parse temperature BC."""
        if not data:
            return None
        return TemperatureBC(
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
    
    def _create_fallback_response(self, request: PrecheckRequest, error: str) -> PrecheckResponse:
        """Create a fallback response when LLM fails."""
        # Use heuristics to provide reasonable defaults
        prompt_lower = request.prompt.lower()
        
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
        
        # Detect heat transfer
        has_heat = any(word in prompt_lower for word in [
            "heat", "thermal", "temperature", "cooling", "heating", "hot", "cold"
        ])
        
        # Detect simulation type
        is_internal = any(word in prompt_lower for word in [
            "pipe", "duct", "channel", "internal", "tube"
        ])
        is_external = any(word in prompt_lower for word in [
            "external", "wind", "aerodynamic", "vehicle", "airfoil", "wing"
        ])
        sim_type = "Internal flow" if is_internal else "External flow" if is_external else "General CFD"
        
        # Build boundary hints from mesh patches
        boundary_hints = None
        if request.mesh:
            boundary_hints = {}
            for patch in request.mesh.patches:
                patch_lower = patch.name.lower()
                if any(x in patch_lower for x in ["inlet", "inflow", "in"]):
                    bc_type = "inlet"
                elif any(x in patch_lower for x in ["outlet", "outflow", "out", "exit"]):
                    bc_type = "outlet"
                elif any(x in patch_lower for x in ["wall", "surface"]):
                    bc_type = "wall"
                elif any(x in patch_lower for x in ["sym", "symmetry"]):
                    bc_type = "symmetry"
                else:
                    bc_type = "wall"  # Default
                
                boundary_hints[patch.name] = BoundaryHint(
                    suggestedType=bc_type,
                    confidence=0.6,
                    reasoning=f"Inferred from patch name '{patch.name}'",
                )
        
        return PrecheckResponse(
            success=False,
            suggestedConfig=SuggestedConfig(
                flowRegime=flow_regime,
                timeScheme=time_scheme,
                compressibility="incompressible",
                enableHeatTransfer=has_heat,
                turbulenceModel="kEpsilon" if flow_regime == "turbulent" else None,
                maxIterations=1000,
                convergenceCriteria=1e-6,
            ),
            boundaryHints=boundary_hints,
            interpretation=Interpretation(
                summary=f"Fallback analysis (LLM error: {error})",
                simulationType=sim_type,
                keyPhysics=["turbulence"] if flow_regime == "turbulent" else [],
                assumptions=["Using heuristic-based defaults due to LLM error"],
                clarifications=["Please review and adjust settings manually"],
            ),
            confidence=ConfidenceScores(
                overall=0.3,
                flowRegime=0.5,
                boundaryConditions=0.3,
                physicsSettings=0.4,
            ),
            nextStep=1,
            shouldShowMeshViewer=request.mesh is not None,
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
