# simd_agent/precheck/__init__.py
"""Precheck package — LLM-based simulation spec extraction from natural language."""

from simd_agent.precheck.models import (
    BC_KNOWLEDGE_DIR,
    BOUNDARY_PLAN_TOOL_SCHEMA,
    CRYOGENIC_KEYWORDS,
    FLUID_PRESETS,
    PATCH_SPEC_TOOL_SCHEMA,
    BoundaryHint,
    ConfidenceScores,
    FieldBC,
    FluidProperties,
    Interpretation,
    PatchBoundaryCondition,
    PrecheckRequest,
    PrecheckResponse,
    PressureBC,
    SolverSettings,
    SuggestedConfig,
    TemperatureBC,
    TurbulenceSettings,
    VelocityBC,
)
from simd_agent.precheck.service import PrecheckService, get_precheck_service

__all__ = [
    "BC_KNOWLEDGE_DIR",
    "BOUNDARY_PLAN_TOOL_SCHEMA",
    "CRYOGENIC_KEYWORDS",
    "FLUID_PRESETS",
    "PATCH_SPEC_TOOL_SCHEMA",
    "BoundaryHint",
    "ConfidenceScores",
    "FieldBC",
    "FluidProperties",
    "Interpretation",
    "PatchBoundaryCondition",
    "PrecheckRequest",
    "PrecheckResponse",
    "PrecheckService",
    "PressureBC",
    "SolverSettings",
    "SuggestedConfig",
    "TemperatureBC",
    "TurbulenceSettings",
    "VelocityBC",
    "get_precheck_service",
]
