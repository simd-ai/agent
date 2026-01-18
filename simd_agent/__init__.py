# simd_agent/__init__.py
"""
SIMD Agent - FastAPI service for orchestrating CFD workflows via WebSockets.

This service provides:
- CFD configuration linting and validation
- OpenFOAM case generation using codegen
- Self-healing execution loop with remote sandbox
- Real-time progress streaming via WebSocket events
"""

__version__ = "0.1.0"

from simd_agent.models import StartRequest, AgentEvent
from simd_agent.orchestration import Orchestrator
from simd_agent.linting import CFDLinter

__all__ = [
    "StartRequest",
    "AgentEvent", 
    "Orchestrator",
    "CFDLinter",
]
