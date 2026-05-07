# tests/test_integration_real.py
"""
Real end-to-end integration tests with actual LLM + simulation server.

These tests:
1. Generate OpenFOAM code using real AI (Gemini)
2. Submit to the simulation server for execution
3. Wait for results and report success/failure

These tests are NOT for CI/CD - they:
   - Call real LLM APIs (costs money)
   - Submit to real simulation server (uses compute resources)
   - Take significant time (minutes per test)

Usage:
    export GEMINI_API_KEY=your-key-here
    export DATABASE_URL=postgresql://...  # or skip for in-memory testing

    pytest tests/test_integration_real.py -v -s
"""

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# Configuration
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

# Skip if no API key
pytestmark = [
    pytest.mark.skipif(
        not GEMINI_KEY,
        reason="GEMINI_API_KEY not set - skipping real integration tests"
    ),
    pytest.mark.integration,
    pytest.mark.timeout(600),  # 10 minute timeout per test
]


class EventCapture:
    """Helper to capture and display WebSocket events."""
    
    def __init__(self):
        self.events = []
        self.start_time = datetime.utcnow()
    
    async def send_json(self, data):
        """Capture event and display progress."""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        self.events.append(data)
        
        event_type = data.get('type', 'unknown')
        message = data.get('message', '')
        level = data.get('level', 'info')
        
        # Color coding
        colors = {
            'debug': '\033[90m',    # Gray
            'info': '\033[94m',     # Blue
            'warn': '\033[93m',     # Yellow
            'error': '\033[91m',    # Red
        }
        reset = '\033[0m'
        color = colors.get(level, '')
        
        # Icons for event types
        icons = {
            'run_started': '🚀',
            'lint_started': '🔍',
            'lint_result': '✅',
            'planning_started': '📋',
            'planning_complete': '📝',
            'subagent_started': '⚡',
            'subagent_update': '💬',
            'subagent_complete': '✓',
            'codegen_started': '🤖',
            'codegen_iteration': '📄',
            'codegen_complete': '📦',
            'error_summary': '🔎',
            'retrying': '🔄',
            'run_succeeded': '✅',
            'run_failed': '❌',
            'final': '🏁',
        }
        icon = icons.get(event_type, '•')
        
        print(f"{color}[{elapsed:6.1f}s] {icon} {event_type}: {message}{reset}")
        
        # Show important payload details
        payload = data.get('payload', {})
        if event_type == 'lint_result':
            print(f"         Regime: {payload.get('regime')} | Re: {payload.get('reynolds_number', 'N/A')}")
        elif event_type == 'error_summary':
            print(f"         Root cause: {payload.get('root_cause', 'Unknown')}")
        elif event_type == 'final':
            print(f"         Status: {payload.get('status')} | Iterations: {payload.get('iterations')}")


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket with event capture."""
    capture = EventCapture()
    ws = MagicMock()
    ws.send_json = AsyncMock(side_effect=capture.send_json)
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    ws.events = capture.events
    ws.capture = capture
    return ws


@pytest.fixture
def mock_store():
    """Create a mock store that doesn't need a real database."""
    store = MagicMock()
    store.create_run = AsyncMock()
    store.append_event = AsyncMock()
    store.finalize_run = AsyncMock()
    store.update_run_status = AsyncMock()
    return store


class TestFullIntegration:
    """Full end-to-end integration tests."""

    async def test_codegen_only(self, mock_websocket, mock_store):
        """
        Test code generation only without simulation execution.

        Useful for testing AI output quality without waiting for execution.
        """
        from simd_agent.event_bus import EventBus
        from simd_agent.linting import CFDLinter
        from simd_agent.planning import Planner
        from simd_agent.packaging import extract_file_blocks, package_from_llm_output
        
        print("\n" + "="*70)
        print("🧪 CODE GENERATION TEST")
        print("="*70 + "\n")
        
        run_id = uuid4()
        event_bus = EventBus(run_id, mock_websocket, mock_store, persist=False)
        
        config = {
            "geometry": {
                "type": "pipe",
                "diameter": 0.1,
                "length": 1.0,
            },
            "inlet": {"velocity": 2.0},
            "fluid": {"viscosity": 1e-6, "density": 1000.0},
        }
        
        requirements = """
        Generate a turbulent pipe flow simulation.
        Reynolds number will be around 200,000.
        Use k-epsilon turbulence model.
        Run for 500 iterations.
        """
        
        # Lint
        print("🔍 Linting...")
        linter = CFDLinter(event_bus=event_bus)
        lint_result = await linter.lint(config, requirements)
        
        print(f"   Case Type: {lint_result.detected_case_type}")
        print(f"   Reynolds: {lint_result.reynolds_number:.0f}")
        print(f"   Regime: {lint_result.detected_regime}")
        
        # Plan
        print("\n📋 Planning...")
        planner = Planner(event_bus)
        planning_result = await planner.plan(lint_result, requirements, lint_result.validated_config)
        
        print(f"   Solver: {planning_result.solver}")
        print(f"   Turbulence: {planning_result.turbulence_model}")
        
        # Generate code
        print("\n🤖 Generating code with Gemini...")
        
        try:
            from codegen import CodeGenerator, GenerationContext
            
            generator = CodeGenerator(provider="gemini3", prompt_pack="simd")
            
            context = GenerationContext(
                task="codegen",
                domain="openfoam_case",
                requirements=requirements,
                extra_context={
                    "validated_config": lint_result.validated_config,
                    "solver": planning_result.solver,
                    "turbulence_model": planning_result.turbulence_model,
                    "mesh_strategy": planning_result.mesh_strategy,
                    "case_type": planning_result.case_type,
                },
            )
            
            result = generator.generate(context)  # Not async
            llm_output = result.final_text
            
        except ImportError:
            print("   ⚠️  codegen not available, skipping LLM call")
            pytest.skip("codegen not installed")
        
        # Extract files
        files = extract_file_blocks(llm_output)
        
        print(f"\n📄 Generated {len(files)} files:")
        for path, content in sorted(files.items()):
            print(f"   {path} ({len(content)} chars)")
        
        # Validate structure
        assert len(files) >= 5, f"Should generate at least 5 files, got {len(files)}"
        
        required = ["controlDict", "fvSchemes", "fvSolution"]
        for req in required:
            assert any(req in f for f in files), f"Missing {req}"
        
        print("\n✅ Code generation successful!")


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_integration_real.py -v -s
    pytest.main([__file__, "-v", "-s", "--timeout=600"])
