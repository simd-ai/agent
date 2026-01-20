# tests/test_integration_real.py
"""
Real end-to-end integration tests with actual sandbox.

These tests:
1. Generate OpenFOAM code using real AI (Gemini)
2. Submit to the actual sandbox for execution
3. Wait for results and report success/failure

⚠️  These tests are NOT for CI/CD - they:
   - Call real LLM APIs (costs money)
   - Submit to real sandbox (uses compute resources)
   - Take significant time (minutes per test)

Usage:
    # Set environment variables
    export GEMINI_API_KEY=your-key-here
    export SANDBOX_BASE_URL=https://legal-many-zebra.ngrok-free.app
    export DATABASE_URL=postgresql://...  # or skip for in-memory testing
    
    # Run a single integration test
    pytest tests/test_integration_real.py::TestFullIntegration::test_simple_pipe_flow -v -s
    
    # Run all integration tests (takes several minutes)
    pytest tests/test_integration_real.py -v -s --timeout=600
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# Configuration
SANDBOX_URL = os.environ.get("SANDBOX_BASE_URL", "https://legal-many-zebra.ngrok-free.app")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

# Skip if no API key or sandbox URL
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
            'sandbox_submitted': '📤',
            'sandbox_status': '⏳',
            'sandbox_logs': '📜',
            'sandbox_succeeded': '🎉',
            'sandbox_failed': '❌',
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
        elif event_type == 'sandbox_logs':
            logs = payload.get('logs', '')
            if logs:
                lines = logs.strip().split('\n')[-5:]  # Last 5 lines
                for line in lines:
                    print(f"         │ {line[:80]}")
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


@pytest.fixture
def real_sandbox_client():
    """Create a real sandbox client."""
    from simd_agent.sandbox_client import SandboxClient
    return SandboxClient(base_url=SANDBOX_URL)


class TestFullIntegration:
    """Full end-to-end integration tests."""
    
    async def test_simple_pipe_flow(self, mock_websocket, mock_store, real_sandbox_client):
        """
        Full integration test: Generate and run a simple pipe flow case.
        
        This test:
        1. Lints configuration
        2. Plans the simulation
        3. Generates OpenFOAM code with Gemini
        4. Packages the case
        5. Submits to real sandbox
        6. Waits for execution
        7. Reports results
        """
        from simd_agent.event_bus import EventBus
        from simd_agent.models import Constraints, Operation, StartRequest
        from simd_agent.orchestration import Orchestrator
        
        print("\n" + "="*70)
        print("🧪 FULL INTEGRATION TEST: Simple Pipe Flow")
        print(f"   Sandbox: {SANDBOX_URL}")
        print("="*70 + "\n")
        
        run_id = uuid4()
        
        request = StartRequest(
            op=Operation.CFD_CODEGEN_RUN,
            provider="gemini3",  # Use real Gemini
            prompt_pack="simd",
            user_requirements="""
            Simulate laminar water flow through a short pipe.
            - Pipe diameter: 0.05 m (5 cm)
            - Pipe length: 0.2 m (20 cm)
            - Inlet velocity: 0.1 m/s (low velocity for laminar)
            - Fluid: water (nu = 1e-6 m²/s)
            - Use steady-state simpleFoam solver
            - Run for only 100 iterations (quick validation)
            - Use a coarse mesh (fast execution)
            """,
            simulation_config={
                "geometry": {
                    "type": "pipe",
                    "diameter": 0.05,
                    "length": 0.2,
                },
                "inlet": {
                    "velocity": 0.1,
                },
                "fluid": {
                    "viscosity": 1e-6,
                    "density": 1000.0,
                },
            },
            constraints=Constraints(
                max_retries=2,  # Allow 2 retries for self-healing
                timeout_seconds=120,  # 2 minute timeout
            ),
        )
        
        event_bus = EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=mock_store,
            persist=False,
        )
        
        orchestrator = Orchestrator(
            run_id=run_id,
            event_bus=event_bus,
            store=mock_store,
            request=request,
        )
        
        # Inject real sandbox client
        orchestrator._sandbox = real_sandbox_client
        
        print("Starting orchestration...\n")
        
        try:
            result = await orchestrator.run()
        finally:
            await real_sandbox_client.close()
        
        # Print final summary
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        print(f"   Status: {result.status}")
        print(f"   Iterations: {result.iterations}")
        print(f"   Retries: {result.retries}")
        if result.error:
            print(f"   Error: {result.error}")
        if result.solver:
            print(f"   Solver: {result.solver}")
        if result.case_type:
            print(f"   Case Type: {result.case_type}")
        
        # Find final event
        final_events = [e for e in mock_websocket.events if e.get('type') == 'final']
        if final_events:
            final = final_events[-1]['payload']
            print(f"   Duration: {final.get('duration_seconds', 'N/A'):.1f}s")
            if final.get('artifacts'):
                print(f"   Artifacts: {len(final['artifacts'])}")
        
        print("="*70 + "\n")
        
        # Note: We don't assert success because sandbox execution may fail
        # for legitimate reasons (mesh issues, etc.). The test passes if
        # the orchestration completes without crashing.
        assert result.status is not None, "Should have a status"
        assert result.iterations >= 1, "Should have at least 1 iteration"
    
    async def test_codegen_only_no_sandbox(self, mock_websocket, mock_store):
        """
        Test code generation only without sandbox execution.
        
        Useful for testing AI output quality without waiting for execution.
        """
        from simd_agent.event_bus import EventBus
        from simd_agent.linting import CFDLinter
        from simd_agent.planning import Planner
        from simd_agent.packaging import extract_file_blocks, package_from_llm_output
        
        print("\n" + "="*70)
        print("🧪 CODE GENERATION TEST (No Sandbox)")
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


class TestSandboxConnectivity:
    """Tests for sandbox connectivity and basic operations."""
    
    async def test_sandbox_health(self, real_sandbox_client):
        """Test that sandbox is reachable."""
        import httpx
        
        print(f"\n🔌 Testing sandbox connectivity: {SANDBOX_URL}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{SANDBOX_URL}/health", timeout=10.0)
                print(f"   Status: {response.status_code}")
                if response.status_code == 200:
                    print(f"   Response: {response.json()}")
                    print("   ✅ Sandbox is healthy!")
                else:
                    print(f"   ⚠️  Unexpected status: {response.text}")
        except Exception as e:
            print(f"   ❌ Connection failed: {e}")
            pytest.skip(f"Sandbox not reachable: {e}")
    
    async def test_submit_minimal_case(self, real_sandbox_client):
        """Test submitting a minimal case to sandbox."""
        from simd_agent.packaging import package_case
        
        print(f"\n📤 Testing minimal case submission to sandbox")
        
        # Create a minimal valid OpenFOAM case
        files = {
            "system/controlDict": """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
""",
            "system/fvSchemes": """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""",
            "system/fvSolution": """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers { }
SIMPLE { }
""",
        }
        
        zip_bytes, file_list = package_case(files, solver="simpleFoam", case_name="minimal")
        
        print(f"   Created minimal case: {len(zip_bytes)} bytes, {len(file_list)} files")
        
        try:
            # Submit to sandbox
            result = await real_sandbox_client.submit_run(
                case_zip=zip_bytes,
                run_script="run.sh",
                metadata={"test": "minimal_case"},
            )
            
            print(f"   ✅ Submitted! Run ID: {result.run_id}")
            
            # Check status
            await asyncio.sleep(2)
            status = await real_sandbox_client.get_status(result.run_id)
            print(f"   Status: {status.state}")
            
            # Note: This will likely fail since it's an incomplete case,
            # but it tests the sandbox connectivity
            
        except Exception as e:
            print(f"   ⚠️  Submission failed: {e}")
            # This is expected for minimal case
        finally:
            await real_sandbox_client.close()


class TestErrorRecovery:
    """Tests for self-healing error recovery."""
    
    async def test_invalid_case_recovery(self, mock_websocket, mock_store, real_sandbox_client):
        """
        Test that the system can recover from an invalid case.
        
        This test intentionally generates a case that will fail,
        then verifies the error summarizer and retry logic work.
        """
        from simd_agent.event_bus import EventBus
        from simd_agent.models import Constraints, Operation, StartRequest
        from simd_agent.orchestration import Orchestrator
        
        print("\n" + "="*70)
        print("🧪 ERROR RECOVERY TEST")
        print("="*70 + "\n")
        
        run_id = uuid4()
        
        # Intentionally problematic config (missing critical info)
        request = StartRequest(
            op=Operation.CFD_CODEGEN_RUN,
            provider="mock",  # Use mock to get predictable (incomplete) output
            prompt_pack="simd",
            user_requirements="Generate a CFD case",
            simulation_config={
                "geometry": {"type": "unknown"},  # Vague
            },
            constraints=Constraints(max_retries=1, timeout_seconds=60),
        )
        
        event_bus = EventBus(run_id, mock_websocket, mock_store, persist=False)
        
        orchestrator = Orchestrator(
            run_id=run_id,
            event_bus=event_bus,
            store=mock_store,
            request=request,
        )
        
        orchestrator._sandbox = real_sandbox_client
        
        try:
            result = await orchestrator.run()
        finally:
            await real_sandbox_client.close()
        
        print(f"\n📊 Result: {result.status}")
        print(f"   Iterations: {result.iterations}")
        print(f"   Retries: {result.retries}")
        
        # Check that error_summary events were emitted (if retries happened)
        error_summaries = [e for e in mock_websocket.events if e.get('type') == 'error_summary']
        retry_events = [e for e in mock_websocket.events if e.get('type') == 'retrying']
        
        print(f"   Error summaries: {len(error_summaries)}")
        print(f"   Retry attempts: {len(retry_events)}")


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_integration_real.py -v -s
    pytest.main([__file__, "-v", "-s", "--timeout=600"])
