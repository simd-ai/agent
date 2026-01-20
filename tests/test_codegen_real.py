# tests/test_codegen_real.py
"""
Real AI code generation tests using Gemini.

These tests are NOT for CI/CD - they call real LLM APIs and produce 
non-deterministic output. Run manually to inspect generated code.

Usage:
    # Set your API key first
    export GEMINI_API_KEY=your-key-here
    export DATABASE_URL=postgresql://...  # or use sqlite for testing
    
    # Run specific test
    pytest tests/test_codegen_real.py::TestRealCodegen::test_pipe_flow_generation -v -s
    
    # Run all real codegen tests
    pytest tests/test_codegen_real.py -v -s
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

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set - skipping real AI tests"
)


@pytest.fixture
def output_dir(tmp_path):
    """Create a directory to save generated outputs for inspection."""
    output = tmp_path / "codegen_output"
    output.mkdir()
    return output


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket that captures all events."""
    ws = MagicMock()
    ws.events = []
    
    async def capture_event(data):
        ws.events.append(data)
        # Also print for visibility
        print(f"\n📡 Event [{data.get('type', 'unknown')}]: {data.get('message', '')}")
        if data.get('payload'):
            # Print key payload items
            payload = data['payload']
            if 'files' in payload:
                print(f"   📁 Files: {payload['files']}")
            if 'validated_config' in payload:
                print(f"   ⚙️  Validated config keys: {list(payload['validated_config'].keys())}")
            if 'regime' in payload:
                print(f"   🌊 Flow regime: {payload['regime']}")
            if 'reynolds_number' in payload:
                print(f"   📊 Reynolds: {payload['reynolds_number']}")
    
    ws.send_json = AsyncMock(side_effect=capture_event)
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
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


class TestRealCodegen:
    """Tests for real AI code generation without sandbox execution."""
    
    async def test_pipe_flow_generation(self, mock_websocket, mock_store, output_dir):
        """
        Test generating a simple pipe flow case.
        
        This will:
        1. Lint the configuration (detect Re, regime, solver)
        2. Run planning with parallel sub-agents
        3. Generate OpenFOAM case files
        4. Save output for inspection (NOT run in sandbox)
        """
        from simd_agent.event_bus import EventBus
        from simd_agent.linting import CFDLinter
        from simd_agent.models import Operation, StartRequest
        from simd_agent.planning import Planner
        from simd_agent.packaging import extract_file_blocks, package_case
        
        print("\n" + "="*60)
        print("🧪 TEST: Pipe Flow Code Generation")
        print("="*60)
        
        run_id = uuid4()
        event_bus = EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=mock_store,
            persist=False,
        )
        
        # Configuration for a turbulent pipe flow
        config = {
            "geometry": {
                "type": "pipe",
                "diameter": 0.1,  # 10 cm
                "length": 1.0,    # 1 m
            },
            "inlet": {
                "velocity": 5.0,  # 5 m/s
            },
            "fluid": {
                "viscosity": 1e-6,  # water
                "density": 1000.0,
            },
        }
        
        requirements = """
        Simulate turbulent water flow through a straight pipe.
        - Pipe diameter: 0.1 m
        - Pipe length: 1.0 m  
        - Inlet velocity: 5 m/s uniform
        - Fluid: water (nu = 1e-6 m²/s)
        - Use steady-state solver
        - Run for 1000 iterations
        """
        
        print(f"\n📋 Requirements:\n{requirements}")
        print(f"\n⚙️  Config: {json.dumps(config, indent=2)}")
        
        # Step 1: Lint configuration
        print("\n" + "-"*40)
        print("Step 1: CFD Linting")
        print("-"*40)
        
        linter = CFDLinter(event_bus=event_bus)
        lint_result = await linter.lint(config, requirements)
        
        print(f"\n✅ Linting Results:")
        print(f"   Case Type: {lint_result.detected_case_type}")
        print(f"   Reynolds: {lint_result.reynolds_number:.0f}")
        print(f"   Regime: {lint_result.detected_regime}")
        print(f"   Solver: {lint_result.selected_solver}")
        print(f"   Issues: {len(lint_result.issues)}")
        print(f"   Changes: {len(lint_result.apply_changes)}")
        
        for change in lint_result.apply_changes:
            print(f"      → {change.path}: {change.value} ({change.reason})")
        
        # Step 2: Planning
        print("\n" + "-"*40)
        print("Step 2: Planning (parallel sub-agents)")
        print("-"*40)
        
        planner = Planner(event_bus)
        planning_result = await planner.plan(
            lint_result=lint_result,
            user_requirements=requirements,
            config=lint_result.validated_config,
        )
        
        print(f"\n✅ Planning Results:")
        print(f"   Work Items: {len(planning_result.work_items)}")
        print(f"   Solver: {planning_result.solver}")
        print(f"   Turbulence: {planning_result.turbulence_model}")
        print(f"   Mesh Strategy: {planning_result.mesh_strategy}")
        
        # Step 3: Code Generation (using real LLM via simd_codegen)
        print("\n" + "-"*40)
        print("Step 3: Code Generation (Gemini)")
        print("-"*40)
        
        try:
            from codegen import CodeGenerator, GenerationContext
            
            generator = CodeGenerator(
                provider="gemini3",
                prompt_pack="simd",
            )
            
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
            
            print("   🤖 Calling Gemini for code generation...")
            result = generator.generate(context)  # Not async
            llm_output = result.final_text
            
        except ImportError:
            print("   ⚠️  codegen not available, using mock generation")
            # Use the mock generator from orchestration
            from simd_agent.orchestration import Orchestrator
            
            # Create minimal orchestrator just for mock code gen
            request = StartRequest(
                op=Operation.CFD_CODEGEN_RUN,
                provider="mock",
                user_requirements=requirements,
                simulation_config=config,
            )
            orchestrator = Orchestrator(run_id, event_bus, mock_store, request)
            llm_output = orchestrator._mock_generate_code(planning_result)  # Not async
        
        # Step 4: Extract and save files
        print("\n" + "-"*40)
        print("Step 4: Extract Generated Files")
        print("-"*40)
        
        files = extract_file_blocks(llm_output)
        
        if not files:
            print("   ❌ No files extracted from LLM output!")
            print(f"\n   Raw output (first 2000 chars):\n{llm_output[:2000]}")
            pytest.fail("No files generated")
        
        print(f"\n✅ Generated {len(files)} files:")
        for path, content in files.items():
            print(f"   📄 {path} ({len(content)} chars)")
            
            # Save to output dir for inspection
            file_path = output_dir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
        
        # Step 5: Package as zip
        print("\n" + "-"*40)
        print("Step 5: Package Case")
        print("-"*40)
        
        zip_bytes, file_list = package_case(
            files=files,
            solver=planning_result.solver,
            case_name="pipe_flow",
        )
        
        zip_path = output_dir / "case.zip"
        zip_path.write_bytes(zip_bytes)
        
        print(f"\n✅ Case packaged:")
        print(f"   📦 {zip_path} ({len(zip_bytes)} bytes)")
        print(f"   Files in zip: {len(file_list)}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        print(f"   Output directory: {output_dir}")
        print(f"   Events emitted: {len(mock_websocket.events)}")
        print(f"   Files generated: {len(files)}")
        print(f"   Zip size: {len(zip_bytes)} bytes")
        print("\n   ℹ️  Inspect the output directory to review generated files")
        print("   ℹ️  The case.zip can be submitted to sandbox for testing")
        
        # Assertions
        assert len(files) >= 5, "Should generate at least 5 files"
        assert any("controlDict" in f for f in files), "Missing controlDict"
        assert any("fvSchemes" in f for f in files), "Missing fvSchemes"
        assert any("fvSolution" in f for f in files), "Missing fvSolution"
        assert len(zip_bytes) > 0, "Zip should not be empty"
    
    async def test_heat_transfer_generation(self, mock_websocket, mock_store, output_dir):
        """Test generating a heat transfer case."""
        from simd_agent.event_bus import EventBus
        from simd_agent.linting import CFDLinter
        from simd_agent.planning import Planner
        from simd_agent.packaging import extract_file_blocks
        
        print("\n" + "="*60)
        print("🧪 TEST: Heat Transfer Code Generation")
        print("="*60)
        
        run_id = uuid4()
        event_bus = EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=mock_store,
            persist=False,
        )
        
        config = {
            "geometry": {
                "type": "pipe",
                "diameter": 0.05,
                "length": 0.5,
            },
            "inlet": {
                "velocity": 1.0,
                "temperature": 300,  # 300 K inlet
            },
            "wall": {
                "temperature": 350,  # 350 K heated wall
            },
            "fluid": {
                "viscosity": 1.5e-5,  # air
                "density": 1.2,
            },
        }
        
        requirements = """
        Simulate heated pipe flow with conjugate heat transfer.
        - Air flowing through a heated pipe
        - Inlet temperature: 300 K
        - Wall temperature: 350 K (constant)
        - Use buoyant solver for temperature effects
        """
        
        linter = CFDLinter(event_bus=event_bus)
        lint_result = await linter.lint(config, requirements)
        
        print(f"\n✅ Detected case type: {lint_result.detected_case_type}")
        print(f"   Reynolds: {lint_result.reynolds_number:.0f if lint_result.reynolds_number else 'N/A'}")
        
        planner = Planner(event_bus)
        planning_result = await planner.plan(
            lint_result=lint_result,
            user_requirements=requirements,
            config=lint_result.validated_config,
        )
        
        print(f"   Solver: {planning_result.solver}")
        
        # For thermal cases, solver should be buoyant*
        assert "heat" in lint_result.detected_case_type or lint_result.detected_case_type == "heat_transfer"
    
    async def test_laminar_cavity_generation(self, mock_websocket, mock_store, output_dir):
        """Test generating a lid-driven cavity case (classic CFD benchmark)."""
        from simd_agent.event_bus import EventBus
        from simd_agent.linting import CFDLinter
        from simd_agent.planning import Planner
        
        print("\n" + "="*60)
        print("🧪 TEST: Lid-Driven Cavity Generation")
        print("="*60)
        
        run_id = uuid4()
        event_bus = EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=mock_store,
            persist=False,
        )
        
        config = {
            "geometry": {
                "type": "cavity",
                "length": 0.1,
                "height": 0.1,
            },
            "lid": {
                "velocity": 0.01,  # Very slow for laminar
            },
            "fluid": {
                "viscosity": 1e-4,  # High viscosity for laminar
                "density": 1000.0,
            },
        }
        
        requirements = """
        Simulate a 2D lid-driven cavity flow (classic benchmark).
        - Square cavity 0.1m x 0.1m
        - Top lid moves at 0.01 m/s
        - Laminar flow (low Reynolds)
        - Use icoFoam or simpleFoam with laminar model
        """
        
        linter = CFDLinter(event_bus=event_bus)
        lint_result = await linter.lint(config, requirements)
        
        print(f"\n✅ Detected:")
        print(f"   Case type: {lint_result.detected_case_type}")
        print(f"   Regime: {lint_result.detected_regime}")
        print(f"   Solver: {lint_result.selected_solver}")
        
        # Cavity should be detected, and flow should be laminar
        assert lint_result.detected_case_type == "cavity"


class TestRealLinting:
    """Tests for linting with various configurations."""
    
    async def test_reynolds_calculation_accuracy(self, mock_websocket, mock_store):
        """Test Reynolds number calculation with known values."""
        from simd_agent.event_bus import EventBus
        from simd_agent.linting import CFDLinter
        from simd_agent.models import FlowRegime
        
        run_id = uuid4()
        event_bus = EventBus(run_id, mock_websocket, mock_store, persist=False)
        linter = CFDLinter(event_bus=event_bus)
        
        # Known case: Re = U * D / nu = 1 * 0.1 / 1e-6 = 100,000 (turbulent)
        config = {
            "geometry": {"diameter": 0.1},
            "inlet": {"velocity": 1.0},
            "fluid": {"viscosity": 1e-6},
        }
        
        result = await linter.lint(config)
        
        print(f"\n✅ Reynolds Calculation Test:")
        print(f"   Expected: 100,000")
        print(f"   Calculated: {result.reynolds_number:.0f}")
        print(f"   Regime: {result.detected_regime}")
        
        assert abs(result.reynolds_number - 100000) < 1
        assert result.detected_regime == FlowRegime.TURBULENT
    
    async def test_regime_boundaries(self, mock_websocket, mock_store):
        """Test correct regime detection at boundaries."""
        from simd_agent.event_bus import EventBus
        from simd_agent.linting import CFDLinter
        from simd_agent.models import FlowRegime
        
        run_id = uuid4()
        event_bus = EventBus(run_id, mock_websocket, mock_store, persist=False)
        linter = CFDLinter(event_bus=event_bus)
        
        print("\n✅ Regime Boundary Tests:")
        
        test_cases = [
            # (velocity, diameter, viscosity, expected_regime)
            (0.02, 0.1, 1e-6, FlowRegime.LAMINAR),       # Re = 2000
            (0.025, 0.1, 1e-6, FlowRegime.TRANSITIONAL), # Re = 2500
            (0.05, 0.1, 1e-6, FlowRegime.TURBULENT),     # Re = 5000
        ]
        
        for vel, diam, visc, expected in test_cases:
            config = {
                "geometry": {"diameter": diam},
                "inlet": {"velocity": vel},
                "fluid": {"viscosity": visc},
            }
            result = await linter.lint(config)
            re = result.reynolds_number
            
            print(f"   Re={re:.0f}: {result.detected_regime} (expected: {expected})")
            assert result.detected_regime == expected


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_codegen_real.py -v -s
    pytest.main([__file__, "-v", "-s"])
