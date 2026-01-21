#!/usr/bin/env python3
"""
Interactive test runner for simd_agent real tests.

Usage:
    python scripts/test_runner.py
    
Required environment variables:
    GEMINI_API_KEY - Your Gemini API key
    
Optional:
    SANDBOX_BASE_URL - Sandbox URL (default: https://legal-many-zebra.ngrok-free.app)
    DATABASE_URL - Database URL (default: uses mock store for testing)
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file into environment (so codegen can access API keys)
from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def check_environment():
    """Check required environment variables (from .env or environment)."""
    # Load from settings (which reads .env)
    try:
        from simd_agent.settings import get_settings
        settings = get_settings()
        gemini_key = settings.gemini_api_key
        sandbox_url = str(settings.sandbox_base_url)
    except Exception:
        # Fallback to environment
        gemini_key = os.environ.get("GEMINI_API_KEY")
        sandbox_url = os.environ.get("SANDBOX_BASE_URL", "https://legal-many-zebra.ngrok-free.app")
    
    print("=" * 60)
    print("SIMD Agent Test Runner")
    print("=" * 60)
    print()
    
    if not gemini_key:
        print("❌ GEMINI_API_KEY not set!")
        print("   Set it with: export GEMINI_API_KEY=your-key-here")
        print("   Or add to .env file")
        print()
        print("   Without it, tests will use mock AI (deterministic but limited)")
        use_mock = True
    else:
        print(f"✅ GEMINI_API_KEY: ...{gemini_key[-8:]}")
        use_mock = False
    
    print(f"🌐 SANDBOX_BASE_URL: {sandbox_url}")
    print()
    
    return use_mock, sandbox_url


async def test_linting():
    """Run linting tests."""
    from unittest.mock import AsyncMock, MagicMock
    from uuid import uuid4
    
    from simd_agent.event_bus import EventBus
    from simd_agent.linting import CFDLinter
    
    print("-" * 40)
    print("🔍 Testing CFD Linting")
    print("-" * 40)
    
    # Create mock websocket
    ws = MagicMock()
    ws.send_json = AsyncMock()
    
    store = MagicMock()
    store.append_event = AsyncMock()
    
    run_id = uuid4()
    event_bus = EventBus(run_id, ws, store, persist=False)
    
    linter = CFDLinter(event_bus=event_bus)
    
    # Test case: turbulent pipe flow
    config = {
        "geometry": {"type": "pipe", "diameter": 0.1, "length": 1.0},
        "inlet": {"velocity": 5.0},
        "fluid": {"viscosity": 1e-6, "density": 1000.0},
    }
    
    print("\n📋 Input Configuration:")
    print(f"   Pipe: D={config['geometry']['diameter']}m, L={config['geometry']['length']}m")
    print(f"   Inlet: U={config['inlet']['velocity']} m/s")
    print(f"   Fluid: ν={config['fluid']['viscosity']} m²/s")
    
    result = await linter.lint(config, "Turbulent pipe flow simulation")
    
    print("\n✅ Linting Results:")
    print(f"   Case Type: {result.detected_case_type}")
    print(f"   Reynolds: {result.reynolds_number:.0f}")
    print(f"   Regime: {result.detected_regime}")
    print(f"   Solver: {result.selected_solver}")
    print(f"   Turbulence: {result.validated_config.get('turbulence_model')}")
    print(f"   Issues: {len(result.issues)}")
    print(f"   Recommendations: {len(result.apply_changes)}")
    
    for change in result.apply_changes[:3]:  # Show first 3
        print(f"      → {change.path}: {change.value}")
    
    return result


async def test_planning():
    """Run planning tests."""
    from unittest.mock import AsyncMock, MagicMock
    from uuid import uuid4
    
    from simd_agent.event_bus import EventBus
    from simd_agent.linting import CFDLinter
    from simd_agent.planning import Planner
    
    print("\n" + "-" * 40)
    print("📋 Testing Planning (Parallel Sub-Agents)")
    print("-" * 40)
    
    ws = MagicMock()
    ws.send_json = AsyncMock()
    store = MagicMock()
    store.append_event = AsyncMock()
    
    run_id = uuid4()
    event_bus = EventBus(run_id, ws, store, persist=False)
    
    # First lint
    linter = CFDLinter(event_bus=event_bus)
    config = {
        "geometry": {"type": "pipe", "diameter": 0.1},
        "inlet": {"velocity": 5.0},
        "fluid": {"viscosity": 1e-6},
    }
    lint_result = await linter.lint(config, "Pipe flow")
    
    # Then plan
    planner = Planner(event_bus)
    result = await planner.plan(
        lint_result=lint_result,
        user_requirements="Turbulent pipe flow",
        config=lint_result.validated_config,
    )
    
    print(f"\n✅ Planning Results:")
    print(f"   Work Items: {len(result.work_items)}")
    for item in result.work_items:
        print(f"      • {item.task}: {item.description[:40]}...")
    print(f"   Solver: {result.solver}")
    print(f"   Turbulence: {result.turbulence_model}")
    print(f"   Mesh Strategy: {result.mesh_strategy}")
    
    return result


async def test_codegen(use_mock: bool):
    """Run code generation tests."""
    from unittest.mock import AsyncMock, MagicMock
    from uuid import uuid4
    
    from simd_agent.event_bus import EventBus
    from simd_agent.linting import CFDLinter
    from simd_agent.packaging import extract_file_blocks
    from simd_agent.planning import Planner
    
    print("\n" + "-" * 40)
    print("🤖 Testing Code Generation")
    print("-" * 40)
    
    ws = MagicMock()
    ws.send_json = AsyncMock()
    store = MagicMock()
    store.append_event = AsyncMock()
    
    run_id = uuid4()
    event_bus = EventBus(run_id, ws, store, persist=False)
    
    config = {
        "geometry": {"type": "pipe", "diameter": 0.1, "length": 1.0},
        "inlet": {"velocity": 5.0},
        "fluid": {"viscosity": 1e-6, "density": 1000.0},
    }
    requirements = """Generate a turbulent pipe flow case with k-epsilon model.

IMPORTANT: Output each file using this exact format:
```file:relative/path/to/file
<file content>
```

For example:
```file:system/controlDict
FoamFile { ... }
application simpleFoam;
```

Generate all required OpenFOAM files: system/controlDict, system/fvSchemes, system/fvSolution, system/blockMeshDict, 0/U, 0/p, constant/transportProperties.
"""
    
    linter = CFDLinter(event_bus=event_bus)
    lint_result = await linter.lint(config, requirements)
    
    planner = Planner(event_bus)
    planning_result = await planner.plan(lint_result, requirements, lint_result.validated_config)
    
    if use_mock:
        print("\n⚠️  Using mock code generator (no API key)")
        from simd_agent.orchestration import Orchestrator
        from simd_agent.models import Operation, StartRequest
        
        request = StartRequest(op=Operation.CFD_CODEGEN_RUN, user_requirements=requirements)
        orchestrator = Orchestrator(run_id, event_bus, store, request)
        llm_output = orchestrator._mock_generate_code(planning_result)  # Not async
    else:
        print("\n🤖 Calling Gemini for code generation...")
        try:
            from codegen import CodeGenerator, GenerationContext
            
            generator = CodeGenerator(provider="gemini3", prompt_pack="default")
            context = GenerationContext(
                task="codegen",
                domain="openfoam_case",
                requirements=requirements,
                extra_context={
                    "validated_config": lint_result.validated_config,
                    "solver": planning_result.solver,
                    "turbulence_model": planning_result.turbulence_model,
                    "case_type": planning_result.case_type,
                    "mesh_strategy": planning_result.mesh_strategy,
                },
            )
            result = generator.generate(context)  # Not async
            llm_output = result.final_text
        except ImportError:
            print("   ⚠️  codegen not installed, using mock")
            from simd_agent.orchestration import Orchestrator
            from simd_agent.models import Operation, StartRequest
            
            request = StartRequest(op=Operation.CFD_CODEGEN_RUN, user_requirements=requirements)
            orchestrator = Orchestrator(run_id, event_bus, store, request)
            llm_output = orchestrator._mock_generate_code(planning_result)  # Not async
    
    files = extract_file_blocks(llm_output)
    
    print(f"\n✅ Generated {len(files)} files:")
    for path, content in sorted(files.items()):
        print(f"   📄 {path} ({len(content)} chars)")
    
    return files


async def test_sandbox_connectivity(sandbox_url: str):
    """Test sandbox connectivity."""
    import httpx
    
    print("\n" + "-" * 40)
    print("🌐 Testing Sandbox Connectivity")
    print("-" * 40)
    
    print(f"\n   URL: {sandbox_url}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{sandbox_url}/health", timeout=10.0)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   Response: {response.json()}")
                print("   ✅ Sandbox is healthy!")
                return True
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False


async def test_precheck(use_mock: bool):
    """Test the precheck endpoint."""
    print("\n" + "-" * 40)
    print("🔮 Testing Precheck (Prompt Analysis)")
    print("-" * 40)
    
    from simd_agent.precheck import PrecheckRequest, get_precheck_service
    
    # Test 1: Simple turbulent pipe flow
    print("\n📝 Test 1: Turbulent pipe flow prompt")
    request1 = PrecheckRequest(
        prompt="I want to simulate turbulent water flow through a 10cm diameter pipe at 2 m/s inlet velocity. I need to achieve a pressure drop less than 500 Pa."
    )
    
    service = get_precheck_service()
    response1 = await service.analyze(request1)
    
    print(f"   ✅ Success: {response1.success}")
    print(f"   📊 Suggested Config:")
    print(f"      Flow Regime: {response1.suggestedConfig.flowRegime}")
    print(f"      Time Scheme: {response1.suggestedConfig.timeScheme}")
    print(f"      Heat Transfer: {response1.suggestedConfig.enableHeatTransfer}")
    print(f"      Turbulence Model: {response1.suggestedConfig.turbulenceModel}")
    print(f"   📋 Interpretation:")
    print(f"      Summary: {response1.interpretation.summary[:80]}...")
    print(f"      Type: {response1.interpretation.simulationType}")
    print(f"      Physics: {response1.interpretation.keyPhysics}")
    print(f"   📈 Confidence: {response1.confidence.overall:.0%}")
    if response1.kpiTargets and response1.kpiTargets.pressureDrop:
        print(f"   🎯 KPI Target: pressure drop < {response1.kpiTargets.pressureDrop.value} {response1.kpiTargets.pressureDrop.unit}")
    
    # Test 2: Heat transfer case
    print("\n📝 Test 2: Heat transfer prompt")
    request2 = PrecheckRequest(
        prompt="Simulate cooling of a hot plate with air at 25°C flowing at 5 m/s. The plate is at 100°C."
    )
    
    response2 = await service.analyze(request2)
    
    print(f"   ✅ Success: {response2.success}")
    print(f"   📊 Heat Transfer Enabled: {response2.suggestedConfig.enableHeatTransfer}")
    print(f"   📋 Key Physics: {response2.interpretation.keyPhysics}")
    
    # Test 3: With mesh info
    print("\n📝 Test 3: Prompt with mesh info")
    from simd_agent.precheck import MeshInfo, MeshPatch, CheckMeshInfo
    
    request3 = PrecheckRequest(
        prompt="Run a steady-state simulation of airflow through this duct",
        mesh=MeshInfo(
            meshId="test-mesh-123",
            fileName="duct.msh",
            patches=[
                MeshPatch(name="inlet", type="patch", nCells=500),
                MeshPatch(name="outlet", type="patch", nCells=500),
                MeshPatch(name="walls", type="wall", nCells=10000),
                MeshPatch(name="symmetryPlane", type="symmetry", nCells=2000),
            ],
            checkMesh=CheckMeshInfo(
                cells=250000,
                faces=780000,
                points=260000,
            ),
        ),
    )
    
    response3 = await service.analyze(request3)
    
    print(f"   ✅ Success: {response3.success}")
    print(f"   📊 Boundary Hints:")
    if response3.boundaryHints:
        for patch_name, hint in response3.boundaryHints.items():
            print(f"      {patch_name}: {hint.suggestedType} (confidence: {hint.confidence:.0%})")
            if hint.reasoning:
                print(f"         → {hint.reasoning[:50]}...")
    print(f"   🖼️  Show Mesh Viewer: {response3.shouldShowMeshViewer}")
    print(f"   ➡️  Next Step: {response3.nextStep}")
    
    return response1, response2, response3


async def main():
    """Main interactive test runner."""
    use_mock, sandbox_url = check_environment()
    
    print("Select test to run:")
    print("  1. Linting only")
    print("  2. Planning only")
    print("  3. Code generation")
    print("  4. Sandbox connectivity")
    print("  5. Precheck (prompt analysis)")
    print("  6. All of the above")
    print("  q. Quit")
    print()
    
    choice = input("Choice [1-6, q]: ").strip()
    
    if choice == 'q':
        return
    
    try:
        if choice in ('1', '6'):
            await test_linting()
        
        if choice in ('2', '6'):
            await test_planning()
        
        if choice in ('3', '6'):
            await test_codegen(use_mock)
        
        if choice in ('4', '6'):
            await test_sandbox_connectivity(sandbox_url)
        
        if choice in ('5', '6'):
            await test_precheck(use_mock)
        
        print("\n" + "=" * 60)
        print("✅ Tests complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
