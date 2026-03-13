# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the development server
uvicorn simd_agent.main:app --reload --port 8000

# Run all tests
pytest

# Run a single test file
pytest tests/test_linting.py

# Run a single test
pytest tests/test_linting.py::test_function_name -v

# Install dependencies (including dev extras)
pip install -e ".[dev]"
```

## Architecture

This is a **FastAPI service** that orchestrates CFD (Computational Fluid Dynamics) workflows using OpenFOAM, driven by LLMs (primarily Google Gemini). It communicates with a frontend via WebSocket and submits simulation cases to an external simulation server.

### Request Flow

Two operations are supported, both initiated via WebSocket at `/ws/run`:

**`CFD_LINT`**: Validate and normalize a simulation config. Returns lint issues, suggested corrections, and Reynolds number analysis without running any simulation.

**`CFD_CODEGEN_RUN`**: Full end-to-end workflow:
1. Normalize raw config → `SimulationConfigV1` (via `normalizer.py` / `config_normalizer.py`)
2. Lint the config — detect errors, infer flow regime, compute Re (via `linting.py`)
3. Select an OpenFOAM solver (via `solver_selector.py`)
4. Load solver-specific prompt pack from `prompts/packs/simd/solvers/<solver>.md`
5. Generate OpenFOAM case files via LLM (via `genai_codegen.py` / `GenAICodeGenerator`)
6. Post-generation validation (`validate_generated_files`)
7. Optional super-model code verification (`code_verifier.py`)
8. Package files into a ZIP and submit to the simulation server (`simulation_server_client.py`)
9. Stream simulation events (SSE from sim server → WebSocket to client)
10. Self-healing loop: on failure, `error_summarizer.py` diagnoses the error and the Orchestrator retries codegen up to `max_retries` times

### Pre-check Flow

A separate two-endpoint pre-check flow parses natural language prompts before a full run:
- `POST /api/precheck` — synchronous JSON response
- `WS /ws/precheck` — streaming with thought tokens

The `PrecheckService` (`precheck.py`) runs a multi-pass LLM pipeline: boundary planner → parallel per-patch agents → merge → review.

### Key Modules

| Module | Role |
|---|---|
| `main.py` | FastAPI app, WebSocket handlers, VTK cache endpoints |
| `orchestration.py` | `Orchestrator` — central workflow coordinator |
| `models.py` | All Pydantic models: `SimulationConfigV1`, `StartRequest`, `AgentEvent`, `EventTypes`, etc. |
| `normalizer.py` / `config_normalizer.py` | Convert camelCase / legacy / partial configs → canonical `SimulationConfigV1` |
| `linting.py` | `CFDLinter` — validates physics, BCs, Reynolds number, mesh compatibility |
| `genai_codegen.py` | `GenAICodeGenerator` — calls Google GenAI to produce OpenFOAM files |
| `solver_selector.py` | `SolverSelector` — picks the right OpenFOAM solver from physics settings |
| `solver_docs.py` | `load_prompt_pack()` — loads solver-specific `.md` prompt files |
| `precheck.py` | `PrecheckService` — multi-pass LLM precheck pipeline |
| `precheck_models.py` | Pydantic models and constants for the precheck flow |
| `simulation_server_client.py` | `SimulationServerClient` — HTTP client for external OpenFOAM runner |
| `event_bus.py` | `EventBus` — emits typed `AgentEvent` objects over the WebSocket |
| `packaging.py` | Packages generated files into a ZIP for submission |
| `error_summarizer.py` | LLM-based diagnosis of simulation failures |
| `code_verifier.py` | Super-model quality gate for generated OpenFOAM code |
| `case_spec.py` | `CaseSpec` / `build_case_spec()` — structured representation of what files to generate |
| `settings.py` | `Settings` / `get_settings()` — env-based config via `pydantic-settings`, loaded from `.env` |
| `store.py` | `EventStore` — Postgres persistence for runs and events |

### Prompt Packs

Solver-specific generation instructions live in:
```
simd_agent/prompts/packs/simd/
  codegen.md              # base codegen prompt (always in shared context)
  system.md               # system prompt
  solvers/
    # Legacy monolithic files (fallback when per-file dir doesn't exist):
    simpleFoam.md
    pimpleFoam.md
    interFoam.md
    rhoPimpleFoam.md
    ...
    # Per-file split structure (preferred — loaded file-by-file per generation call):
    rhoSimpleFoam/
      _solver.md          # short identity + global rules (in shared context)
      system/fvSchemes.md
      system/fvSolution.md
      system/controlDict.md
      constant/thermophysicalProperties.md
      constant/turbulenceProperties.md
      fields/U.md
      fields/p.md
      fields/T.md
      fields/k.md
      fields/omega.md
      fields/epsilon.md
      fields/nut.md
      fields/alphat.md
    simpleFoam/           # same structure
    pimpleFoam/           # same structure
```

**Loading logic** (`genai_codegen.py`):
- Shared context: `_load_solver_base(solver)` → tries `solvers/{solver}/_solver.md`, falls back to `solvers/{solver}.md`
- Per-file context: `_load_solver_file_doc(solver, relpath)` → loads `solvers/{solver}/{relpath}` or returns `""` (graceful fallback to `_brief_solver_note()`)
- File path → doc path: `_file_doc_relpath(file_path)` maps `system/fvSchemes` → `system/fvSchemes.md`, `0/U` → `fields/U.md`

### Config Schema

The canonical config is `SimulationConfigV1` (in `models.py`):
- `mesh`: `MeshInfoV1` — mesh ID, patches, checkMesh stats
- `physics`: `PhysicsV1` — flow regime, time scheme, compressibility, multiphase, heat transfer
- `solver`: `SolverV1` — solver type, iterations, convergence, end time
- `fluid`: `FluidV1` — density, viscosity, thermal properties
- `turbulence`: `TurbulenceConfigV1` — model, pre-computed k/ω/ε/νt values
- `boundary_conditions`: `dict[str, BoundaryConditionV1]` — keyed by patch name

The frontend sends camelCase keys; `AliasChoices` on model fields handles both camelCase and snake_case transparently.

### Environment Variables (`.env`)

Required:
- `DATABASE_URL` — Neon Postgres connection string

Key optional overrides:
- `SIMULATION_SERVER_URL` — external OpenFOAM runner base URL
- `GEMINI_API_KEY` — Google Gemini API key (primary LLM)
- `GEMINI_MODEL` / `GEMINI_SUPER_MODEL` — model names for codegen and verification
- `MAX_RETRIES` — self-healing retry limit (default 3)
- `VTK_CACHE_DIR` — local directory for caching VTP files (default `/tmp/simd_vtk_cache`)

### VTK Results

After a successful simulation, VTP files are downloaded from the sim server once and cached locally. All subsequent requests are served directly from disk via:
- `GET /api/runs/{run_id}/vtk-results` — metadata + field list
- `GET /api/runs/{run_id}/vtk/surface.vtp` — latest surface
- `GET /api/runs/{run_id}/vtk-timestep/{time}/surface.vtp` — per-timestep
- `GET /api/runs/{run_id}/playback` — SSE playback stream

### Mesh Module

`mesh.py` (optional, requires PyVista/VTK) provides `/api/mesh/convert` for mesh format conversion. It is imported with a try/except guard and silently disabled if unavailable.
