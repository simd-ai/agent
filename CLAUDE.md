# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the development server
uvicorn simd_agent.main:app --reload --port 8000

# Run all tests (parallel by default via pyproject.toml addopts="-n 30")
pytest

# Run a single test file
pytest tests/test_linting.py

# Run a single test (use -n 0 to disable parallel for debugging)
pytest tests/test_linting.py::test_function_name -v -n 0

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
| `orchestration.py` | `Orchestrator` — central workflow coordinator (calls `plugin.validate_full()` for post-generation validation) |
| `models.py` | All Pydantic models: `SimulationConfigV1`, `StartRequest`, `AgentEvent`, `EventTypes`, etc. |
| `normalizer.py` / `config_normalizer.py` | Convert camelCase / legacy / partial configs → canonical `SimulationConfigV1` |
| `linting.py` | `CFDLinter` — validates physics, BCs, Reynolds number, mesh compatibility |
| `llm/` | LLM provider registry — auto-discovers providers under `llm/<name>/`; Gemini is the default. Contributors add new providers by creating a package |
| `llm/base.py` | `LLMProvider` abstract base class — contract for provider plugins |
| `llm/registry.py` | `LLMRegistry` — auto-discovers provider packages, configures from settings, exposes `get_provider()` |
| `llm/gemini/` | Default Gemini provider — wraps `google-genai` SDK |
| `genai_codegen.py` | `GenAICodeGenerator` — uses the LLM provider to produce OpenFOAM files; delegates to plugins for per-file prompts and required-file lists |
| `solver_selector.py` | `SolverSelector` — picks the right OpenFOAM solver from physics settings; roster is injected from the registry at call time |
| `solver_docs.py` | `load_prompt_pack()` — thin adapter that iterates `plugin.required_files()` / `plugin.prompt_for_file()` for the API response |
| `solvers/base.py` | `SolverPlugin` abstract base + universal validation helpers (`_fix_controldict_solver`, `_fix_pressure_field`, …) and `validate_full()` orchestrator entry point |
| `solvers/registry.py` | `SolverRegistry` — auto-discovers plugin packages and exposes classification queries (`p_solvers()`, `energy_solvers()`, `gravity_solvers()`, …) |
| `solvers/{name}/solver.py` | Per-solver plugin: class attributes (`algorithm`, `pressure_field`, `is_transient`, …) + `matches()`, `required_files()`, `validate()` |
| `precheck.py` | `PrecheckService` — multi-pass LLM precheck pipeline |
| `precheck_models.py` | Pydantic models and constants for the precheck flow |
| `simulation_server_client.py` | `SimulationServerClient` — HTTP client for external OpenFOAM runner |
| `event_bus.py` | `EventBus` — emits typed `AgentEvent` objects over the WebSocket |
| `packaging.py` | Packages generated files into a ZIP for submission |
| `error_summarizer.py` | LLM-based diagnosis of simulation failures |
| `code_verifier.py` | Super-model quality gate for generated OpenFOAM code |
| `case_spec.py` | `CaseSpec` / `build_case_spec()` — solver properties (`algorithm`, `pressure_field`, …) are derived from the plugin attributes via the registry |
| `settings.py` | `Settings` / `get_settings()` — env-based config via `pydantic-settings`, loaded from `.env` |
| `store.py` | `EventStore` — Postgres persistence for runs and events |

### LLM Provider Registry

The LLM layer follows the same plugin pattern as solvers. Providers are auto-discovered from sub-packages under `simd_agent/llm/`. Gemini is the default.

```
simd_agent/llm/
  __init__.py          # exports get_provider(), get_llm_registry()
  base.py              # LLMProvider abstract base class
  registry.py          # auto-discovery + settings-based configuration
  gemini/
    __init__.py        # exports provider_plugin = GeminiProvider()
    provider.py        # wraps google-genai SDK
```

**Adding a new provider** — create `simd_agent/llm/<name>/` with:
- `__init__.py` exporting `provider_plugin = YourProvider()`
- `provider.py` subclassing `LLMProvider` with `configure()`, `client`, `types`, `generate()`, `generate_stream()`

Set `DEFAULT_PROVIDER=<name>` in `.env` and add the provider's API key to settings.

### Solver Plugins — self-contained packages

Each OpenFOAM solver is a fully self-contained plugin package under `simd_agent/solvers/{solver}/`. Dropping a new directory into `solvers/` is the entire onboarding for a new solver — the registry auto-discovers it, no other file needs editing.

```
simd_agent/solvers/
  base.py              # SolverPlugin abstract base + universal validation helpers
  registry.py          # auto-discovery + classification queries
  simpleFoam/
    __init__.py        # exports `solver_plugin = SimpleFoamSolver()`
    solver.py          # matches(), required_files(), validate()
    prompts/
      _solver.md       # short identity + global rules (loaded into shared context)
      system/{controlDict,fvSchemes,fvSolution}.md
      constant/{transportProperties,turbulenceProperties}.md
      fields/{U,p,k,omega,epsilon,nut}.md
  pimpleFoam/ …
  rhoSimpleFoam/ …
  rhoPimpleFoam/ …
  buoyantSimpleFoam/   # adds system/fvOptions.md, constant/g.md, fields/{p_rgh,T,alphat}.md
  buoyantPimpleFoam/ …
```

**Plugin contract** — a plugin subclass of `SolverPlugin` must provide:
- Class attributes: `name`, `algorithm`, `pressure_field`, `is_transient`, `is_compressible`, `supports_energy`, `needs_gravity`, `is_multiphase`
- `matches(config) -> MatchResult` — scoring logic for solver selection
- `required_files(config) -> list[str]` — exact file manifest the LLM must generate
- `validate(files, config) -> ValidationResult` — solver-specific post-generation fixes (base helpers available via `self._fix_*`)
- `prompts/_solver.md` + per-file `prompts/{system,constant,fields}/*.md` docs

**Loading logic** (everything flows through the plugin):
- Shared context per run: `plugin.system_prompt()` → reads `solvers/{name}/prompts/_solver.md`
- Per-file context: `plugin.prompt_for_file(file_path)` → reads the matching doc under `system/`, `constant/`, or `fields/`
- Validation entry point: `plugin.validate_full(files, config)` → single call from `orchestration.py` that runs universal helpers + plugin-specific checks

**Shared (non-per-solver) prompts** still live in `simd_agent/prompts/packs/simd/`:
- `codegen.md` — universal output format rules
- `system.md` — agent role
- `codefix.md` — error recovery protocol
- `fluids/*.md` — fluid-specific EOS packs (liquidNitrogen, liquidHydrogen, …) injected into shared context when a known cryogenic fluid is configured

Multiphase solvers (`compressibleInterFoam`, `interFoam`, …) are still served by legacy monolithic `.md` files under `prompts/packs/simd/solvers/` and the fallback path in `build_required_files_list()` until they are ported to plugin packages.

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
- `SIMULATION_SERVER_URL` — external OpenFOAM runner base URL (configurable endpoint)
- `DEFAULT_PROVIDER` — LLM provider name (default `gemini`, must match a package under `simd_agent/llm/`)
- `GEMINI_API_KEY` — Google Gemini API key (primary LLM)
- `GEMINI_MODEL` / `GEMINI_SUPER_MODEL` — model names for codegen and verification
- `VTK_CACHE_DIR` — local directory for caching VTP files (default `/tmp/simd_vtk_cache`)

### VTK Results

After a successful simulation, VTP files are downloaded from the sim server once and cached locally. All subsequent requests are served directly from disk via:
- `GET /api/runs/{run_id}/vtk-results` — metadata + field list
- `GET /api/runs/{run_id}/vtk/surface.vtp` — latest surface
- `GET /api/runs/{run_id}/vtk-timestep/{time}/surface.vtp` — per-timestep
- `GET /api/runs/{run_id}/playback` — SSE playback stream

### Mesh Module

`mesh.py` (optional, requires PyVista/VTK) provides `/api/mesh/convert` for mesh format conversion. It is imported with a try/except guard and silently disabled if unavailable.
