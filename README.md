# SIMD Agent

FastAPI service for orchestrating CFD (Computational Fluid Dynamics) workflows via WebSockets. This service provides end-to-end simulation management from configuration validation to execution in a remote sandbox.

## Features

- **CFD Configuration Linting**: Validate simulation configurations, detect flow regimes, and recommend appropriate solver/turbulence model settings
- **OpenFOAM Code Generation**: Generate complete OpenFOAM case files using LLM-powered code generation
- **Self-Healing Execution Loop**: Automatically retry failed simulations with error analysis and fixes
- **Real-Time Progress Streaming**: WebSocket-based event streaming for live progress updates
- **Parallel Sub-Agents**: Concurrent execution of planning tasks for faster setup
- **Persistent Event Storage**: All events and run metadata stored in Neon Postgres

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL database (Neon recommended)
- Access to SIMD Sandbox service
- API keys for LLM providers (Gemini, Grok, etc.)

### Installation

```bash
# Clone and install
cd simd_agent
pip install -e ".[dev]"
```

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
# Required
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Sandbox
SANDBOX_BASE_URL=https://sandbox.simd.dev
SANDBOX_TIMEOUT=300

# LLM Providers (at least one required)
GEMINI_API_KEY=your-gemini-key
GROK_API_KEY=your-grok-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Optional
DEFAULT_PROVIDER=gemini3
MAX_RETRIES=3
LOG_LEVEL=INFO
```

### Running the Service

```bash
# Development
uvicorn simd_agent.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn simd_agent.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Running Tests

```bash
pytest tests/ -v
```

## WebSocket Protocol

### Endpoint

```
ws://localhost:8000/ws/run
```

### Client → Server: StartRequest

#### Minimal Example (CFD_LINT)

```json
{
  "op": "CFD_LINT",
  "provider": "gemini3",
  "user_requirements": "Simulate water flow through a pipe",
  "simulation_config": {
    "mesh": "mesh-abc123"
  }
}
```

#### Full V1 Example (CFD_CODEGEN_RUN)

```json
{
  "op": "CFD_CODEGEN_RUN",
  "provider": "gemini3",
  "prompt_pack": "simd",
  "user_requirements": "Simulate turbulent water flow through a 10cm diameter pipe at 5 m/s inlet velocity. Calculate pressure drop.",
  "simulation_config": {
    "mesh": {
      "mesh_id": "mesh-abc123",
      "file_name": "pipe.stl",
      "patches": [
        { "name": "inlet", "type": "patch", "n_faces": 100 },
        { "name": "outlet", "type": "patch", "n_faces": 100 },
        { "name": "wall", "type": "wall", "n_faces": 5000 }
      ],
      "check_mesh": {
        "cells": 50000,
        "faces": 150000,
        "points": 52000,
        "bounding_box": { "min": [0, 0, 0], "max": [1, 0.1, 0.1] },
        "characteristic_length": 0.1
      }
    },
    "physics": {
      "flow_regime": "turbulent",
      "time_scheme": "steady",
      "compressibility": "incompressible",
      "heat_transfer": false,
      "turbulence_model": "kEpsilon"
    },
    "solver": {
      "type": "simpleFoam",
      "max_iterations": 2000,
      "convergence_criteria": 1e-6,
      "write_interval": 100
    },
    "fluid": {
      "name": "water",
      "density": 1000,
      "kinematic_viscosity": 1e-6
    },
    "geometry": {
      "type": "pipe",
      "diameter": 0.1,
      "length": 1.0
    },
    "boundary_conditions": {
      "inlet": {
        "patch_type": "inlet",
        "velocity": {
          "type": "fixedValue",
          "value": [5, 0, 0]
        },
        "pressure": {
          "type": "zeroGradient"
        }
      },
      "outlet": {
        "patch_type": "outlet",
        "velocity": {
          "type": "zeroGradient"
        },
        "pressure": {
          "type": "fixedValue",
          "value": 0
        }
      },
      "wall": {
        "patch_type": "wall"
      }
    },
    "kpi_targets": [
      { "name": "pressure_drop", "value": 100, "unit": "Pa" }
    ]
  },
  "constraints": {
    "max_retries": 3,
    "timeout_seconds": 600
  },
  "metadata": {
    "user_id": "user-123",
    "project_id": "proj-456"
  }
}
```

**Required fields for `CFD_CODEGEN_RUN`:**
- `mesh.mesh_id` - Mesh identifier
- `boundary_conditions.inlet` with `velocity` specified
- `boundary_conditions.outlet`

If these are missing, the backend will emit `config_incomplete` event and stop.

### Operations

| Operation | Description |
|-----------|-------------|
| `CFD_LINT` | Validate/normalize config, detect regime, recommend solver |
| `CFD_CODEGEN_RUN` | Generate OpenFOAM case and execute in sandbox |

### Server → Client: AgentEvent

```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "seq": 5,
  "ts": "2024-01-15T10:30:00.000Z",
  "level": "info",
  "type": "lint_result",
  "message": "Linting complete: 0 issues, 3 recommendations",
  "payload": {
    "validated_config": {...},
    "apply_changes": [...],
    "regime": "turbulent",
    "reynolds_number": 100000
  }
}
```

## Event Types

### Lifecycle Events

| Type | Description |
|------|-------------|
| `run_started` | Run has begun processing |
| `run_succeeded` | Run completed successfully |
| `run_failed` | Run failed after retries |
| `simulation_not_clear` | Couldn't determine simulation type |

### Config Validation Events (NEW)

| Type | Description |
|------|-------------|
| `config_received` | Config parsed, shows what keys were found |
| `config_normalized` | Config converted to canonical V1 format |
| `config_incomplete` | Required fields missing, codegen blocked |

### Linting Events

| Type | Description |
|------|-------------|
| `lint_started` | CFD linting has started |
| `lint_result` | Linting complete with results |

### Planning Events

| Type | Description |
|------|-------------|
| `planning_started` | Planning phase started |
| `planning_complete` | Planning complete with work items |
| `subagent_started` | Sub-agent task started |
| `subagent_update` | Sub-agent progress update |
| `subagent_complete` | Sub-agent task completed |

### Code Generation Events

| Type | Description |
|------|-------------|
| `codegen_started` | Code generation started |
| `codegen_iteration` | Files generated in this iteration |
| `codegen_complete` | Code generation complete |

### Sandbox Events

| Type | Description |
|------|-------------|
| `sandbox_submitted` | Case submitted to sandbox |
| `sandbox_status` | Sandbox status update |
| `sandbox_logs` | Execution logs from sandbox |
| `sandbox_succeeded` | Sandbox run succeeded |
| `sandbox_failed` | Sandbox run failed |

### Self-Healing Events

| Type | Description |
|------|-------------|
| `error_summary` | Analysis of sandbox failure |
| `retrying` | Starting retry attempt |

### Final Event

| Type | Description |
|------|-------------|
| `final` | Final result with complete summary |

## CFD Linting

The linter performs these checks:

1. **Units Validation**
   - All dimensions must be positive
   - Viscosity must be positive
   - Velocity magnitude checked for compressibility

2. **Reynolds Number Calculation**
   - `Re = (U × D) / ν`
   - Regime detection: laminar (<2300), transitional (2300-4000), turbulent (>4000)

3. **Solver Selection**
   - Laminar: `simpleFoam` + `laminar` model
   - Turbulent: `simpleFoam` + RANS (`kEpsilon`, `kOmegaSST`)
   - Heat transfer: `buoyantSimpleFoam` variants

4. **Mesh Guidance**
   - Resolution recommendations based on Reynolds number
   - Grading suggestions for boundary layers

5. **Boundary Conditions**
   - Checks for inlet/outlet/wall definitions
   - Validates BC coherence

## Adding Custom Lint Rules

Add new rules to `simd_agent/linting.py`:

```python
def _validate_custom(self, config: dict) -> tuple[list[LintIssue], list[ApplyChange]]:
    issues = []
    changes = []
    
    # Your validation logic here
    if config.get("my_field") is None:
        issues.append(LintIssue(
            code="MISSING_FIELD",
            path="my_field",
            message="my_field is required",
            severity="error",
        ))
    
    return issues, changes
```

Then call it from the main `lint()` method.

## Adding New Providers

The service uses `codegen` for LLM operations. To add a new provider:

1. Implement the provider in `codegen`
2. Add the API key environment variable to `settings.py`:

```python
my_provider_api_key: str | None = Field(
    default=None,
    description="My Provider API key",
)
```

3. Update the `default_provider` options in `settings.py`

## Project Structure

```
simd_agent/
├── __init__.py          # Package exports
├── main.py              # FastAPI app + WebSocket endpoint
├── settings.py          # Environment configuration
├── models.py            # Pydantic models
├── db.py                # Database engine + session
├── store.py             # EventStore for persistence
├── orchestration.py     # Main workflow orchestrator
├── linting.py           # CFDLinter service
├── planning.py          # Planner + parallel sub-agents
├── sandbox_client.py    # Sandbox HTTP client
├── packaging.py         # OpenFOAM case packaging
├── error_summarizer.py  # Error analysis agent
├── event_bus.py         # Event emission + streaming
└── prompts/
    ├── __init__.py      # Prompt loader
    └── packs/
        └── simd/
            ├── system.md
            ├── lint.md
            ├── planner.md
            ├── codegen_openfoam.md
            └── error_summary.md

tests/
├── conftest.py              # Fixtures
├── test_ws_protocol.py      # Protocol tests
├── test_linting.py          # Linting tests
└── test_orchestration_mock.py  # Mock orchestration tests
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/ws/run` | WebSocket | Main run endpoint |
| `/runs/{run_id}` | GET | Get run details |
| `/runs/{run_id}/events` | GET | Get run events |

## Database Schema

### runs table

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| created_at | TIMESTAMPTZ | Creation time |
| op | VARCHAR | Operation type |
| status | VARCHAR | Run status |
| provider | VARCHAR | LLM provider |
| prompt_pack | VARCHAR | Prompt pack name |
| user_requirements | TEXT | User's requirements |
| simulation_config | JSONB | Original config |
| validated_config | JSONB | Validated config |
| attempts | INTEGER | Number of attempts |
| result | JSONB | Final result |

### events table

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| run_id | UUID | Foreign key to runs |
| seq | INTEGER | Sequence number |
| ts | TIMESTAMPTZ | Event timestamp |
| level | VARCHAR | Event level |
| type | VARCHAR | Event type |
| message | TEXT | Human-readable message |
| payload | JSONB | Event data |

