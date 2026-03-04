# CFD Simulation Planning Prompt

You are planning the setup of a CFD simulation. Based on the validated configuration and user
requirements, select the correct solver and create a detailed execution plan.

---

## Input

You will receive:
- `user_requirements`: What the user wants to simulate
- `validated_config`: Configuration after linting/validation
- `case_type`: Detected simulation type
- `regime`: Flow regime (laminar/transitional/turbulent)
- `reynolds_number`: Calculated Reynolds number

---

## Available Solvers (complete reference)

Only these solvers exist in this system. Do NOT invent others.

| Solver | Time | Compressible | Phases | Energy/Temp | Gravity | Pressure field |
|--------|------|-------------|--------|-------------|---------|----------------|
| `simpleFoam` | Steady | ❌ | 1 | ❌ | ❌ | `p` kinematic `[0 2 -2 0 0 0 0]` |
| `icoFoam` | Transient | ❌ | 1 | ❌ | ❌ | `p` kinematic `[0 2 -2 0 0 0 0]` |
| `pimpleFoam` | Transient | ❌ | 1 | ❌ | ❌ | `p` kinematic `[0 2 -2 0 0 0 0]` |
| `rhoSimpleFoam` | Steady | ✅ | 1 | ✅ h/e | ❌ | `p` absolute Pa `[1 -1 -2 0 0 0 0]` |
| `rhoPimpleFoam` | Transient | ✅ | 1 | ✅ h/e + `0/T` | ❌ | `p` absolute Pa `[1 -1 -2 0 0 0 0]` |
| `interFoam` | Transient | ❌ | 2 | ❌ | ✅ | `p_rgh` `[1 -1 -2 0 0 0 0]` |
| `interIsoFoam` | Transient | ❌ | 2 | ❌ | ✅ | `p_rgh` `[1 -1 -2 0 0 0 0]` |
| `compressibleInterFoam` | Transient | ✅ | 2 | ✅ `0/T` | ✅ | `p_rgh` `[1 -1 -2 0 0 0 0]` |
| `compressibleInterIsoFoam` | Transient | ✅ | 2 | ✅ `0/T` | ✅ | `p_rgh` `[1 -1 -2 0 0 0 0]` |
| `compressibleMultiphaseInterFoam` | Transient | ✅ | N≥3 | ✅ `0/T` | ✅ | `p_rgh` `[1 -1 -2 0 0 0 0]` |

---

## Solver Selection Decision Tree

### Step 1 — How many phases?

- **N ≥ 3 immiscible phases** → `compressibleMultiphaseInterFoam`
- **2 phases** → go to Step 2
- **1 phase** → go to Step 3

### Step 2 — Two-phase: compressible or not?

- **Compressible** (density varies significantly, heat transfer involved):
  - Standard interface (MULES) → `compressibleInterFoam`
  - Sharp interface needed (droplets, thin films, jets) → `compressibleInterIsoFoam`
- **Incompressible** (free surface, waves, sloshing, water/air):
  - Standard interface → `interFoam`
  - Sharp interface needed → `interIsoFoam`

> Both `interFoam`/`interIsoFoam` families ALWAYS require `constant/g` even when gravity = false.

### Step 3 — Single phase: compressible or not?

- **Compressible** (Mach > ~0.3, gas with significant density/temperature variation, heat transfer required):
  - Steady-state → `rhoSimpleFoam`
  - Transient → `rhoPimpleFoam`
- **Incompressible**:
  - **Steady-state** → `simpleFoam`
  - **Transient + laminar only** (Re < ~2300, no turbulence model) → `icoFoam`
  - **Transient + turbulence possible** → `pimpleFoam`

> `icoFoam` is laminar-only (no turbulence files). If there is any chance of turbulence, prefer `pimpleFoam`.

---

## Turbulence Model Selection

| Regime | Model | Fields generated |
|--------|-------|-----------------|
| Laminar | none | No turbulence files |
| Turbulent (general) | `kOmegaSST` (preferred) | `0/k`, `0/omega`, `0/nut` |
| Turbulent (legacy/simpler) | `kEpsilon` | `0/k`, `0/epsilon`, `0/nut` |
| LES | `Smagorinsky` or `WALE` | depends on model |

- `icoFoam`: turbulence is **never** used — do not set a turbulence model.
- Compressible solvers: use `0/alphat` (turbulent thermal diffusivity) when turbulence + energy are both active.
- Compressible multiphase: use `0/mut` instead of `0/nut`.

---

## Planning Tasks

### 1. Solver Selection
Apply the decision tree above. State the chosen solver and the reasons (phases, compressibility, steady/transient).

### 2. Turbulence Model Selection
Use the table above. For turbulent cases, default to `kOmegaSST` unless there is a reason to prefer another.

### 3. Mesh Strategy
- The mesh is provided externally (`gmshToFoam`). Do NOT plan `blockMeshDict` generation.
- Note dimensionality (2D → 1-cell thick mesh with `empty` frontAndBack patch).

### 4. Boundary Conditions
- Use EXACT patch names from `config.mesh.patches[].name`.
- Every patch must appear in every `0/*` field file.
- Empty patches (2D): `{ type empty; }` in ALL field files.

### 5. Numerical Schemes & Solution Control
- Follow the fvSchemes/fvSolution templates in the chosen solver pack.
- Steady solvers: `endTime` = max_iterations, `deltaT 1`.
- Transient solvers: `endTime` = physical seconds, `deltaT` from config.

---

## Output Format

Return a JSON object with:

```json
{
  "work_items": [
    {
      "id": "unique_id",
      "task": "task_name",
      "description": "What this task does",
      "priority": 1,
      "dependencies": ["other_task_id"]
    }
  ],
  "decisions": {
    "solver": "simpleFoam",
    "turbulence_model": "kOmegaSST",
    "time_stepping": "steady",
    "phases": 1,
    "compressible": false
  },
  "rationale": "Brief explanation of key decisions"
}
```
