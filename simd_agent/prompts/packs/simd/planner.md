# CFD Simulation Planning Prompt

You are planning the setup of a CFD simulation. Based on the validated configuration and user requirements, create a detailed execution plan.

## Input

You will receive:
- `user_requirements`: What the user wants to simulate
- `validated_config`: Configuration after linting/validation
- `case_type`: Detected simulation type
- `regime`: Flow regime (laminar/transitional/turbulent)
- `reynolds_number`: Calculated Reynolds number

## Planning Tasks

Break down the simulation setup into these work items:

### 1. Solver Selection
- Choose appropriate OpenFOAM solver
- Consider: steady/transient, incompressible/compressible, single/multiphase, heat transfer

### 2. Turbulence Model Selection
- For laminar: no model needed
- For turbulent: select RANS model (k-epsilon, k-omega SST, etc.)
- Consider wall treatment requirements

### 3. Mesh Strategy
- Determine mesh type (blockMesh, snappyHexMesh)
- Calculate required resolution
- Define grading for boundary layers
- Consider symmetry and dimensionality

### 4. Boundary Conditions
- Define all patches (inlet, outlet, walls, symmetry)
- Specify appropriate BC types for each field
- Set initial values

### 5. Numerical Schemes
- Select discretization schemes
- Choose interpolation methods
- Configure gradient limiters if needed

### 6. Solution Control
- Set relaxation factors
- Define convergence criteria
- Configure time stepping (if transient)

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
    "turbulence_model": "kEpsilon",
    "mesh_type": "blockMesh",
    "mesh_cells": [20, 20, 1],
    "time_stepping": "steady"
  },
  "rationale": "Brief explanation of key decisions"
}
```
