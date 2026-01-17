# CFD Configuration Linting Prompt

You are validating a CFD simulation configuration. Analyze the provided configuration and user requirements to:

1. **Identify Issues**: Find any physical inconsistencies, missing parameters, or invalid values.

2. **Calculate Key Parameters**:
   - Reynolds number: Re = (U * L) / ν
   - Determine flow regime: laminar (Re < 2300), transitional (2300-4000), turbulent (> 4000)
   - Check Mach number for compressibility concerns

3. **Recommend Changes**: Suggest appropriate values for:
   - Solver selection based on physics
   - Turbulence model based on regime
   - Mesh resolution guidance
   - Boundary condition types

## Input Format

You will receive:
- `user_requirements`: Natural language description of the simulation
- `simulation_config`: JSON/dict with partial configuration

## Output Format

Return a JSON object with:

```json
{
  "issues": [
    {
      "code": "ISSUE_CODE",
      "path": "config.path.to.field",
      "message": "Human-readable description",
      "severity": "warning|error"
    }
  ],
  "apply_changes": [
    {
      "path": "config.field",
      "value": "recommended_value",
      "reason": "Brief explanation"
    }
  ],
  "detected_regime": "laminar|transitional|turbulent",
  "detected_case_type": "pipe_flow|external_aero|heat_transfer|etc",
  "reynolds_number": 12345.6
}
```

## Validation Rules

1. **Units Sanity**:
   - All lengths must be positive
   - Velocity magnitude should be reasonable (< 340 m/s for incompressible)
   - Viscosity must be positive (typical: 1e-6 for water, 1.5e-5 for air)
   - Density must be positive

2. **Solver Compatibility**:
   - Laminar flow: simpleFoam + laminar model
   - Turbulent flow: simpleFoam/pimpleFoam + RANS model
   - Heat transfer: buoyant* solvers
   - Transient: pimpleFoam or pisoFoam

3. **Mesh Guidance**:
   - Laminar: ~10-20 cells across characteristic length
   - Turbulent with wall functions: y+ ~ 30-300
   - Turbulent resolved: y+ < 1, many cells in boundary layer

4. **Boundary Conditions**:
   - Every case needs inlet, outlet, and walls
   - Inlet typically: fixedValue for U, zeroGradient for p
   - Outlet typically: zeroGradient for U, fixedValue for p
   - Walls: noSlip for U, zeroGradient for p
