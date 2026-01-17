# OpenFOAM Case Generation Prompt

You are generating a complete OpenFOAM case. Create all necessary files for a working simulation.

## CRITICAL: Output Format

**You MUST output each file using this exact format:**

```file:relative/path/to/file
<complete file content>
```

For example:

```file:system/controlDict
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     simpleFoam;
...
```

**Do NOT include any other code blocks or explanatory text between files.**
**Do NOT use language identifiers like ```cpp or ```python - only use ```file:path**

## Required Files

Generate ALL of these files:

### system/ directory
1. `system/controlDict` - Simulation control
2. `system/fvSchemes` - Discretization schemes
3. `system/fvSolution` - Solver settings
4. `system/blockMeshDict` - Mesh definition (if using blockMesh)

### 0/ directory (initial conditions)
5. `0/U` - Velocity field
6. `0/p` - Pressure field
7. Additional fields as needed (k, epsilon, omega, nut, T, etc.)

### constant/ directory
8. `constant/transportProperties` - Fluid properties
9. `constant/turbulenceProperties` - Turbulence settings (if turbulent)

## Input

You will receive:
- `requirements`: User's simulation requirements
- `validated_config`: Validated configuration with solver, turbulence model, etc.
- `case_type`: Type of simulation (pipe_flow, external_aero, etc.)
- `solver`: Selected OpenFOAM solver
- `turbulence_model`: Selected turbulence model
- `mesh_strategy`: Mesh resolution guidance
- `previous_errors`: (If retrying) Previous execution errors and their causes

## Guidelines

1. **Minimal First**: Create the simplest case that works. Avoid unnecessary complexity.

2. **Correct Syntax**: Use proper OpenFOAM dictionary syntax with semicolons and braces.

3. **Matching Names**: Ensure patch names match exactly between blockMeshDict and field files.

4. **Physical Values**: Use realistic values for the specified case type.

5. **Convergence Focus**: Set conservative relaxation factors and robust schemes for initial runs.

6. **Short Runtime**: Set endTime and writeInterval for a quick validation run (e.g., 100-1000 iterations).

## If Retrying After Error

When `previous_errors` is provided:
1. Carefully read the error summary
2. Apply the suggested fixes
3. Only modify files that need fixing
4. Keep working parts unchanged

## Common Fixes

- **blockMesh errors**: Check vertex numbering, ensure right-hand rule for blocks
- **Boundary mismatch**: Ensure all patches in blockMesh are defined in 0/* files
- **Divergence**: Reduce relaxation factors, use upwind schemes
- **Negative cells**: Fix vertex ordering or grading

## Example Case Structure

A minimal pipe flow case would have:

```file:system/controlDict
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1000;
deltaT 1;
writeControl timeStep;
writeInterval 100;
```

```file:system/fvSchemes
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
```

... and so on for all required files.
