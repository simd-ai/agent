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

## Solver Selection (CRITICAL)

You MUST select the solver based on these rules:
- **Steady-state incompressible flow** → `simpleFoam`
- **Transient incompressible flow** → `pimpleFoam`

**Do NOT use buoyantSimpleFoam or buoyantPimpleFoam** — buoyancy is not supported yet.
Even if heat transfer is mentioned, use `simpleFoam` (steady) or `pimpleFoam` (transient).

## File Consistency Rules (CRITICAL — violations cause crashes)

1. The `application` field in `system/controlDict` determines which solver runs.
2. `simpleFoam` reads the file `0/p` — you MUST generate `0/p`, NOT `0/p_rgh`.
3. `pimpleFoam` reads the file `0/p` — you MUST generate `0/p`, NOT `0/p_rgh`.
4. Every field file in `0/` must list ALL patch names from the mesh. Missing patches = crash.
5. Patch names must be IDENTICAL across ALL files (case-sensitive).
6. Do NOT generate `constant/thermophysicalProperties` or `constant/g` — not needed without buoyancy.

## Required Files

Generate ALL of these files:

### system/ directory
1. `system/controlDict` — Simulation control
2. `system/fvSchemes` — Discretization schemes
3. `system/fvSolution` — Solver settings

**Do NOT generate blockMeshDict — we use an external mesh file converted by the simulation server.**

### 0/ directory (initial conditions)
4. `0/U` — Velocity field
5. `0/p` — Pressure field (ALWAYS `p`, never `p_rgh`)
6. `0/k` — Turbulent kinetic energy (if turbulent)
7. `0/omega` — Specific dissipation rate (if kOmegaSST)
8. `0/epsilon` — Turbulent dissipation (if kEpsilon)
9. `0/nut` — Turbulent viscosity (if turbulent)
10. `0/T` — Temperature field (only if heat_transfer is true in config)

### constant/ directory
11. `constant/transportProperties` — Fluid properties (nu)
12. `constant/turbulenceProperties` — Turbulence model settings (if turbulent)

## Boundary Condition Rules

Use the EXACT boundary condition types and values from the validated_config.

For each patch in `boundary_conditions`:
- **inlet**: Use the exact velocity, pressure, turbulence values from config
- **outlet**: Use the exact pressure, velocity types from config
- **wall**: Use `noSlip` for U, wall functions for turbulence fields
- **symmetry**: Use `type symmetry;` for ALL fields — BUT ONLY if the mesh patch type is `symmetry` or `symmetryPlane`
- **empty**: Use `type empty;` for ALL fields — BUT ONLY if the mesh patch type is `empty`

**CRITICAL: Do NOT invent patch names.** Only use patch names that are explicitly listed in the `boundary_conditions` or `mesh.patches` configuration. In particular:
- Do **NOT** generate a `front_and_back` patch (with underscores). This does not exist in the mesh.
- The correct 2D patch name is `frontAndBack` (camelCase) — only include it if it appears in the config.

## 2D Simulations (CRITICAL)

For 2D simulations (meshes created in Gmsh for 2D geometries):
- The mesh will have a `frontAndBack` patch of type `empty` after conversion with `gmshToFoam`.
- You **MUST** include `frontAndBack` with `type empty;` in ALL `0/*` field files (U, p, T, k, omega, nut, epsilon).
- If the mesh patches include `frontAndBack`, it MUST appear in every field file or OpenFOAM will crash.
- A post-mesh-conversion fix script will also ensure this, but include it in generated code for correctness.

## CRITICAL: Constraint Type Matching (violations = instant crash)

OpenFOAM constraint types (`empty`, `symmetry`, `wedge`, `cyclic`) MUST match the mesh patch type.
- If mesh says patch type is `patch` → you CANNOT use `type empty;` or `type symmetry;`. Use `zeroGradient` instead.
- If mesh says patch type is `wall` → use wall BCs (`noSlip`, wall functions). NEVER `empty`.
- If mesh says patch type is `empty` → you MUST use `type empty;`.
- If mesh says patch type is `symmetry` → you MUST use `type symmetry;`.

The mesh patch types are provided in the validated configuration under `mesh.patches`.

For `0/T` (temperature field):
- Only generate if `heat_transfer: true` in the physics config
- Use `fixedValue` with the temperature from boundary_conditions
- Use `zeroGradient` for outlets
- For walls with specified temperature, use `fixedValue`

## fvSchemes: wallDist (CRITICAL for turbulent models)

If the simulation uses a turbulence model (kOmegaSST, kEpsilon, etc.), you **MUST** include a `wallDist` sub-dictionary in `system/fvSchemes`:

```
wallDist
{
    method meshWave;
}
```

Place it at the end of fvSchemes, after `snGradSchemes`. Without this, OpenFOAM will crash with:
`Entry 'method' not found in dictionary "system/fvSchemes/wallDist"`

## Guidelines

1. **Correct Syntax**: Use proper OpenFOAM dictionary syntax with semicolons and braces.
2. **Patch Names**: Use the EXACT patch names from the validated_config — these match the mesh.
3. **Physical Values**: Use values from the validated_config (density, viscosity, etc.).
4. **Conservative Settings**: Use relaxation factors 0.3 for p, 0.7 for U/k/omega/epsilon.
5. **End Time**: Set `endTime` to the value of `solver.max_iterations` from the validated_config. Always use `startFrom startTime; startTime 0;` in controlDict.
6. **Dimensions**: Always include correct OpenFOAM dimensions arrays.

## Pressure Dimensions

For `simpleFoam` and `pimpleFoam` (incompressible solvers):
- Pressure dimensions: `[0 2 -2 0 0 0 0]` (kinematic pressure, m²/s²)

## If Retrying After Error

When `previous_errors` is provided:
1. Read the error carefully — especially "cannot find file" errors
2. Check that ALL files referenced by the solver exist in `0/`
3. Check that ALL patch names match across files
4. Apply conservative settings if divergence occurred
