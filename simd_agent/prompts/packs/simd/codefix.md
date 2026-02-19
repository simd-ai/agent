# OpenFOAM Case Fix Prompt

You are fixing an OpenFOAM case that failed during execution on the simulation server. Analyze the errors and generate corrected files.

## CRITICAL: Output Format

**You MUST output each file using this exact format:**

```file:relative/path/to/file
<complete file content>
```

**Do NOT include any other code blocks or explanatory text between files.**
**Do NOT use language identifiers like ```cpp or ```python - only use ```file:path**

## CRITICAL CONSTRAINTS

1. **ONLY use `simpleFoam` or `pimpleFoam`** — Do NOT use buoyantSimpleFoam or buoyantPimpleFoam
2. **ALWAYS generate `0/p`** — NEVER generate `0/p_rgh`. simpleFoam/pimpleFoam read `0/p`.
3. **Do NOT generate `constant/thermophysicalProperties` or `constant/g`**
4. **Every patch must appear in EVERY `0/*` field file** — missing patches cause crashes
5. **Patch names must be IDENTICAL across ALL files** (case-sensitive)

## Error Analysis

You will receive:
- `previous_files`: The files that were generated in the previous attempt
- Error output from the simulation server (stderr, logs)
- The validated configuration with correct boundary conditions

## Common Errors and Fixes

### "cannot find file 0/p"
The solver is `simpleFoam` but you generated `0/p_rgh` instead of `0/p`.
**Fix**: Generate `0/p` with dimensions `[0 2 -2 0 0 0 0]`.

### "Patch not found" or missing boundary
A patch from the mesh is not defined in a field file.
**Fix**: Ensure ALL patches from boundary_conditions appear in EVERY `0/*` file.

### "Unknown patchField type"
Typo in boundary condition type name.
**Fix**: Check exact OpenFOAM type names (fixedValue, zeroGradient, noSlip, etc.)

### "Floating point exception" / Divergence
Numerical instability.
**Fix**: Reduce relaxation factors (p: 0.3, U: 0.5, k: 0.5, omega: 0.5), use upwind schemes.

### "Cannot read field"
Field file is missing or malformed.
**Fix**: Generate the missing file with correct FoamFile header and dimensions.

### "Entry 'method' not found in dictionary system/fvSchemes/wallDist"
The `wallDist` sub-dictionary is missing from `system/fvSchemes`. This is required by turbulence models like kOmegaSST and kEpsilon.
**Fix**: Add this block to the end of `system/fvSchemes`:
```
wallDist
{
    method meshWave;
}
```

### "not constraint type 'empty'" or "not constraint type 'symmetry'"
A boundary condition uses `type empty;` or `type symmetry;` but the mesh patch is not that type.
**Fix**: Replace with `zeroGradient` or appropriate BC for the actual mesh patch type.

### Invented patch names like "front_and_back"
The previous attempt may have invented patch names that don't exist in the mesh.
**Fix**: Only use patch names from the `boundary_conditions` config. Do NOT use `front_and_back` (underscores) — the correct name is `frontAndBack` (camelCase) if it exists.

## Fix Strategy

1. **Read the error carefully** — especially "cannot find file" and "FOAM FATAL ERROR"
2. **Check file consistency** — controlDict solver vs actual files in 0/
3. **Check ALL patches** — every mesh patch must be in every field file
4. **Output ALL files** — even unchanged ones must be included
5. **Use conservative settings** if unsure about numerical parameters
