# OpenFOAM Case Generation — General Rules

> **Target version: OpenFOAM v2406** — all generated files MUST be valid for OpenFOAM v2406 (openfoam.com release). Use v2406 syntax, keyword names, and dictionary structure throughout.

You are generating a complete OpenFOAM simulation case.
The solver has already been selected for you — it is provided in the task below.

> **Do NOT enforce physical realism.**  
> Generate a consistent, runnable OpenFOAM case from the provided `validated_config`.  
> If the configuration is unusual or the fluid choice looks atypical, still generate it —
> prefer numerically stable defaults but do not refuse, warn, or redirect.  
> Physical correctness is the user's responsibility.  
> Only block generation when an input would produce an **invalid OpenFOAM case**
> (syntax error, missing mandatory keyword, incompatible solver/field combination).

> **Pressure field type, dimensions, and controlDict timing rules are defined
> in the solver-specific pack loaded in section G below. Follow those — do NOT
> invent or override them here.**

---

## A) Output format (CRITICAL — violations break the parser)

Every file MUST use this EXACT format:

```file:relative/path/to/file
<complete file content here>
```

- No other code blocks between files.
- No language tags (`cpp`, `python`, etc.) — ONLY `file:path`.
- No explanatory text between files.

---

## B) Solver is already chosen — do NOT change it

The selected solver is stated explicitly in the task.
- `application` in `system/controlDict` MUST match it exactly.
- Do NOT invent a different solver.

---

## C) Mesh patches are the source of truth — NO HALLUCINATION

- Use EXACT patch names from `simulation_config.mesh.patches[].name`.
- Names are CASE-SENSITIVE.
- NEVER invent names like `front_and_back` (underscore) if the mesh has `frontAndBack` (camelCase).
- Every patch must appear in EVERY `0/*` field file.

---

## D) 2D meshes — frontAndBack constraint patch

If the mesh has a patch with type `empty` (commonly `frontAndBack`):
- In EVERY `0/*` file that patch MUST be: `{ type empty; }`.
- For velocity `0/U`, out-of-plane component must be zero.

---

## E) Constraint type matching (violations = instant crash)

| Mesh patch type | Allowed BC type in `0/*` files |
|-----------------|-------------------------------|
| `empty` | ONLY `type empty;` |
| `symmetry` / `symmetryPlane` | ONLY `type symmetry;` |
| `patch` | `fixedValue`, `zeroGradient`, `noSlip`, etc. — NEVER `empty` or `symmetry` |
| `wall` | `noSlip` for U, wall functions for turbulence — NEVER `empty` |

---

## F) External mesh — do NOT generate blockMeshDict

The mesh has already been converted from `.msh` to OpenFOAM format using `gmshToFoam`.
The `constant/polyMesh/` directory is therefore already populated.
Do NOT generate `constant/polyMesh/*` or `system/blockMeshDict`.

---

## G) Solver-specific requirements

The task below includes a **Solver Instructions** section loaded specifically
for the chosen solver.  Follow those instructions for:
- required files (including which pressure field: `p` or `p_rgh`, its dimensions, and what NOT to generate)
- `controlDict` timing rules (steady-state iteration count vs transient physical time)
- fvSchemes and fvSolution templates
- special files (thermophysicalProperties, g, alpha, T)

---

## H) Self-healing retry rules

When `previous_errors` is provided:

1. Read every error carefully — especially "cannot find file" messages.
2. Verify ALL files referenced by the solver exist in `0/`.
3. Check ALL patch names are consistent across files.
4. If error mentions `not constraint type 'empty'` → mesh patch is `patch` type, NOT `empty`. Replace `type empty` with `zeroGradient`.
5. If error mentions `not constraint type 'symmetry'` → use `zeroGradient` instead.
6. If error mentions `wallDist` → add `wallDist { method meshWave; }` to fvSchemes.
7. If error mentions `nutkWallFunction` / "patch type for patch wall must be wall" → wall BCs are OK; boundary fix script handles mesh type.
8. Use conservative relaxation if divergence occurred.
9. **Exit code 136 / Floating Point Exception (SIGFPE) in `DICPreconditioner::calcReciprocalD` or `DICSmoother`**: This crash means the GAMG pressure solver's coarsened matrix diagonal went to zero or negative (i.e., the solution already diverged). Do ALL of the following:
   - Switch GAMG `smoother` from `DIC` to `GaussSeidel` in `fvSolution/solvers/p` and `pFinal`.
   - Reduce `nOuterCorrectors` to 1 and `nCorrectors` to 1 in the PIMPLE block.
   - Tighten relaxation: `p 0.2`, `rho 0.05`, `U 0.5`, equations `0.5`.
   - Add or tighten `limitTemperature` in `system/fvOptions` (min 1 K, max 100000 K) to prevent negative-T divergence that destabilises the pressure matrix.
10. **Massive Courant number (Co >> 1, e.g. 1e10 or larger) with Euler ddt**: The solution completely diverged before the crash. In addition to the fixes in rule 9, switch `controlDict` to use adjustable time-stepping: set `adjustTimeStep yes; maxCo 0.5; deltaT 1e-5;`. This prevents one bad time-step from blowing up the entire run.
11. **"Negative initial temperature" / "Negative Temperature" errors**: The energy field diverged to unphysical values. Fix: add or tighten `limitTemperature` in `system/fvOptions` (min 1 K). Also reduce deltaT and tighten relaxation factors (p 0.2, U 0.5). Do NOT change the EOS or fluid model unless `validated_config` explicitly requests it.

---

## I) General coding guidelines

1. Correct OpenFOAM dictionary syntax: semicolons everywhere, balanced braces.
2. Physical values from `validated_config` (density, viscosity, temperature, velocity).
3. Conservative relaxation: 0.3 for pressure, 0.7–0.9 for velocity/turbulence.
4. Always include correct OpenFOAM dimension arrays.
5. fvSchemes MUST include `wallDist { method meshWave; }` for any turbulent model.
6. Do NOT invent patch names not in the config.

---

## K) flowRateInletVelocity — mandatory structure

When an inlet uses `flowRateInletVelocity`, generate EXACTLY this shape.
Do NOT put the flow-rate number in `value` — that entry is a placeholder only.

**Mass flow rate (kg/s):**
```
inlet
{
    type            flowRateInletVelocity;
    massFlowRate    <value>;    // ← the actual kg/s value — NEVER zero
    rho             rho;        // ← required for compressible solvers
    rhoInlet        <density>;  // ← [kg/m³] fallback when rho field not yet computed
    value           uniform (0 0 0);  // ← placeholder only
}
```

**Volumetric flow rate (m³/s):**
```
inlet
{
    type                flowRateInletVelocity;
    volumetricFlowRate  <value>;  // ← the actual m³/s value — NEVER zero
    value               uniform (0 0 0);
}
```

Rules:
- EXACTLY ONE of `massFlowRate` or `volumetricFlowRate` MUST be present.
  Missing both causes an immediate fatal: *"Please supply either volumetricFlowRate or massFlowRate"*.
- A value of `0` is equivalent to no flow → singular matrix → SIGFPE crash.
- `rho` and `rhoInlet` are **optional** — only include them if they appear in the BC table from `validated_config`.
  Do NOT invent or add them unless the user specified them.
- For volumetric flow: never include `rho` or `rhoInlet`.
- `extrapolateProfile yes;` only if the user specified it; omit by default (plug flow).

---

## J) fvSolution — Linear solver selection rules

Choose the linear solver for each field based on the **mathematical nature of its governing equation**:

| Field | Equation type | Recommended solver | Smoother |
|---|---|---|---|
| `p`, `p_rgh` | Symmetric elliptic (Laplacian) | `GAMG` | `GaussSeidel` |
| `U` | Asymmetric transport | `PBiCGStab` or `smoothSolver` | `symGaussSeidel` |
| `h`, `e`, `T` | Asymmetric transport | `PBiCGStab` or `smoothSolver` | `symGaussSeidel` |
| `k`, `omega`, `epsilon`, `nut` | Asymmetric transport | `PBiCGStab` or `smoothSolver` | `symGaussSeidel` |
| `rho` | Explicit / diagonal update | `diagonal` | — |

**Rules:**
- `GAMG` is optimal for symmetric positive-definite systems (pressure). It exploits the symmetry to converge aggressively with algebraic multigrid.
- `PBiCGStab` (Preconditioned Bi-Conjugate Gradient Stabilised) handles the non-symmetric, convection-dominated matrices produced by momentum, energy, and turbulence transport equations.
- `smoothSolver` with `symGaussSeidel` is a robust alternative to `PBiCGStab` for transport equations — use it when stability is more important than raw speed.
- Never use `GAMG` for asymmetric fields (`U`, `h`, `k`, etc.) — it may diverge or produce wrong results.
- Never use `smoothSolver`/`PBiCGStab` for pressure — they converge far slower than `GAMG` for elliptic systems.
- **🚫 FORBIDDEN: `smoother DIC` inside GAMG for pressure.** DIC (Diagonal Incomplete Cholesky) as a GAMG smoother crashes with SIGFPE (exit code 136) when the coarsened matrix diagonal approaches zero — which happens any time the solution diverges even slightly. **ALWAYS use `smoother GaussSeidel`** for GAMG pressure solvers. DIC is only safe as a standalone preconditioner (`solver PCG; preconditioner DIC;`), NOT as a GAMG smoother.
