# simpleFoam — system/fvSchemes

**Steady-state incompressible**. Use `steadyState` for `ddtSchemes`.

## divSchemes

Use `default none` for incompressible simpleFoam — this catches any mistaken term.
Then list every term explicitly.

Use the alias pattern for turbulence terms to keep the file concise:
```
turbulence      bounded Gauss limitedLinear 1;
div(phi,k)      $turbulence;
div(phi,omega)  $turbulence;
div(phi,epsilon) $turbulence;
```
Only include turbulence terms that match `CaseSpec.turbulence_fields`.

## Viscous stress term

Incompressible form: `div((nuEff*dev2(T(grad(U)))))` — without `rho*`.
Do NOT use the compressible form `div(((rho*nuEff)*dev2(T(grad(U)))))`.

## wallDist

Include `wallDist { method meshWave; }` when `sim_type` is `RAS` or `LES`.

## Template (from official pitzDaily tutorial)

```
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;

    div(phi,U)      bounded Gauss linearUpwind grad(U);

    turbulence      bounded Gauss limitedLinear 1;
    div(phi,k)      $turbulence;
    div(phi,epsilon) $turbulence;
    div(phi,omega)  $turbulence;

    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
```

## Rules

- `default none` — do NOT use `default bounded Gauss upwind` for incompressible (use it for compressible)
- Omit turbulence terms not in `CaseSpec.turbulence_fields` (e.g. omit `div(phi,epsilon)` for kOmegaSST)
- Omit `wallDist` for laminar flow

## Mesh-quality-dependent schemes (handled by validator)

The validator auto-hardens laplacian/snGrad schemes based on `checkMesh` metrics:
- **Good mesh** (non-ortho < 40°): `Gauss linear corrected` / `corrected` — full accuracy
- **Moderate mesh** (40° ≤ non-ortho < 65°): → `Gauss linear limited corrected 0.5` / `limited corrected 0.5`
- **Poor mesh** (non-ortho ≥ 65°): → `Gauss linear limited corrected 0.33` / `limited corrected 0.33`

Pure `corrected` causes SIGFPE in `GAMGSolver::scale` on non-orthogonal meshes because
the explicit non-orthogonal correction creates ill-conditioned pressure matrices.
`limited corrected` blends corrected and uncorrected based on local mesh quality.

You can always generate the template above (pure `corrected`) — the validator will
downgrade it automatically if needed.

## 2D / 3D notes

- No changes to fvSchemes for 2D vs 3D — the same numerical schemes apply
- `wallDist { method meshWave; }` still needed for turbulent 2D (wall functions apply to wall patches, not empty/wedge patches)
- The viscous stress form `div((nuEff*dev2(T(grad(U)))))` is correct for both 2D and 3D
