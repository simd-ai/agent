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
