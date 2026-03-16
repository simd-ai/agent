# pimpleFoam — system/fvSchemes

**Transient incompressible**. Use `Euler` (first-order) or `CrankNicolson 0.9` (second-order) for `ddtSchemes`.

## ddtSchemes choice

- `Euler` — first-order, more robust, recommended for startup or complex geometry
- `CrankNicolson 0.9` — second-order accurate, requires smaller time step, good for vortex/acoustic problems
- `backward` — second-order, conditionally stable

Default: use `Euler` unless the user specifically requests higher-order time integration.

## divSchemes

`default none` — list terms explicitly.
**NEVER include compressible terms** (`div(phid,p)`, `div(phi,K)`, `div(phi,Ekp)`, `div(((rho*nuEff)*...))`) — they crash pimpleFoam.
Viscous stress term is `div((nuEff*dev2(T(grad(U)))))` — no `rho*` prefix.

## Turbulence vs laminar — CRITICAL

**Laminar** (`simulationType laminar`): omit ALL turbulence div terms and `wallDist`.
**Turbulent** (RAS/LES): include only the turbulence fields that are actually active — `k`+`omega` for kOmegaSST, `k`+`epsilon` for kEpsilon, never both omega and epsilon.

## Template — Turbulent (RAS)

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    default         none;

    div(phi,U)      bounded Gauss linearUpwind grad(U);

    // include only turbulence terms matching active model:
    div(phi,k)      bounded Gauss limitedLinear 1;
    div(phi,omega)  bounded Gauss limitedLinear 1;
    // div(phi,epsilon) bounded Gauss limitedLinear 1;  // kEpsilon only

    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }

// include wallDist only for turbulent (wall functions need wall distance):
wallDist { method meshWave; }
```

## Template — Laminar

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    default         none;

    div(phi,U)      bounded Gauss linearUpwind grad(U);

    div((nuEff*dev2(T(grad(U))))) Gauss linear;

    // no turbulence div terms for laminar
}

laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }

// no wallDist for laminar
```
