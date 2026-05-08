# pimpleFoam — system/fvSchemes

**Transient incompressible**. Use `Euler` for `ddtSchemes`.

## ddtSchemes

- `Euler` — first-order implicit, unconditionally stable, recommended default
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

## Mesh-quality-dependent schemes (handled by validator)

The validator auto-hardens laplacian/snGrad schemes based on `checkMesh` metrics:
- **Good mesh** (non-ortho < 40 deg): `Gauss linear corrected` / `corrected` — full accuracy
- **Moderate mesh** (40 deg <= non-ortho < 65 deg): `Gauss linear limited corrected 0.5` / `limited corrected 0.5`
- **Poor mesh** (non-ortho >= 65 deg): `Gauss linear limited corrected 0.33` / `limited corrected 0.33`

Pure `corrected` causes SIGFPE in `GAMGSolver::scale` on non-orthogonal meshes.
`limited corrected` blends corrected and uncorrected based on local mesh quality.

You can always generate the template below (pure `corrected`) — the validator will
downgrade it automatically if needed.

**Note**: fvSchemes is generated deterministically by the validator — not by the LLM.
Any LLM-generated version is replaced.

## Template

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

wallDist { method meshWave; }
```

## 2D / 3D notes

- No changes to fvSchemes for 2D vs 3D — the same numerical schemes apply
- `wallDist { method meshWave; }` still needed for turbulent 2D
- The viscous stress form `div((nuEff*dev2(T(grad(U)))))` is correct for both 2D and 3D
