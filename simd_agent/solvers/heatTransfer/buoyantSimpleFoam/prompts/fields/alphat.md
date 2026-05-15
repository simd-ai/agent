# buoyantSimpleFoam — 0/alphat

Turbulent thermal diffusivity. Required for ALL turbulent compressible cases.
Dimensions: `[1 -1 -1 0 0 0 0]` (kg/m·s).

```
dimensions      [1 -1 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    // ALL wall patches — compressible wall function required
    hotWall
    {
        type    compressible::alphatWallFunction;
        Prt     0.85;
        value   uniform 0;
    }
    coldWall
    {
        type    compressible::alphatWallFunction;
        Prt     0.85;
        value   uniform 0;
    }

    // Inlet/outlet — calculated (alphat is interior quantity)
    inlet   { type calculated; value uniform 0; }
    outlet  { type calculated; value uniform 0; }

    // Symmetry
    symmetry { type symmetryPlane; }

    // Empty (2D)
    frontAndBack { type empty; }
}
```

## Critical

- Use `compressible::alphatWallFunction` — NOT bare `alphatWallFunction`.
  Bare form is rejected: "Unknown patchField type alphatWallFunction for patch type wall".
- `Prt 0.85` is the turbulent Prandtl number (standard value for air).
- Generate this file whenever turbulence is active (kOmegaSST, kEpsilon, etc.).
