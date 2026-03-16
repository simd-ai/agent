# buoyantSimpleFoam — 0/nut

Turbulent kinematic viscosity. Dimensions: `[0 2 -1 0 0 0 0]` (m²/s).

```
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    // Wall — compressible wall function (modern OF2406 ESI)
    walls
    {
        type    compressible::nutUWallFunction;
        value   uniform 0;
    }

    // Non-wall patches
    inlet   { type calculated; value uniform 0; }
    outlet  { type calculated; value uniform 0; }

    // Symmetry
    symmetry { type symmetryPlane; }

    // Empty (2D)
    frontAndBack { type empty; }
}
```

## Version note

Modern OpenFOAM 2406 ESI uses `nut` (kinematic, `[0 2 -1]`) with `compressible::nutUWallFunction`.
Older OF (< OF5) used `mut` (dynamic, `[1 -1 -1]`) with `mutUWallFunction`.
Always use `nut` with `compressible::nutUWallFunction` for OF2406.
