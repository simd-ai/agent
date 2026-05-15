# buoyantSimpleFoam — 0/k

Turbulent kinetic energy. Only generate when turbulence model uses k (kOmegaSST, kEpsilon).

```
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform <k_initial>;   // from turbulence_initial_values.k
```

## Boundary conditions

```
// Wall — compressible wall function
walls
{
    type    compressible::kqRWallFunction;
    value   uniform <k_initial>;
}

// Inlet — fixed value (from turbulence block: intensity + length scale)
inlet
{
    type    fixedValue;
    value   uniform <k_inlet>;
}

// Outlet — inletOutlet (prevents reverse flow from importing k)
outlet
{
    type        inletOutlet;
    inletValue  uniform <k_inlet>;
    value       uniform <k_inlet>;
}

// Symmetry
symmetry { type symmetryPlane; }

// Empty (2D)
frontAndBack { type empty; }
```

## Typical initial values for natural convection in air

- `k = 1.5 * (U_ref * I)^2` where I = turbulence intensity (0.05 for indoor, 0.1 for outdoor)
- For a room at ΔT = 20 K with no forced flow: k ≈ 0.001–0.01 m²/s²
- Use values from `turbulence_initial_values.k` when provided.
