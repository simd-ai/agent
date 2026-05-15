# buoyantSimpleFoam — 0/omega

Specific turbulence dissipation rate. Only generate when turbulence model is kOmegaSST.

```
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform <omega_initial>;   // from turbulence_initial_values.omega
```

## Boundary conditions

```
// Wall — compressible wall function
walls
{
    type    compressible::omegaWallFunction;
    value   uniform <omega_initial>;
}

// Inlet
inlet
{
    type    fixedValue;
    value   uniform <omega_inlet>;
}

// Outlet
outlet
{
    type        inletOutlet;
    inletValue  uniform <omega_inlet>;
    value       uniform <omega_inlet>;
}

// Symmetry
symmetry { type symmetryPlane; }

// Empty (2D)
frontAndBack { type empty; }
```

## Typical values

omega = sqrt(k) / (Cmu^0.25 * L_t) where L_t = turbulence length scale.
For a room: omega ≈ 1–10 s⁻¹ depending on scale.
Use `turbulence_initial_values.omega` when provided.
