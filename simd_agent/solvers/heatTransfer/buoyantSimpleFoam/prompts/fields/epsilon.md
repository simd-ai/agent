# buoyantSimpleFoam — 0/epsilon

Turbulence dissipation rate. Only generate when turbulence model is kEpsilon.

```
dimensions      [0 2 -3 0 0 0 0];
internalField   uniform <epsilon_initial>;   // from turbulence_initial_values.epsilon
```

## Boundary conditions

```
// Wall — compressible wall function
walls
{
    type    compressible::epsilonWallFunction;
    value   uniform <epsilon_initial>;
}

// Inlet
inlet
{
    type    fixedValue;
    value   uniform <epsilon_inlet>;
}

// Outlet
outlet
{
    type        inletOutlet;
    inletValue  uniform <epsilon_inlet>;
    value       uniform <epsilon_inlet>;
}

// Symmetry
symmetry { type symmetryPlane; }

// Empty (2D)
frontAndBack { type empty; }
```

## Typical values

epsilon = Cmu^0.75 * k^1.5 / L_t where Cmu = 0.09.
Use `turbulence_initial_values.epsilon` when provided.
