# buoyantSimpleFoam — 0/T

```
dimensions      [0 0 0 1 0 0 0];   // Kelvin
internalField   uniform <inlet_temperature_K>;  // or ambient temperature for natural convection
```

## Boundary condition patterns

```
// Hot wall — drives natural convection upward
hotWall
{
    type    fixedValue;
    value   uniform <T_hot>;    // e.g. 350 K
}

// Cold wall — sink for natural convection
coldWall
{
    type    fixedValue;
    value   uniform <T_cold>;   // e.g. 293 K
}

// Adiabatic wall (insulated)
adiabatic_wall
{
    type    zeroGradient;
}

// Forced convection inlet
inlet
{
    type    fixedValue;
    value   uniform <T_inlet>;
}

// Outlet
outlet
{
    type    inletOutlet;
    inletValue  uniform <T_ambient>;
    value       uniform <T_ambient>;
}

// Symmetry
symmetry { type symmetryPlane; }

// Empty (2D)
frontAndBack { type empty; }
```

## Critical rules

1. `internalField` should equal the ambient/initial temperature of the fluid domain.
   For a heated room, this is the initial air temperature (e.g. 293 K), NOT a hot wall temp.
2. DO NOT set `internalField uniform 300` when the actual temperature is different — this
   causes density mismatch on iteration 0 for perfectGas: ρ = p/(R*T), wrong T → wrong ρ → divergence.
3. `0/T` is the ONLY temperature field. Never generate `0/h` or `0/e`.
4. For outlet BCs, `inletOutlet` prevents temperature from being advected back in.
