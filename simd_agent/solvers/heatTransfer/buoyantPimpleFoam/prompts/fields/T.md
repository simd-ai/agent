# buoyantPimpleFoam — 0/T

```
dimensions      [0 0 0 1 0 0 0];   // Kelvin
internalField   uniform <ambient_temperature_K>;  // initial domain temperature
```

## Boundary condition patterns

```
// Hot wall — heat source (drives transient convection)
hotWall
{
    type    fixedValue;
    value   uniform <T_hot>;    // e.g. 400 K
}

// Cold wall — heat sink
coldWall
{
    type    fixedValue;
    value   uniform <T_cold>;   // e.g. 293 K
}

// Adiabatic wall
adiabatic_wall { type zeroGradient; }

// Inlet (if forced convection present)
inlet
{
    type    fixedValue;
    value   uniform <T_inlet>;
}

// Outlet — inletOutlet prevents reverse flow from injecting wrong temperature
outlet
{
    type        inletOutlet;
    inletValue  uniform <T_ambient>;
    value       uniform <T_ambient>;
}

// Open vent (natural convection)
vent
{
    type        inletOutlet;
    inletValue  uniform <T_ambient>;
    value       uniform <T_ambient>;
}

// Symmetry
symmetry { type symmetryPlane; }

// Empty (2D)
frontAndBack { type empty; }
```

## Critical rules

1. `internalField` = initial temperature of the fluid domain (ambient condition).
   For a heated room starting from rest, use the initial air temperature (e.g. 293 K).
2. Use `inletOutlet` at outlets/vents — prevents unphysical temperature advection
   back into the domain during reverse flow in transient buoyancy-driven flows.
3. Never generate `0/h` or `0/e` — thermo reads T and computes h/e internally.
