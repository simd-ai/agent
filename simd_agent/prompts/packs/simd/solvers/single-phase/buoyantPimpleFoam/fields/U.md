# buoyantPimpleFoam — 0/U

```
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
```

## Boundary condition patterns

```
// No-slip wall
hotWall  { type noSlip; }
coldWall { type noSlip; }

// Forced inlet (if present)
inlet
{
    type    fixedValue;
    value   uniform (<Ux> <Uy> <Uz>);
}

// Pressure outlet — inletOutlet to handle reverse flow at transient boundaries
outlet
{
    type        inletOutlet;
    inletValue  uniform (0 0 0);
    value       uniform (0 0 0);
}

// Open boundaries (vents, atmosphere) for natural convection
vent
{
    type        pressureInletOutletVelocity;
    value       uniform (0 0 0);
}

// Symmetry
symmetry { type symmetryPlane; }

// Empty (2D)
frontAndBack { type empty; }
```

## Notes

- For transient natural convection from rest, `internalField uniform (0 0 0)` is correct.
- `pressureInletOutletVelocity` is appropriate when a boundary can act as inlet or outlet
  depending on the transient buoyancy-driven flow direction.
- `noSlip` is equivalent to `fixedValue (0 0 0)` — preferred for clarity in transient cases.
