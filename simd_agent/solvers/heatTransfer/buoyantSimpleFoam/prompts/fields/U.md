# buoyantSimpleFoam — 0/U

```
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
```

## Boundary condition patterns

```
// No-slip wall (natural convection — all walls are no-slip)
hotWall  { type fixedValue; value uniform (0 0 0); }
coldWall { type fixedValue; value uniform (0 0 0); }

// Forced convection inlet (when flow is driven externally)
inlet
{
    type        fixedValue;
    value       uniform (<Ux> <Uy> <Uz>);
}

// Pressure outlet with possible backflow
outlet
{
    type        inletOutlet;
    inletValue  uniform (0 0 0);
    value       uniform (0 0 0);
}

// OR: simple outflow outlet
outlet { type zeroGradient; }

// Symmetry plane
symmetry { type symmetryPlane; }

// Empty (2D)
frontAndBack { type empty; }
```

## Notes

- For PURE natural convection (no forced inlet), all walls are `noSlip`/`fixedValue (0 0 0)`
  and any open boundaries (vents) use `pressureInletOutletVelocity`.
- For forced convection + buoyancy, use `fixedValue` at inlet and `inletOutlet` at outlet.
- Natural convection does NOT have a velocity inlet — flow is driven entirely by density
  differences due to temperature gradients. All boundaries are walls or pressure open.
