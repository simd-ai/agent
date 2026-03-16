# buoyantSimpleFoam — 0/p_rgh

**Primary solved pressure variable** — modified pressure accounting for hydrostatic head.
Relationship: `p = p_rgh + rho*(g·x)`

## Dimensions and internalField

```
dimensions      [1 -1 -2 0 0 0 0];   // Pa
internalField   uniform <operating_pressure>;  // e.g. 100000 (1 atm)
```

**`operating_pressure`** = absolute operating pressure in Pa (NOT 0 — this is absolute).

## Boundary condition patterns

```
// Pressure outlet (Dirichlet — fixes absolute p_rgh at outlet)
outlet
{
    type    fixedValue;
    value   uniform 100000;   // operating pressure [Pa]
}

// Walls — NO flow through; fixedFluxPressure maintains flux consistency with buoyancy
// DO NOT use zeroGradient on walls — it ignores the rho*g*h buoyancy correction
hotWall
{
    type    fixedFluxPressure;
    value   $internalField;
}
coldWall
{
    type    fixedFluxPressure;
    value   $internalField;
}

// Velocity inlet — fixedFluxPressure (velocity is specified; let solver adjust pressure flux)
inlet
{
    type    fixedFluxPressure;
    value   $internalField;
}

// Symmetry plane
symmetry
{
    type    symmetryPlane;
}

// Empty (2D frontAndBack)
frontAndBack
{
    type    empty;
}
```

## Critical rules

1. `fixedFluxPressure` on walls — NOT `zeroGradient`. The buoyancy source modifies
   the pressure gradient at walls; `zeroGradient` ignores this and produces inconsistent flux.
2. `fixedValue` at pressure outlets — specifies the reference p_rgh level.
3. `internalField` MUST be the operating pressure (e.g. 100000 Pa), not 0.
4. When no patch has `fixedValue`, add `pRefCell 0; pRefValue 0;` in SIMPLE block
   (underdetermined pressure system for closed domains).
