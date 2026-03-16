# compressibleMultiphaseInterFoam — system/setFieldsDict

## When this file is generated

`system/setFieldsDict` is generated when `initial_phase_layout` is:
- `liquid_region_in_gas` — liquid occupies a geometric sub-region; rest is vapour
- `gas_region_in_liquid` — a gas bubble inside a liquid-filled domain

It is **NOT generated** when `initial_phase_layout` is `uniform_gas` or `uniform_liquid`
(those are handled directly in `0/alpha.<liquidPhase>` internalField).

## Companion file

A `0/alpha.<liquidPhase>.orig` file is also generated — a uniform template that the
`setFields` utility reads and overwrites within the specified region.

The runner must execute before starting the solver:
```
cp 0/alpha.<liquidPhase>.orig 0/alpha.<liquidPhase>
setFields
```

---

## Template

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      setFieldsDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defaultFieldValues
(
    volScalarFieldValue alpha.<liquidPhase>  0
);

regions
(
    boxToCell
    {
        box  (<xmin> <ymin> <zmin>) (<xmax> <ymax> <zmax>);
        fieldValues
        (
            volScalarFieldValue alpha.<liquidPhase>  1
        );
    }
);

// ************************************************************************* //
```

---

## Geometry rules

### `liquid_region_in_gas` (most common cryogenic injection case)

`defaultFieldValues`: alpha = 0 (entire domain starts as vapour)

`regions`: one `boxToCell` or `cylinderToCell` block setting alpha = 1 in the liquid-occupied region.

- If the user specifies a fill level: `box (x0 y0 0) (x1 y1 <fill_height>)`.
- If no geometry is specified: use a thin region near the inlet (≈ 10 % of domain length).
- For axisymmetric / pipe inlet: use `cylinderToCell` with the inlet circle geometry.

### `gas_region_in_liquid` (bubble / cavity in filled domain)

`defaultFieldValues`: alpha = 1 (entire domain starts as liquid)

`regions`: one block setting alpha = 0 in the gas region (bubble centre + radius, or box).

---

## Alternative geometry selectors

```
// Cylinder (for pipe/axisymmetric):
cylinderToCell
{
    p1  (<x0> <y0> <z0>);
    p2  (<x1> <y1> <z1>);
    radius  <r>;
    fieldValues ( volScalarFieldValue alpha.<liquidPhase>  1 );
}

// Sphere (for bubble):
sphereToCell
{
    centre  (<cx> <cy> <cz>);
    radius  <r>;
    fieldValues ( volScalarFieldValue alpha.<liquidPhase>  0 );
}
```

---

## Phase name substitution

Replace `<liquidPhase>` with the actual liquid phase name from the case spec:
- LN2  → `liquidNitrogen`
- LH2  → `liquidHydrogen`
- LOX  → `liquidOxygen`
- LHe  → `liquidHelium`
- Water → `water`
