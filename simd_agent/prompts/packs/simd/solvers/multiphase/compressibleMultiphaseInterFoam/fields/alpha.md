# compressibleInterFoam — 0/alpha.<liquidPhase> (or .orig)

Generate the phase-fraction field for the liquid phase.

**Dimensions**: `[0 0 0 0 0 0 0]`

## Meaning
- `alpha.<liquidPhase> = 1` — cell is fully occupied by the liquid phase.
- `alpha.<liquidPhase> = 0` — cell is fully occupied by the vapour/gas phase.

---

## File name — uniform vs. setFields cases

The `initial_phase_layout` value in the case spec determines which file to generate:

| `initial_phase_layout`    | File to generate                    | Notes |
|---|---|---|
| `uniform_gas`             | `0/alpha.<liquidPhase>`             | internalField uniform 0 |
| `uniform_liquid`          | `0/alpha.<liquidPhase>`             | internalField uniform 1 |
| `liquid_region_in_gas`    | `0/alpha.<liquidPhase>.orig`        | template (setFields fills region) |
| `gas_region_in_liquid`    | `0/alpha.<liquidPhase>.orig`        | template (setFields clears region) |

For `.orig` files, the `system/setFieldsDict` file is also generated.

---

## internalField selection

```
uniform_gas          → internalField uniform 0;   // domain starts as vapour
uniform_liquid       → internalField uniform 1;   // domain starts as liquid
liquid_region_in_gas → internalField uniform 0;   // .orig template; setFields fills liquid zone
gas_region_in_liquid → internalField uniform 1;   // .orig template; setFields clears gas bubble
```

---

## Boundary conditions

```
inlet       → fixedValue uniform 1   // liquid injected from inlet
            → fixedValue uniform 0   // vapour injected from inlet
outlet      → zeroGradient
wall        → zeroGradient
frontAndBack (2D) → empty
symmetry    → symmetry
```

**Default for cryogenic injection cases** (LN2, LH2, LOX, LHe entering an empty vessel):
- internalField `uniform 0` (domain starts as vapour)
- inlet `fixedValue uniform 1` (liquid enters)

---

## Constraints
- Generate ONLY the liquid phase fraction field (file name `0/alpha.<liquidPhase>` or `.orig`).
- Do NOT generate `0/alpha.<vapourPhase>` — it is derived as `1 - alpha.<liquidPhase>`.
- The file name must use the exact phase name from `phase_names[0]` (the liquid phase).
