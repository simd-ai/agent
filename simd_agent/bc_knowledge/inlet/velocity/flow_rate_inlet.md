# Inlet velocity from flow rate (`flowRateInletVelocity`)

## When to use
Use this when the user specifies:
- mass flow rate (kg/s, g/s, t/h, lb/s …)
- volumetric flow rate (m³/s, L/min, L/s, CFM …)
- wording like "flow rate", "mfr", "vfr", "flux"

This is the correct inlet velocity BC whenever the physical requirement is a
prescribed flow rate.  Do **not** convert it to a `fixedValue` velocity.

---

## OpenFOAM v2406 dictionary structure

### Option A — mass flow rate [kg/s]

```foam
<patchName>
{
    type            flowRateInletVelocity;

    massFlowRate    <value>;    // MANDATORY — actual kg/s value (Function1<scalar>)

    rho             rho;        // name of the density field (default: rho)
    rhoInlet        <density>;  // [kg/m³] fallback density used at iteration 0
                                // when the rho field has not yet been computed.
                                // REQUIRED for compressible solvers (rhoSimpleFoam,
                                // rhoPimpleFoam, …) because p and T are not yet
                                // converged at startup.

    // extrapolateProfile  yes; // optional: match interior velocity profile
                                // omit for uniform plug flow (default, safer for codegen)

    value           uniform (0 0 0);  // placeholder for patch initialisation only
}
```

### Option B — volumetric flow rate [m³/s]

```foam
<patchName>
{
    type                flowRateInletVelocity;

    volumetricFlowRate  <value>;  // MANDATORY — actual m³/s value

    // rho / rhoInlet not required for volumetric flow

    value               uniform (0 0 0);
}
```

---

## Field entry reference (OpenFOAM v2406)

| Entry | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `type` | word | yes | — | Must be `flowRateInletVelocity` |
| `massFlowRate` | Function1<scalar> | choice | — | Option A; units: kg/s |
| `volumetricFlowRate` | Function1<scalar> | choice | — | Option B; units: m³/s |
| `rho` | word | no | `rho` | Name of density field (used for mass→volumetric conversion) |
| `rhoInlet` | scalar | no | -VGREAT | Fallback density [kg/m³] when field not in database |
| `extrapolateProfile` | bool | no | `false` | `yes` = match interior profile; `no` = uniform plug |
| `value` | vectorField | yes | — | Placeholder for initialisation — NOT the flow rate |

---

## CRITICAL rules

1. **Exactly one** of `massFlowRate` or `volumetricFlowRate` MUST be present.
   OpenFOAM fatals: `"Please supply either 'volumetricFlowRate' or 'massFlowRate'"`.

2. **`value uniform (0 0 0)` is NOT the flow rate.**  
   It is only a patch-initialisation placeholder.  Never put the flow rate number here.

3. **`rho` and `rhoInlet` are optional** — only include them if the user explicitly
   provided them in the configuration.  Do NOT invent or add them automatically.

4. **For volumetric flow rate**: never include `rho` or `rhoInlet`.

5. **`extrapolateProfile`** is optional — only include it if the user specified it.
   Default behaviour is uniform plug flow (no entry needed).

---

## Solver-specific guidance

| Solver | Mode | Mandatory | Optional (only if user specified) |
|--------|------|-----------|----------------------------------|
| any | `volumetricFlowRate` | `volumetricFlowRate` + `value` | — |
| any | `massFlowRate` | `massFlowRate` + `value` | `rho`, `rhoInlet`, `extrapolateProfile` |

---

## LLM retrieval rule
If the prompt contains a mass flow rate or volumetric flow rate:
1. Select `flowRateInletVelocity` as the BC family.
2. Keep the flow rate as the primary user-owned value — do NOT convert it to a fixed velocity.
3. Set `value uniform (0 0 0)` as the placeholder.
4. A mean velocity may be estimated for UI display but must NOT become the `value` entry.

## UI-only derived helper
You may compute `estimatedMeanVelocity = massFlowRate / (rho * area)` for display.
Do NOT send this as the main BC value or as `fixedValue U`.
