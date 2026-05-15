# chtMultiRegionSimpleFoam — global rules

Steady multi-region conjugate heat transfer.

## Identity

- **Algorithm:** SIMPLE outer loop, per-region inner solves.
- **Pressure field (fluid regions):** `p_rgh`.
- **Energy variable (fluid):** `h`; **(solid):** `T` (conduction only).
- **Density:** compressible in fluid regions (`heRhoThermo`); constant in
  solid regions (`heSolidThermo`).

## Multi-region file tree

```
constant/
  regionProperties                          ← deterministic
  <fluid>/
    thermophysicalProperties                ← deterministic (heRhoThermo)
    turbulenceProperties                    ← deterministic (RAS)
    g                                       ← deterministic
  <solid>/
    thermophysicalProperties                ← deterministic (heSolidThermo)
system/
  controlDict                               ← LLM
  fvSchemes, fvSolution                     ← top-level (placeholder)
  <fluid>/
    fvSchemes, fvSolution                   ← deterministic
    changeDictionaryDict                    ← deterministic
  <solid>/
    fvSchemes, fvSolution                   ← deterministic
    changeDictionaryDict                    ← deterministic
0/
  <fluid>/{T, U, p, p_rgh, k, epsilon}      ← deterministic
  <solid>/T                                 ← deterministic
```

## Coupled boundaries

Every fluid-solid interface gets a `compressible::turbulentTemperatureCoupledBaffleMixed`
patch on **both sides**:

  * fluid side: `kappaMethod fluidThermo`
  * solid side: `kappaMethod solidThermo`

Patch names follow the OF convention `<self>_to_<other>` (e.g.
`topAir_to_heater`, `heater_to_topAir`).  The `interfaces` list on each
`RegionSpec` drives the patch generation — populate it from the mesh
boundary file in the precheck pipeline.

## Region presets — pick a fluid / solid by name

Each region can declare a preset that fills the physics defaults
(Cp / μ / Pr / mol_weight / ρ_solid / κ_solid / Cp_solid):

```yaml
regions:
  fluid:
    - name: hotWater
      fluid_preset: water        # → Cp=4182, μ=1.002e-3, Pr=7.0
      interfaces: [tube]
    - name: cryoLine
      fluid_preset: ln2          # → Cp=2042, μ=1.58e-4, thermo_profile=cryogenic
  solid:
    - name: tube
      solid_preset: copper       # → ρ=8960, κ=400, Cp=385
    - name: insulation
      solid_preset: concrete     # → ρ=2300, κ=1.4, Cp=880
```

**Available fluid presets:** `air`, `water`, `oil`, `ln2`, `lox`, `lh2`,
`lng`, `helium`.  **Available solid presets:** `steel`, `copper`,
`aluminum`, `concrete`, `glass`, `stainless`.

Per-field overrides win — `{fluid_preset: "water", Cp: 4200}` keeps the
water μ / Pr / mol_weight defaults and uses Cp=4200.  Unknown preset
names silently fall back to air-like / steel-like defaults.

## Status

- ✅ **Phase 1:** RegionSpec / CaseRegions contract, regionProperties,
  per-region thermophysicalProperties.
- ✅ **Phase 2:** Per-region fvSchemes / fvSolution, per-fluid
  turbulenceProperties + g, per-region 0-fields with coupled T BCs,
  changeDictionaryDict.
- ✅ **Phase 2.5:** Fluid + solid region presets — `fluid_preset: water`
  etc. drives the physics values; no more hardcoded air/steel defaults.
- ⏳ **Phase 3 (orchestrator integration):**
  - Tree-structured manifest emission in `run/orchestration.py`.
  - Multi-region case ZIP packaging in `run/packaging.py`.
  - Allrun-style scripts (`changeDictionary`, `splitMeshRegions`,
    `setFields`) — currently the user runs them by hand.

## LLM responsibility

Only `system/controlDict` is LLM-generated.  Everything else (region
properties, per-region thermo, fvSchemes, fvSolution, 0-fields,
changeDictionaryDict) is rendered deterministically from the
`RegionSpec` strategy.  **Do not generate** any file under
`constant/<region>/` or `system/<region>/` or `0/<region>/`.

## Reference tutorial

`OpenFOAM-4.x/tutorials/heatTransfer/chtMultiRegionSimpleFoam/multiRegionHeater`.
