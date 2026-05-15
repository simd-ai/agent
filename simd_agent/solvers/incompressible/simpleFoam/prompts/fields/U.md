# simpleFoam — 0/U

**Dimensions**: `[0 1 -1 0 0 0 0]` (m/s)
**internalField**: `uniform (<Ux> <Uy> <Uz>)` — initialize with inlet velocity for faster steady-state convergence. Do NOT use `(0 0 0)` — starting from rest wastes iterations.

## 2D velocity constraints

- **Planar 2D** (`is_2d: true`, `empty` patches): The out-of-plane velocity component MUST be 0.
  For XY-plane simulations: `internalField uniform (<Ux> <Uy> 0)` — Uz = 0 in ALL BCs.
- **Axisymmetric 2D** (`wedge` patches): Circumferential component = 0. Velocity has axial + radial components only.
- **3D**: All components may be non-zero.

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet (fixed velocity) | `fixedValue` | `value uniform (<Ux> <Uy> <Uz>)` |
| inlet (mass flow rate) | `flowRateInletVelocity` | See rules below |
| inlet (volumetric flow) | `flowRateInletVelocity` | See rules below |
| outlet | `inletOutlet` | `inletValue uniform (0 0 0); value uniform (0 0 0);` — prevents backflow divergence in recirculating flows. NEVER use `zeroGradient` at outlets. |
| wall | `noSlip` | |
| symmetry | `symmetry` | |
| symmetryPlane | `symmetryPlane` | |
| empty (2D planar) | `empty` | No `value` — just `type empty;` |
| wedge (2D axi) | `wedge` | No `value` — just `type wedge;` |

## flowRateInletVelocity — CRITICAL rules for incompressible solvers

### `rho` keyword — NEVER use a number; NEVER write `rho rho;` for simpleFoam

The `rho` keyword expects a **word** (field name), never a scalar number.
- `rho 880;` → **FOAM FATAL IO ERROR**: "Wrong token type — expected word, found double 880"
- `rho rho;` → **FOAM error**: simpleFoam has no `rho` field — lookup fails with no `rhoInlet` fallback

### Option A — volumetric flow rate (PREFERRED)

The CaseSpec pre-computes `volumetricFlowRate = massFlowRate / rho`:

```
inlet
{
    type                flowRateInletVelocity;
    volumetricFlowRate  constant <Q_m3_per_s>;    // = massFlowRate / rho (pre-computed)
    value               uniform (0 0 0);           // required placeholder — NOT the velocity
}
```

### Option B — mass flow rate with constant density (valid per OF docs)

```
inlet
{
    type            flowRateInletVelocity;
    massFlowRate    constant <mdot_kg_per_s>;
    rhoInlet        <density_kg_per_m3>;           // scalar fallback — no `rho` keyword!
    value           uniform (0 0 0);
}
```

- `constant` qualifier REQUIRED in OF 2406 for `Function1<scalar>` values.
- EXACTLY ONE of `volumetricFlowRate` or `massFlowRate` — never both.
- NEVER write `rho <word>;` — simpleFoam has no density field.
- `value uniform (0 0 0)` is a required placeholder — NEVER the flow rate value.
