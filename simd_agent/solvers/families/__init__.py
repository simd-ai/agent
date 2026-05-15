"""Solver family bases — paradigm-specific abstractions.

Each OpenFOAM solver inherits from a family base (Steady or Transient) and
optionally one or more mixins (Compressible, Boussinesq).  The family
encodes the *paradigm* shared by all solvers of that kind — the SIMPLE or
PIMPLE algorithm block, the per-paradigm relaxation pattern, the
algorithm-driven `Final`-variant coverage — so an edit to "all transient
solvers need Finals" lands in one place and is physically isolated from
the steady solvers that don't need it.

Inheritance map:

    SolverPlugin (ABC)            ← abstract base in solvers/base.py
    │
    ├── SteadyBase                ← families/_steady.py
    │   ├── SimpleFoamSolver
    │   ├── RhoSimpleFoamSolver   (also CompressibleMixin)
    │   └── BuoyantSimpleFoamSolver (also BoussinesqMixin)
    │
    └── TransientBase             ← families/_transient.py
        ├── PimpleFoamSolver
        ├── RhoPimpleFoamSolver   (also CompressibleMixin)
        └── BuoyantPimpleFoamSolver (also BoussinesqMixin)

Mixins (no SolverPlugin inheritance, composed via MRO):

    CompressibleMixin             ← families/_compressible.py
        rho solver block (with rhoFinal for PIMPLE), compressible bounds,
        energy_var-driven divScheme choices.

    BoussinesqMixin               ← families/_boussinesq.py
        pressure_field defaults to p_rgh, _ensure_gravity, alphat patches.

Why this layout (vs a single 2700-LOC base.py):

  * A change to "PIMPLE needs Finals for every solved field" belongs in
    TransientBase.  It physically cannot affect rhoSimpleFoam because
    rhoSimpleFoam inherits SteadyBase.
  * Solver-specific overrides remain in the plugin's own ``solver.py``;
    families are the *default*, not the law.
"""

from simd_agent.solvers.families._steady import SteadyBase
from simd_agent.solvers.families._transient import TransientBase
from simd_agent.solvers.families._compressible import CompressibleMixin
from simd_agent.solvers.families._boussinesq import (
    BoussinesqMixin,
    IncompressibleBoussinesqMixin,
)
from simd_agent.solvers.families._multi_region import MultiRegionBase

__all__ = [
    "SteadyBase",
    "TransientBase",
    "CompressibleMixin",
    "BoussinesqMixin",
    "IncompressibleBoussinesqMixin",
    "MultiRegionBase",
]
