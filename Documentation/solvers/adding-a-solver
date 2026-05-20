adding a solver
===============

Drop a directory into `simd_agent/solvers/<name>/`. The registry
finds it. No registry edits, no `__init__.py` changes elsewhere.


the contract
------------

Your plugin class subclasses `SolverPlugin`
(`simd_agent/solvers/base.py`) and lives at
`simd_agent/solvers/<name>/solver.py`. The package `__init__.py`
exports a configured instance as `solver_plugin`.

A minimal plugin:

    # simd_agent/solvers/mySolverFoam/solver.py
    from simd_agent.solvers.base import (
        SolverPlugin, MatchResult, ValidationResult,
    )

    class MySolverFoamPlugin(SolverPlugin):

        # ── identity ────────────────────────────────────────
        name = "mySolverFoam"
        algorithm = "SIMPLE"          # "SIMPLE" | "PIMPLE" | "PISO"
        pressure_field = "p"          # "p" | "p_rgh"
        is_transient = False
        is_compressible = False
        supports_energy = False
        needs_gravity = False
        is_multiphase = False

        # ── solver selection ────────────────────────────────
        def matches(self, config):
            score = 0
            phys = config.get("physics") or {}
            if not phys.get("compressible"):
                score += 10
            if phys.get("time_scheme") == "steady":
                score += 5
            return MatchResult(score=score, reasons=[
                "incompressible steady single-region",
            ])

        # ── file manifest ───────────────────────────────────
        def required_files(self, config):
            return [
                "system/controlDict",
                "system/fvSchemes",
                "system/fvSolution",
                "constant/transportProperties",
                "constant/turbulenceProperties",
                "0/U",
                "0/p",
                # turbulence fields are auto-appended by
                # turbulence_fields() — no need to list them here
            ]

        # ── validation ──────────────────────────────────────
        def validate(self, files, config):
            issues = []
            # solver-specific fixes — base class helpers cover
            # the universal ones (constraint patch BCs,
            # controlDict solver name, brace balance, ...)
            return ValidationResult(files=files, issues=issues)

The `__init__.py`:

    # simd_agent/solvers/mySolverFoam/__init__.py
    from simd_agent.solvers.mySolverFoam.solver import MySolverFoamPlugin
    solver_plugin = MySolverFoamPlugin()
    __all__ = ["solver_plugin"]


per-file prompt docs
--------------------

The plugin reads prompts from `solvers/<name>/prompts/`:

    solvers/mySolverFoam/prompts/
      _solver.md                          # global identity + rules
      system/controlDict.md
      system/fvSchemes.md
      system/fvSolution.md
      constant/transportProperties.md
      constant/turbulenceProperties.md
      fields/U.md
      fields/p.md
      fields/k.md
      fields/omega.md
      fields/nut.md
      ...

The base class's `prompt_for_file(file_path)` maps a path to the
matching `.md`. The codegen loop feeds one prompt per file to the
LLM so the context stays focused.

`_solver.md` is the shared context loaded on every per-file call —
keep it short, just the solver identity and the must-remember
global rules.


registration is automatic
-------------------------

The registry uses `pkgutil.iter_modules` to discover every package
under `simd_agent/solvers/` that exports `solver_plugin`. There's
nothing to edit in `__init__.py`, `registry.py`, or any selector.

The classification queries (`p_solvers()`, `energy_solvers()`,
`gravity_solvers()`, …) read your class attributes. The selector's
LLM roster auto-includes you.


validators worth reusing
------------------------

`SolverPlugin` provides these helpers — call them from your
`validate()`:

  - `_fix_brace_balance(files, issues)` — repairs mismatched
    curly braces in any file.
  - `_fix_constraint_patch_bcs(files, config, issues)` — forces
    `symmetry`/`empty`/`wedge` BCs to match the mesh-side patch
    type.
  - `_fix_controldict_solver(files, issues)` — ensures
    `controlDict` declares your solver name.
  - `_fix_pressure_field(files, config, issues)` — swaps `p` and
    `p_rgh` where the chosen pressure variable was inconsistent.

For solver-specific checks (your equivalent of "rhoSimpleFoam needs
`div(phid,p)`"), write small regex passes over `files["system/..."]`
and emit `ValidationIssue(severity, file_path, message)` for each
finding.


example: copying simpleFoam as a starting point
-----------------------------------------------

    cp -R simd_agent/solvers/simpleFoam simd_agent/solvers/mySolverFoam

    # rename the class and the export
    sed -i '' 's/SimpleFoam/MySolverFoam/g; s/simpleFoam/mySolverFoam/g' \
      simd_agent/solvers/mySolverFoam/{solver.py,__init__.py}

    # tweak prompts/
    # tweak required_files() and matches()
    # add solver-specific validators if any

Restart the agent. Your solver shows up in:

    curl http://localhost:8000/api/solvers
    # ["icoFoam", "simpleFoam", "pimpleFoam", "mySolverFoam", ...]

And it's automatically a candidate for the LLM solver selector.


testing
-------

The test suite has a few helpful patterns:

  - `tests/test_solver_plugin_attributes.py` — checks every
    registered plugin against MRO landmines. Yours will be picked
    up automatically.

  - `tests/test_cht_phase3_pipeline.py` — pattern for "this case
    config should produce these files with these contents."
    Adapt for your solver.

  - `tests/test_<solver>.py` — write your own focused tests for
    auto-fixes and validation rules.

Run with `pytest -v tests/test_solver_plugin_attributes.py
tests/test_<your_solver>.py`.


multi-region solvers
--------------------

If your solver is multi-region, subclass `MultiRegionBase`
(`simd_agent/solvers/families/_multi_region.py`) instead of
`SolverPlugin` directly. You inherit:

  - the deterministic per-region renderer
  - the coupled-baffle BC builder
  - the `regionProperties` writer
  - the turbulence field set including `alphat`

Override only what's specific to your variant (SIMPLE vs PIMPLE
relaxation blocks, time scheme).
