"""Boundary-condition + initial-condition fixers for the ``0/*`` directory.

Hoisted out of ``SolverPlugin`` for the same reasons as ``legacy_fixers``:

  * The logic is the same for every compressible solver (rho*, buoyant*) —
    nothing customises per subclass.  Putting it on the abstract class
    just hid that fact.
  * Free functions document their inputs: classify takes a config dict,
    rewrite takes text + patch name + new body.  No ``self.*`` magic.
  * Discoverability: searching for "inlet turbulence BC rewrite" goes
    to one place.

Three layers:

  * ``rewrite_patch_body``      — text utility (depth-balanced replace of
                                  one patch block inside a 0/* file).
  * ``classify_patches``        — derives ``(outlet_names, inlet_names)``
                                  from the precheck-shaped boundary_conditions.
  * ``fix_outlet_backflow_bcs`` / ``fix_inlet_turbulence_bc_types`` —
                                  the actual robustness rewrites that match
                                  the OpenFOAM reference rhoSimpleFoam /
                                  rhoPimpleFoam tutorials.

Calling convention: ``files`` and ``issues`` are mutated in place; the
``files`` dict is returned for ergonomic chaining (mirrors
``legacy_fixers``).  All other inputs are positional/required.
"""

from __future__ import annotations

import re
from typing import Any

from simd_agent.solvers.base import ValidationIssue


# ────────────────────────────────────────────────────────────────────────────
# Text utility — rewrite the body of one named patch block
# ────────────────────────────────────────────────────────────────────────────


def rewrite_patch_body(
    content: str, patch_name: str, new_body: str
) -> str | None:
    """Replace the body of ``patch_name { … }`` in an OpenFOAM dict file.

    Uses brace-depth tracking so nested sub-dicts inside the patch body
    (rare but valid — e.g. ``turbulentMixingLengthFrequencyInlet`` with
    a ``coordinateSystem { … }`` sub-block) are not miscounted.

    Returns the rewritten text, or ``None`` if the patch wasn't found.
    The caller treats ``None`` as a no-op.
    """
    m = re.search(
        rf"(^|\n)(\s*){re.escape(patch_name)}\s*\{{",
        content,
    )
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(content) and depth > 0:
        c = content[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    end = i - 1
    indent = m.group(2)
    body_lines = new_body.strip().splitlines()
    body_indented = "\n".join(f"{indent}    {ln}" for ln in body_lines)
    return (
        content[: m.start() + len(m.group(1))]
        + f"{indent}{patch_name}\n"
        + f"{indent}{{\n"
        + body_indented + "\n"
        + f"{indent}}}"
        + content[end + 1 :]
    )


# ────────────────────────────────────────────────────────────────────────────
# Patch classification
# ────────────────────────────────────────────────────────────────────────────


def classify_patches(
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return ``(outlet_names, inlet_names)`` from ``boundary_conditions``.

    Reads ``patch_class`` (precheck shape), ``patchClass`` (camelCase
    variant) and ``patch_type`` (legacy).  Anything outside
    {inlet, outlet, wall, …} is ignored — the fixers only act on inlets
    and outlets.
    """
    outlets: list[str] = []
    inlets: list[str] = []
    bcs = config.get("boundary_conditions") or {}
    for name, bc in bcs.items():
        if not isinstance(bc, dict):
            continue
        pc = (
            bc.get("patch_class")
            or bc.get("patchClass")
            or bc.get("patch_type")
            or ""
        ).lower()
        if pc == "outlet":
            outlets.append(name)
        elif pc == "inlet":
            inlets.append(name)
    return outlets, inlets


# ────────────────────────────────────────────────────────────────────────────
# Outlet backflow safety (zeroGradient → inletOutlet)
# ────────────────────────────────────────────────────────────────────────────


def fix_outlet_backflow_bcs(
    files: dict[str, str],
    issues: list[ValidationIssue],
    config: dict[str, Any],
) -> dict[str, str]:
    """Outlet U/T/k/ω/ε must use ``inletOutlet`` (not ``zeroGradient``).

    On mass-flow-driven cases with high pressure ratios, the outlet
    face flux transiently reverses during startup.  ``zeroGradient``
    lets the reversed flow pull garbage upstream; ``inletOutlet``
    snaps to ``inletValue = $internalField`` whenever the flux turns
    negative.  Matches the OF rhoSimpleFoam / rhoPimpleFoam reference
    tutorials.

    Leaves patches that already use ``inletOutlet`` (or anything more
    specific) untouched — never downgrades a hand-tuned BC.
    """
    outlets, _ = classify_patches(config)
    if not outlets:
        return files

    for field in ("U", "T", "k", "omega", "epsilon", "nut", "alphat"):
        fpath = f"0/{field}"
        content = files.get(fpath, "")
        if not content:
            continue
        new_content = content
        for patch_name in outlets:
            if patch_name not in new_content:
                continue
            m = re.search(
                rf"\n\s*{re.escape(patch_name)}\s*\{{([^}}]*)\}}",
                new_content,
            )
            if not m:
                continue
            body = m.group(1)
            if "zeroGradient" not in body or "inletOutlet" in body:
                continue
            new_body = (
                "type            inletOutlet;\n"
                "inletValue      $internalField;\n"
                "value           $internalField;"
            )
            rewritten = rewrite_patch_body(
                new_content, patch_name, new_body
            )
            if rewritten is not None:
                new_content = rewritten
                issues.append(
                    ValidationIssue(
                        "warning",
                        fpath,
                        f"Outlet '{patch_name}' on {field}: "
                        "zeroGradient → inletOutlet (prevents backflow "
                        "from pulling garbage into the domain).",
                    )
                )
        if new_content != content:
            files[fpath] = new_content
    return files


# ────────────────────────────────────────────────────────────────────────────
# Inlet turbulence BC type rewrites
# ────────────────────────────────────────────────────────────────────────────


def fix_inlet_turbulence_bc_types(
    files: dict[str, str],
    issues: list[ValidationIssue],
    config: dict[str, Any],
) -> dict[str, str]:
    """Inlet k/ω/ε: derive from inlet U at runtime, not fixedValue.

    OpenFOAM provides BC types that compute the turbulence quantity
    from the *actual* inlet velocity at every step:

      * ``turbulentIntensityKineticEnergyInlet``           — k
      * ``turbulentMixingLengthFrequencyInlet``            — ω
      * ``turbulentMixingLengthDissipationRateInlet``      — ε

    A precheck-precomputed ``fixedValue`` drifts away from the real U
    produced by the mass-flow inlet — gives tiny k, ν_t → 0, no
    stabilising diffusion, ω goes negative, the case crashes.  These
    runtime-derived BCs adapt automatically.

    Mixing length defaults to ``0.07 · D_h`` (clamped to [1 mm, 1 m]).
    Intensity defaults to 5 % (clamped to [0.1 %, 50 %]).
    """
    _, inlets = classify_patches(config)
    if not inlets:
        return files

    turb_cfg = config.get("turbulence") or {}
    if not isinstance(turb_cfg, dict):
        turb_cfg = {}

    d_h_raw = (
        turb_cfg.get("hydraulic_diameter")
        or turb_cfg.get("hydraulicDiameter")
    )
    try:
        d_h = float(d_h_raw) if d_h_raw is not None else None
    except (TypeError, ValueError):
        d_h = None
    if d_h and d_h > 0:
        mixing_length = max(1e-3, min(1.0, 0.07 * d_h))
    else:
        mixing_length = 0.01

    ti_raw = (
        turb_cfg.get("turbulence_intensity")
        or turb_cfg.get("turbulenceIntensity")
    )
    try:
        ti_pct = float(ti_raw) if ti_raw is not None else 5.0
    except (TypeError, ValueError):
        ti_pct = 5.0
    intensity_frac = max(0.001, min(0.5, ti_pct / 100.0))

    field_to_new_type = {
        "k": (
            "turbulentIntensityKineticEnergyInlet",
            f"intensity       {intensity_frac:.4f};",
        ),
        "omega": (
            "turbulentMixingLengthFrequencyInlet",
            f"mixingLength    {mixing_length:.4f};",
        ),
        "epsilon": (
            "turbulentMixingLengthDissipationRateInlet",
            f"mixingLength    {mixing_length:.4f};",
        ),
    }

    for field, (new_type, param_line) in field_to_new_type.items():
        fpath = f"0/{field}"
        content = files.get(fpath, "")
        if not content:
            continue
        new_content = content
        for patch_name in inlets:
            if patch_name not in new_content:
                continue
            m = re.search(
                rf"\n\s*{re.escape(patch_name)}\s*\{{([^}}]*)\}}",
                new_content,
            )
            if not m:
                continue
            body = m.group(1)
            if "fixedValue" not in body or new_type in body:
                continue
            value_match = re.search(
                r"value\s+(uniform\s+[^;]+;)", body
            )
            value_line = (
                f"value           {value_match.group(1)}"
                if value_match
                else "value           uniform 0;"
            )
            new_body = (
                f"type            {new_type};\n"
                f"{param_line}\n"
                f"{value_line}"
            )
            rewritten = rewrite_patch_body(
                new_content, patch_name, new_body
            )
            if rewritten is not None:
                new_content = rewritten
                issues.append(
                    ValidationIssue(
                        "warning",
                        fpath,
                        f"Inlet '{patch_name}' on {field}: "
                        f"fixedValue → {new_type} "
                        "(derives the value from actual inlet U).",
                    )
                )
        if new_content != content:
            files[fpath] = new_content
    return files


# ────────────────────────────────────────────────────────────────────────────
# Initial-condition seeding — kills the iteration-1 impulsive shock
# ────────────────────────────────────────────────────────────────────────────


def fix_initial_velocity_field(
    files: dict[str, str],
    issues: list[ValidationIssue],
    *,
    has_impulsive_inlets: bool,
    bulk_velocity: float,
) -> dict[str, str]:
    """Seed ``0/U.internalField`` with the estimated bulk velocity.

    When a case has ``flowRateInletVelocity`` inlets but the LLM emits
    ``0/U.internalField uniform (0 0 0)``, the first time step has to
    accelerate the fluid from 0 → U_inlet within one Δt — producing a
    giant pressure spike that the adaptive time-stepper can't escape.
    Δt collapses to floating-point underflow.

    Seeding the internal field with a non-zero estimate of the bulk
    flow speed removes that shock.  The exact direction doesn't matter
    much (pressure correction redistributes within ~5 iterations) —
    we pick ``(0.5·U_bulk, 0, 0)`` as a stable conservative default.

    Only fires when:
      * ``has_impulsive_inlets`` is True (the fix is irrelevant for
        velocity-fixed-value inlets), AND
      * the current internalField is ``(0 0 0)`` (|U| < 1e-3 m/s) —
        never overwrite a hand-tuned initial field.
    """
    if not has_impulsive_inlets or bulk_velocity <= 0:
        return files

    u_path = "0/U"
    content = files.get(u_path, "")
    if not content:
        return files

    m = re.search(
        r"(internalField\s+uniform\s+)\(\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\)",
        content,
    )
    if not m:
        return files

    try:
        ux, uy, uz = float(m.group(2)), float(m.group(3)), float(m.group(4))
    except ValueError:
        return files

    if (ux * ux + uy * uy + uz * uz) ** 0.5 >= 1e-3:
        return files

    u_seed = 0.5 * bulk_velocity
    new_internal = f"{m.group(1)}({u_seed:.4g} 0 0)"
    new_content = content[: m.start()] + new_internal + content[m.end():]
    files[u_path] = new_content

    issues.append(
        ValidationIssue(
            "warning",
            "0/U",
            f"Seeded internalField (0 0 0) -> ({u_seed:.4g} 0 0) to "
            f"prevent the iteration-1 impulsive shock from a "
            f"flowRateInletVelocity inlet (U_bulk approx {bulk_velocity:.1f} m/s).",
        )
    )
    return files
