# simd_agent/run/value_filler/prompts.py
"""Prompt builder for the value filler.

The prompt is composed from small section helpers — heading, the
authoritative per-patch table, the canonical ``case_defaults`` block,
the user prompt, this-file targets, rules, and finally the template.
Each helper takes the relevant slice of the per-file context, so
contributors swapping out a section (e.g. tightening rules, expanding
the fact table) only touch one function.

Both ``mode == "single"`` and ``mode == "multi"`` go through the same
top-level :func:`build_prompt`; the section helpers branch internally
where the wording legitimately differs (heading, this-file targets).
The rules are identical in both modes — that's the entire point of
the generalisation.
"""

from __future__ import annotations

from typing import Any


# Field-specific guidance — keeps the LLM grounded on OpenFOAM semantics.
_FIELD_HINT: dict[str, str] = {
    "T":     "temperature in Kelvin (e.g. 77 for LN2, 290 for room-temp water, 300 for room-temp air)",
    "U":     "velocity in m/s as a 3-vector ``(Ux Uy Uz)`` (positive Ux for +x, negative for counter-flow -x)",
    "p":     "pressure in Pa (101325 = 1 atm; 100000 = 1 bar; use atmospheric outlet unless the prompt says otherwise)",
    "p_rgh": "modified pressure p_rgh in Pa — same numerical seed as p",
}


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


def build_prompt(
    path: str, content: str, ctx: dict[str, Any], user_requirements: str,
) -> str:
    """Compose the full prompt from section helpers.

    Section order is fixed — earlier sections set authority that later
    rules refer back to ("the per-patch table is law", "case_defaults
    is the secondary source", "the prompt is tertiary").
    """
    return "\n\n".join([
        _heading_section(ctx),
        _authoritative_table_section(ctx),
        _case_defaults_section(ctx),
        _user_prompt_section(user_requirements),
        _file_target_section(path, ctx),
        _rules_section(ctx),
        _template_section(content),
    ])


# ────────────────────────────────────────────────────────────────────────────
# Section helpers
# ────────────────────────────────────────────────────────────────────────────


def _heading_section(ctx: dict[str, Any]) -> str:
    """Opening paragraph — describes what the LLM is being asked to do."""
    region_phrase = (
        "one OpenFOAM ``0/<region>/<field>`` field file for a multi-region (CHT) case"
        if ctx["mode"] == "multi"
        else "one OpenFOAM ``0/<field>`` field file for a single-region case"
    )
    return (
        f"You are generating the final content of {region_phrase}.  You "
        "receive a TEMPLATE with placeholder numerics (T = 300, U = (0 0 0), "
        "p = 100000) and you must PRODUCE the FINAL file with the real "
        "values substituted in."
    )


def _authoritative_table_section(ctx: dict[str, Any]) -> str:
    """The per-patch fact table — the PRIMARY source of truth."""
    field = ctx["field"]
    field_key = "p_rgh" if field == "p_rgh" else field
    hint = _FIELD_HINT.get(field, "scalar field value")

    lines = [
        "==========================================================",
        "AUTHORITATIVE PER-PATCH VALUES — THIS IS THE PRIMARY SOURCE.",
        "These come from the precheck / UI config and are the ground truth.",
        "ALWAYS use the value here when one is present.  Only fall back to",
        "case_defaults / the natural-language prompt / generic defaults when",
        'the value is "no explicit … recorded".',
        "",
        f"target field: {field}  ({hint})",
        "",
    ]
    for p in ctx["patches"]:
        val = p.get(f"{field_key}_value")
        typ = p.get(f"{field_key}_type")
        if val is None and typ is None:
            lines.append(
                f"  - {p['name']} (role: {p['role']})  →  no explicit "
                f"{field_key} value recorded — use role default"
            )
        else:
            joined = ", ".join(
                x for x in (
                    f"type={typ}" if typ else "",
                    f"value={val}" if val is not None else "",
                ) if x
            )
            lines.append(f"  - {p['name']} (role: {p['role']})  →  {joined}")
    if not ctx["patches"]:
        lines.append("  (no patches)")
    return "\n".join(lines)


def _case_defaults_section(ctx: dict[str, Any]) -> str:
    """The case-level state — SECONDARY source.

    For single-region: ``case_defaults`` is the per-case inlet state
    (one inlet, one set of values).
    For multi-region: the per-region ``T_init`` / ``U_init`` / ``p_init``
    block is the per-region state — case_defaults is a weak global
    fallback only.  The two are rendered as separate sub-sections so
    the rules can refer to them distinctly.
    """
    is_multi = ctx["mode"] == "multi"
    lines: list[str] = []

    if is_multi:
        lines.extend([
            "==========================================================",
            "REGION-LEVEL STATE — SECONDARY SOURCE (this region only).",
            "Resolved by RegionExtractor + region_inits from THIS region's",
            "inlet BC.  Use these for this region's internalField and any",
            "inlet patch in this region whose per-patch row had no value.",
            "",
            f"  region:     {ctx.get('name')}",
            f"  flow_kind:  {ctx.get('flow_kind') or '(unspecified)'}",
            f"  T_init:     {ctx.get('T_init')}",
            f"  U_init:     {ctx.get('U_init')}",
            f"  p_init:     {ctx.get('p_init')}",
            "",
        ])
        if ctx.get("flow_kind") == "closed":
            lines.extend([
                "NOTE: this is a CLOSED region — no inlet patch exists.",
                "internalField is seeded from T_init / U_init / p_init.",
                "U_init MUST be (0 0 0) for a sealed cavity unless the",
                "user explicitly says otherwise.",
                "",
            ])

    cd = ctx.get("case_defaults") or {}
    label = (
        "GLOBAL CASE_DEFAULTS — LAST-RESORT FALLBACK (multi-region: weak,"
        "\nuse REGION-LEVEL STATE above first)."
        if is_multi
        else "CANONICAL CASE-LEVEL VALUES (case_defaults — SECONDARY source).\n"
             "Resolved by the enrichment pipeline from the user's wizard inputs\n"
             "and per-patch BCs.  Use these as inlet/internal defaults when the\n"
             "per-patch table above has gaps."
    )
    lines.append("==========================================================")
    lines.append(label)
    lines.append("")
    if not cd:
        lines.append("  (no case_defaults resolved)")
    else:
        for key in (
            "inlet_velocity", "inlet_temperature", "inlet_pressure",
            "ambient_pressure", "bulk_temperature",
            "bulk_density", "bulk_kinematic_viscosity",
            "bulk_dynamic_viscosity", "bulk_prandtl",
            "turbulence_intensity", "wall_temperatures",
        ):
            v = cd.get(key)
            if v is None or v == {} or v == []:
                continue
            lines.append(f"  {key}: {v}")
    return "\n".join(lines)


def _user_prompt_section(user_requirements: str) -> str:
    """The natural-language prompt — TERTIARY source."""
    return (
        "==========================================================\n"
        "USER'S NATURAL-LANGUAGE PROMPT (tertiary — only consult when both\n"
        "the per-patch table and case_defaults have gaps):\n"
        f"{user_requirements.strip() or '(empty)'}"
    )


def _file_target_section(path: str, ctx: dict[str, Any]) -> str:
    """What this specific file is for."""
    lines = [
        "==========================================================",
        "THIS FILE TARGETS",
        f"  path:     {path}",
    ]
    if ctx["mode"] == "multi":
        lines.append(f"  region:   {ctx['name']}  ({ctx['kind']})")
        preset = ctx.get("fluid_preset") or ctx.get("solid_preset") or "(unspecified)"
        lines.append(f"  preset:   {preset}")
        ifaces = ", ".join(ctx.get("interfaces") or []) or "(none)"
        lines.append(f"  interfaces (coupled CHT walls): {ifaces}")
    else:
        if ctx.get("fluid_name"):
            lines.append(f"  fluid:    {ctx['fluid_name']}")
    return "\n".join(lines)


def _rules_section(ctx: dict[str, Any]) -> str:
    """The rules — single-region and multi-region share rule 1, 4, 5, 7.

    The authority chain (rules 2-3-6) differs because multi-region has
    per-region state (``T_init`` / ``U_init`` / ``p_init``) that is
    the actual secondary source for this region; case_defaults is then
    a weak global fallback.  Single-region keeps the historical chain.
    """
    field = ctx["field"]
    is_multi = ctx["mode"] == "multi"
    coupled_clause = (
        ", coupled-baffle block ({Tnbr, kappaMethod})" if is_multi else ""
    )

    # Authority chain — multi-region has an extra rule for per-region
    # state, so rule counts differ (single = 6, multi = 7).
    if is_multi:
        body = f"""1. PER-PATCH VALUE FROM THE TABLE IS LAW.
   When the per-patch row above gives ``value=X`` for the target field
   ({field}), every ``uniform`` numeric on that patch in the template
   MUST become exactly X.

2. REGION-LEVEL STATE IS THE SECONDARY SOURCE for THIS region.
   When the per-patch table has no explicit value for an inlet patch
   in this region, use this region's ``T_init`` / ``U_init`` /
   ``p_init`` (shown in the REGION-LEVEL STATE block above).  These
   represent the region's own initial / inlet state and ALWAYS win
   over the global case_defaults.

3. GLOBAL CASE_DEFAULTS is the TERTIARY fallback.
   Use case_defaults entries only when both the per-patch table and
   the region-level state are silent.  case_defaults carries case-
   wide signals (ambient pressure, bulk fluid properties) that all
   regions can share — it is NEVER the right source for region-
   specific inlet T/U/p when T_init / U_init / p_init is set on
   this region.

4. ROLE DEFAULTS (used only when all sources above are silent):
     • Inlet T: ambient 300 K for air, otherwise the matching fluid
       boiling point (LN2=77, LH2=20, LOX=90, LHe=4.2, water=290).
     • Inlet U: zero unless the prompt mentions a magnitude.
     • Outlet p / p_rgh: 101325 Pa (atmospheric).
     • Solid T: take from the coupling-side fluid's T_init or 300 K.

5. CONSTRAINT PATCHES are left alone.
   ``symmetry``, ``empty``, ``noSlip``, ``zeroGradient``,
   ``calculated``, ``fixedFluxPressure`` blocks: keep the ``type``
   line and the ``value uniform X;`` numeric where the template
   carries one — substitute via the same rules above.

6. internalField.
   For ``T`` / ``U`` / ``p`` / ``p_rgh``: use this region's
   ``T_init`` / ``U_init`` / ``p_init`` if set.  Otherwise the
   inlet patch's value, then case_defaults, then the role default.
   For closed regions (flow_kind=closed) the internalField IS the
   cavity seed — T_init drives T, U_init must be (0 0 0) unless the
   prompt says otherwise, p_init drives p / p_rgh.

7. STRUCTURE IS SACRED.
   Do NOT modify, rename, add, or remove any ``type ...;`` line{coupled_clause},
   patch name, FoamFile header field, ``dimensions`` line, comment,
   or whitespace pattern.  Output must contain the exact same set
   of patch entries as the template, in the same order.

8. OUTPUT FORMAT.
   Emit the COMPLETE file starting with ``FoamFile``.  No markdown
   fences, no commentary, no leading or trailing prose.  The first
   non-blank line of your answer MUST be ``FoamFile``."""
    else:
        body = f"""1. PER-PATCH VALUE FROM THE TABLE IS LAW.
   When the per-patch row above gives ``value=X`` for the target field
   ({field}), every ``uniform`` numeric on that patch in the template
   MUST become exactly X.

2. CASE_DEFAULTS IS THE SECONDARY SOURCE.
   When the per-patch table has no explicit value for an inlet, use
   the matching case_defaults entry (``inlet_velocity`` /
   ``inlet_temperature`` / ``inlet_pressure``).  Only fall through
   to the user prompt when both the table and case_defaults are silent.

3. ROLE DEFAULTS (used only when all sources above are silent):
     • Inlet T: ambient 300 K for air, otherwise the matching fluid
       boiling point (LN2=77, LH2=20, LOX=90, LHe=4.2, water=290).
     • Inlet U: zero unless the prompt mentions a magnitude.
     • Outlet p / p_rgh: 101325 Pa (atmospheric).

4. CONSTRAINT PATCHES are left alone.
   ``symmetry``, ``empty``, ``noSlip``, ``zeroGradient``,
   ``calculated``, ``fixedFluxPressure`` blocks: keep the ``type``
   line and the ``value uniform X;`` numeric where the template
   carries one — substitute via the same rules above.

5. internalField.
   For ``T`` / ``U`` fields: use the inlet patch's value (from rule
   1 then 2 then 3), otherwise the inlet role default from rule 3.
   For ``p`` / ``p_rgh``: use 101325 Pa unless the table or
   case_defaults says otherwise.

6. STRUCTURE IS SACRED.
   Do NOT modify, rename, add, or remove any ``type ...;`` line,
   patch name, FoamFile header field, ``dimensions`` line, comment,
   or whitespace pattern.  Output must contain the exact same set
   of patch entries as the template, in the same order.

7. OUTPUT FORMAT.
   Emit the COMPLETE file starting with ``FoamFile``.  No markdown
   fences, no commentary, no leading or trailing prose.  The first
   non-blank line of your answer MUST be ``FoamFile``."""

    return "==========================================================\nRULES — follow each one.\n\n" + body


def _template_section(content: str) -> str:
    """The template — what to rewrite."""
    return (
        "==========================================================\n"
        "DETERMINISTIC TEMPLATE — REPLACE THE PLACEHOLDERS:\n"
        f"{content}"
    )
