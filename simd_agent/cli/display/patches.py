"""Interactive patch-review REPL.

After precheck returns its suggested boundary conditions, the CLI lets
the user inspect and edit them before the run starts.  Mirrors the
patch-table editor in the frontend, terminal-style.

UX in plain English:

  - print a table of every patch with its proposed role/type/value
  - prompt for an action:
        [enter]  accept everything as proposed
        e        edit one patch
        d        dump the full config to stdout (debug)
        q        cancel the run

The edit flow drills into a single patch and lets the user change
``type`` or ``value``.  Coupled CHT interfaces (``*_to_*``) are
read-only — they're managed by the deterministic renderer.

Output of :func:`review` is a dict the caller merges into the
simulation_config before sending to ``/ws/run``.  Returning ``None``
means the user cancelled.
"""

from __future__ import annotations

from typing import Any

from rich.table import Table

from simd_agent.cli.display import console
from simd_agent.cli.display.arrow import arrow_choice


# CHT coupling patches are owned by the deterministic renderer; the user
# can't meaningfully change them via the CLI.
def _is_coupled(patch_name: str) -> bool:
    return "_to_" in patch_name


# ── Table rendering ──────────────────────────────────────────


def _render_table(patches: list[dict[str, Any]]) -> Table:
    """Build a Rich table summarising every patch."""
    t = Table(
        show_header=True,
        header_style="bold",
        title_style="bold",
        row_styles=["", "dim"],
        title_justify="left",
        pad_edge=False,
    )
    t.add_column("#",      style="cyan", width=3, justify="right")
    t.add_column("patch",  style="bold")
    t.add_column("role")
    t.add_column("type")
    t.add_column("value",  overflow="fold")

    for i, p in enumerate(patches, start=1):
        name  = p.get("name") or p.get("patch") or "?"
        role  = p.get("role") or p.get("patch_class") or "?"
        bcs   = _summarise_bcs(p)
        typ   = bcs["type"]
        value = bcs["value"]
        style = "dim italic" if _is_coupled(name) else None
        t.add_row(str(i), name, role, typ, value, style=style)
    return t


def _summarise_bcs(patch: dict[str, Any]) -> dict[str, str]:
    """Collapse a patch's per-field BCs into a one-line table cell.

    The precheck response carries per-field BCs (T, U, p, k, omega, ...).
    For the overview table we just show the dominant type + a compact
    value summary — the user can drill in for details.
    """
    fields = patch.get("fields") or patch.get("bc") or patch.get("boundary_conditions") or {}

    # Dominant type — pick the first non-empty across U / T / p.
    type_str = "—"
    for key in ("U", "T", "p", "p_rgh"):
        f = fields.get(key) if isinstance(fields, dict) else None
        if isinstance(f, dict) and f.get("type"):
            type_str = f["type"]
            break

    # Value summary — show what we have for U / T / p
    parts: list[str] = []
    if isinstance(fields, dict):
        for key in ("T", "U", "p", "p_rgh"):
            f = fields.get(key)
            if isinstance(f, dict) and "value" in f and f["value"] is not None:
                v = f["value"]
                if isinstance(v, (list, tuple)) and len(v) >= 3:
                    parts.append(f"{key}=({v[0]:g},{v[1]:g},{v[2]:g})")
                else:
                    parts.append(f"{key}={v}")
    value_str = ", ".join(parts) if parts else "—"
    return {"type": type_str, "value": value_str}


# ── Main entry point ─────────────────────────────────────────


def review(
    patches: list[dict[str, Any]],
    *,
    auto_accept: bool = False,
) -> dict[str, Any] | None:
    """Run the patch-review REPL.

    Args:
        patches: list of patch records from the precheck response.
            Each must carry at least ``name`` and ``fields``.
        auto_accept: skip the prompt (``--yes`` flag).

    Returns:
        A dict ``{patch_name: {field: {type, value}}}`` of overrides
        the user made, or ``None`` if they cancelled.
    """
    overrides: dict[str, Any] = {}

    if auto_accept:
        return overrides

    console.print(_render_table(patches))

    while True:
        console.print("")
        try:
            idx = arrow_choice(
                "what next?",
                [
                    "accept            — start the run with these BCs",
                    "edit a patch      — change a single BC type or value",
                    "dump full config  — re-print the table",
                    "cancel            — abort this run",
                ],
            )
        except KeyboardInterrupt:
            return None

        if idx == 0:
            return overrides
        if idx == 1:
            _edit_loop(patches, overrides)
            continue
        if idx == 2:
            console.print(_render_table(patches))
            continue
        if idx == 3:
            return None


def _edit_loop(
    patches: list[dict[str, Any]],
    overrides: dict[str, Any],
) -> None:
    """Inner REPL: pick a patch, edit one field, return to the outer."""
    # Build menu options — show patch name + the proposed type, so the
    # user has enough context to decide what to edit without re-running
    # ``dump full config``.  Append a "back" entry so cancellation is
    # a one-arrow action; KeyboardInterrupt also bails cleanly.
    options = []
    for p in patches:
        name = p.get("name") or "?"
        fields = p.get("fields") or {}
        # Pick the most-interesting BC type to show as a hint.
        hint = ""
        for key in ("U", "T", "p", "p_rgh"):
            if key in fields and isinstance(fields[key], dict):
                hint = f"  {key}={fields[key].get('type', '?')}"
                break
        options.append(f"{name}{hint}")
    options.append("← back")

    try:
        idx = arrow_choice("which patch?", options)
    except KeyboardInterrupt:
        return
    if idx == len(options) - 1:
        return  # "back"
    target = patches[idx]

    name = target.get("name") or "?"
    if _is_coupled(name):
        console.print(
            f"  [yellow]{name}[/yellow] is a CHT coupling interface — "
            "managed by the deterministic renderer, can't edit here."
        )
        return

    console.print(f"\n  [bold]{name}[/]")
    # Build the field menu from what the patch actually carries, plus the
    # standard set as fallback so the user can also add a field that
    # precheck didn't propose.
    proposed = list((target.get("fields") or {}).keys())
    standard = ["U", "T", "p", "p_rgh", "k", "omega", "epsilon", "nut"]
    field_options = list(dict.fromkeys(proposed + standard))  # preserve order, dedupe
    field_options.append("← back")
    try:
        f_idx = arrow_choice("which field?", field_options)
    except KeyboardInterrupt:
        return
    if f_idx == len(field_options) - 1:
        return
    field = field_options[f_idx]

    new_type  = input(f"  new type for {field} (or blank to keep): ").strip() or None
    new_value = input(f"  new value for {field} (or blank to keep): ").strip() or None

    if new_type is None and new_value is None:
        console.print("  (no change)")
        return

    # Coerce numeric / vector values where possible.
    parsed_value: Any = new_value
    if new_value is not None:
        parsed_value = _parse_value(new_value)

    overrides.setdefault(name, {}).setdefault(field, {})
    if new_type is not None:
        overrides[name][field]["type"] = new_type
    if new_value is not None:
        overrides[name][field]["value"] = parsed_value
    console.print(f"  ✓ {name}.{field} updated")


def _parse_value(raw: str) -> Any:
    """Best-effort parse ``77``, ``(0.5 0 0)``, ``101325`` into JSON values."""
    raw = raw.strip()
    # Vector form (x y z) or (x, y, z) or [x, y, z]
    if raw.startswith(("(", "[")) and raw.endswith((")", "]")):
        inner = raw[1:-1].replace(",", " ")
        try:
            return [float(x) for x in inner.split() if x]
        except ValueError:
            return raw
    # Scalar
    try:
        return float(raw)
    except ValueError:
        return raw
