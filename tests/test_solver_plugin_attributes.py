# tests/test_solver_plugin_attributes.py
"""Tests for the :class:`SolverPlugin` class-attribute contract.

The base class declares boolean role flags (``is_transient``,
``is_compressible``, ``supports_energy``, ``needs_gravity``,
``is_multiphase``, ``is_multi_region``) with sensible defaults so
every registered plugin has them — without these defaults, code that
branches on ``self.is_multi_region`` (notably
:meth:`SolverPlugin.validate_full`) raises ``AttributeError`` on any
plugin that doesn't explicitly set the flag.

This file pins that contract:

  * Every registered plugin exposes the full attribute set.
  * The default for ``is_multi_region`` is ``False`` (single-region
    plugins inherit it).
  * Multi-region (``MultiRegionBase``) subclasses override it to
    ``True``.
  * :meth:`validate_full` runs on a single-region plugin without
    raising — regression for the user-reported
    ``AttributeError: 'SimpleFoamSolver' object has no attribute
    'is_multi_region'``.
"""

from __future__ import annotations

import pytest

from simd_agent.solvers import get_registry


# Boolean role flags every plugin must expose so any consumer can
# branch on them without ``hasattr`` / ``getattr(_, _, False)`` guards.
_REQUIRED_BOOL_ATTRS = (
    "is_transient",
    "is_compressible",
    "supports_energy",
    "needs_gravity",
    "is_multiphase",
    "is_multi_region",
)


def _all_plugins():
    """Every registered plugin instance — drives the parametrised tests."""
    return list(get_registry().all_solvers())


@pytest.mark.parametrize("plugin", _all_plugins(), ids=lambda p: p.name if p else "?")
def test_every_plugin_exposes_required_bool_flags(plugin):
    """No plugin is missing one of the standard role flags."""
    assert plugin is not None
    for attr in _REQUIRED_BOOL_ATTRS:
        assert hasattr(plugin, attr), (
            f"plugin {plugin.name!r} is missing class attribute {attr!r}"
        )
        assert isinstance(getattr(plugin, attr), bool), (
            f"plugin {plugin.name!r}.{attr} is not a bool"
        )


def test_single_region_plugins_default_is_multi_region_to_false():
    """Single-region plugins inherit ``is_multi_region = False`` from the base."""
    reg = get_registry()
    for name in ("simpleFoam", "pimpleFoam", "rhoSimpleFoam", "rhoPimpleFoam",
                 "buoyantSimpleFoam"):
        plugin = reg.get(name)
        if plugin is None:
            continue  # tolerate optional plugins
        assert plugin.is_multi_region is False, (
            f"{name} unexpectedly marked as multi-region"
        )


def test_cht_plugins_override_is_multi_region_to_true():
    reg = get_registry()
    for name in ("chtMultiRegionSimpleFoam", "chtMultiRegionFoam"):
        plugin = reg.get(name)
        assert plugin is not None, f"missing CHT plugin {name!r}"
        assert plugin.is_multi_region is True


def test_no_attribute_error_landmines_in_mro_for_any_plugin():
    """Sweep every plugin × every ``self.X`` reachable through its MRO.

    Catches the same bug class as :func:`is_multi_region` for any
    attribute we might overlook in the future.  Approach:

      1. AST-scan each class in the solver module tree for ``self.X``
         reads in its methods.
      2. For every registered plugin, walk its MRO and union the
         per-class read-sets — that is the exact set of attribute reads
         the plugin's code paths can reach.
      3. For each such attribute, ``hasattr(plugin, attr)`` must be True.

    Filters out names that are dataclass / local-variable references
    (``severity``, ``message`` etc.) which happen to share a name with
    ``self.X`` reads in non-plugin classes that no plugin inherits from.
    """
    import ast
    from collections import defaultdict
    from pathlib import Path

    from simd_agent.solvers import get_registry

    # Names that are NEVER plugin attributes — dataclass fields with
    # accidentally-overlapping ``self.X`` reads.  Whitelisting these
    # keeps the audit precise; if a new dataclass field shows up here,
    # add it.
    skip = {"file", "files", "severity", "message", "score", "issues", "fix", "items"}

    # Source roots — every file that can contribute a method to a plugin's MRO.
    roots: list[Path] = [
        Path("simd_agent/solvers/base.py"),
        Path("simd_agent/solvers/families/_steady.py"),
        Path("simd_agent/solvers/families/_transient.py"),
        Path("simd_agent/solvers/families/_compressible.py"),
        Path("simd_agent/solvers/families/_boussinesq.py"),
        Path("simd_agent/solvers/families/_multi_region.py"),
    ]
    roots.extend(Path("simd_agent/solvers").rglob("solver.py"))

    class_attr_reads: dict[str, set[str]] = defaultdict(set)

    class Scanner(ast.NodeVisitor):
        def __init__(self):
            self.cur_class: str | None = None
        def visit_ClassDef(self, node):
            outer = self.cur_class
            self.cur_class = node.name
            self.generic_visit(node)
            self.cur_class = outer
        def visit_FunctionDef(self, node):
            if self.cur_class:
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Attribute)
                            and isinstance(sub.value, ast.Name)
                            and sub.value.id == "self"):
                        class_attr_reads[self.cur_class].add(sub.attr)
        visit_AsyncFunctionDef = visit_FunctionDef

    for p in roots:
        Scanner().visit(ast.parse(p.read_text()))

    missing: list[tuple[str, str, str]] = []
    for plugin in get_registry().all_solvers():
        for klass in type(plugin).__mro__:
            for attr in class_attr_reads.get(klass.__name__, ()):
                if attr in skip or attr.startswith("_"):
                    continue
                if not hasattr(plugin, attr):
                    missing.append((plugin.name, attr, klass.__name__))

    assert not missing, (
        "Plugin attribute references unresolved through MRO — these will "
        "raise AttributeError at runtime:\n"
        + "\n".join(
            f"  {name}.{attr}  (read by {src})"
            for name, attr, src in missing
        )
    )


def test_validate_full_runs_on_single_region_plugin_without_attribute_error():
    """Regression for the AttributeError reported when running simpleFoam end-to-end.

    The failure was at base.py:240 — ``if not self.is_multi_region`` raised
    AttributeError because the base class never declared the attribute and
    single-region plugins never explicitly set it.  This test exercises the
    same code path with a minimal config so the bug can't come back.
    """
    plugin = get_registry().get("simpleFoam")
    assert plugin is not None

    files = {
        "0/U": (
            "FoamFile { class volVectorField; object U; }\n"
            "dimensions [0 1 -1 0 0 0 0];\n"
            "internalField uniform (0 0 0);\n"
            "boundaryField\n{\n"
            "    inlet  { type fixedValue; value uniform (1 0 0); }\n"
            "    outlet { type zeroGradient; }\n"
            "    wall   { type noSlip; }\n"
            "}\n"
        ),
        "0/p": (
            "FoamFile { class volScalarField; object p; }\n"
            "dimensions [0 2 -2 0 0 0 0];\n"
            "internalField uniform 0;\n"
            "boundaryField\n{\n"
            "    inlet  { type zeroGradient; }\n"
            "    outlet { type fixedValue; value uniform 0; }\n"
            "    wall   { type zeroGradient; }\n"
            "}\n"
        ),
    }
    config = {
        "solver": "simpleFoam",
        "physics": {"compressibility": "incompressible", "flow_regime": "laminar"},
        "mesh": {"patches": [
            {"name": "inlet", "type": "patch"},
            {"name": "outlet", "type": "patch"},
            {"name": "wall", "type": "wall"},
        ]},
        "fluid": {"name": "water", "density": 998.0, "kinematic_viscosity": 1.0e-6},
        "boundary_conditions": {},
        "turbulence_model": "laminar",
    }

    # Should NOT raise AttributeError — that was the original failure.
    result = plugin.validate_full(files, config)
    assert result is not None
    assert "0/U" in result.files
    assert "0/p" in result.files
