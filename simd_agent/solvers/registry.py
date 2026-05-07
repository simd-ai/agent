# simd_agent/solvers/registry.py
"""Auto-discovery registry for solver plugins.

At startup the registry scans ``simd_agent/solvers/`` for sub-packages
that export a ``SolverPlugin`` subclass.  No core code changes are needed
to add a new solver — just drop a package in this directory.

Usage::

    from simd_agent.solvers import get_registry

    registry = get_registry()
    solver = registry.get("simpleFoam")
    best   = registry.best_match(validated_config)
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any

from simd_agent.solvers.base import MatchResult, SolverPlugin

logger = logging.getLogger(__name__)

_SOLVERS_PKG = "simd_agent.solvers"
_SOLVERS_DIR = Path(__file__).parent


class SolverRegistry:
    """Central registry of all available solver plugins."""

    def __init__(self) -> None:
        self._solvers: dict[str, SolverPlugin] = {}
        self._discovered = False

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, plugin: SolverPlugin) -> None:
        """Register a solver plugin instance."""
        if not plugin.name:
            raise ValueError(f"Solver plugin {type(plugin).__name__} has no name set")
        if plugin.name in self._solvers:
            logger.warning(
                "Solver '%s' already registered (from %s), overwriting with %s",
                plugin.name,
                type(self._solvers[plugin.name]).__name__,
                type(plugin).__name__,
            )
        self._solvers[plugin.name] = plugin
        logger.debug("Registered solver plugin: %s", plugin.name)

    def discover(self) -> None:
        """Auto-discover solver plugins from sub-packages.

        Scans every direct sub-package of ``simd_agent/solvers/`` and looks
        for a module-level attribute named ``solver_plugin`` (a SolverPlugin
        instance) or a class named ``Solver`` (instantiated automatically).

        Convention for a solver package ``simd_agent/solvers/simpleFoam/``::

            # __init__.py
            from simd_agent.solvers.simpleFoam.solver import SimpleFoamSolver
            solver_plugin = SimpleFoamSolver()

        Or shorter::

            # __init__.py
            from .solver import Solver
        """
        if self._discovered:
            return

        for importer, modname, ispkg in pkgutil.iter_modules(
            [str(_SOLVERS_DIR)], prefix=f"{_SOLVERS_PKG}."
        ):
            if not ispkg:
                continue  # skip non-package modules (base.py, registry.py)

            try:
                mod = importlib.import_module(modname)
            except Exception:
                logger.warning("Failed to import solver package %s", modname, exc_info=True)
                continue

            # Convention 1: module-level `solver_plugin` instance
            plugin = getattr(mod, "solver_plugin", None)
            if isinstance(plugin, SolverPlugin):
                self.register(plugin)
                continue

            # Convention 2: module-level `Solver` class
            solver_cls = getattr(mod, "Solver", None)
            if solver_cls and isinstance(solver_cls, type) and issubclass(solver_cls, SolverPlugin):
                self.register(solver_cls())
                continue

            logger.debug(
                "Package %s has no solver_plugin instance or Solver class — skipping",
                modname,
            )

        self._discovered = True
        logger.info(
            "Solver discovery complete: %d solvers registered — %s",
            len(self._solvers),
            list(self._solvers.keys()),
        )

    # ── Lookup ────────────────────────────────────────────────────────────

    def get(self, name: str) -> SolverPlugin | None:
        """Get a solver plugin by name. Returns None if not found."""
        self.discover()
        return self._solvers.get(name)

    def get_or_raise(self, name: str) -> SolverPlugin:
        """Get a solver plugin by name. Raises KeyError if not found."""
        plugin = self.get(name)
        if plugin is None:
            available = ", ".join(sorted(self._solvers.keys()))
            raise KeyError(
                f"Solver '{name}' not found. Available: {available}"
            )
        return plugin

    def all_solvers(self) -> list[SolverPlugin]:
        """Return all registered solver plugins."""
        self.discover()
        return list(self._solvers.values())

    def names(self) -> list[str]:
        """Return sorted list of registered solver names."""
        self.discover()
        return sorted(self._solvers.keys())

    # ── Classification queries (replaces hardcoded module-level sets) ─────

    def allowed_solvers(self) -> set[str]:
        """All registered solver names."""
        self.discover()
        return set(self._solvers.keys())

    def p_solvers(self) -> set[str]:
        """Solvers whose primary pressure field is ``p``."""
        self.discover()
        return {p.name for p in self._solvers.values() if p.pressure_field == "p"}

    def p_rgh_solvers(self) -> set[str]:
        """Solvers whose primary pressure field is ``p_rgh``."""
        self.discover()
        return {p.name for p in self._solvers.values() if p.pressure_field == "p_rgh"}

    def energy_solvers(self) -> set[str]:
        """Solvers that solve an energy equation."""
        self.discover()
        return {p.name for p in self._solvers.values() if p.supports_energy}

    def gravity_solvers(self) -> set[str]:
        """Solvers that require ``constant/g``."""
        self.discover()
        return {p.name for p in self._solvers.values() if p.needs_gravity}

    def thermo_solvers(self) -> set[str]:
        """Solvers that require ``constant/thermophysicalProperties``.

        Currently identical to ``energy_solvers()``, but kept as a separate
        method so future non-energy thermo solvers (e.g. mixture transport
        without heat equation) can be flagged independently.
        """
        self.discover()
        return {p.name for p in self._solvers.values() if p.supports_energy}

    def best_match(self, config: dict[str, Any]) -> tuple[SolverPlugin, MatchResult] | None:
        """Find the solver with the highest match score for *config*.

        Returns (plugin, match_result) or None if no solver matches.
        """
        self.discover()
        best_plugin: SolverPlugin | None = None
        best_result: MatchResult | None = None

        for plugin in self._solvers.values():
            try:
                result = plugin.matches(config)
            except Exception:
                logger.warning(
                    "Solver %s.matches() failed", plugin.name, exc_info=True
                )
                continue

            if result.matches and (
                best_result is None or result.score > best_result.score
            ):
                best_plugin = plugin
                best_result = result

        if best_plugin and best_result:
            return best_plugin, best_result
        return None

    # ── Metadata for API / UI ─────────────────────────────────────────────

    def solver_info(self) -> list[dict[str, Any]]:
        """Return metadata about all solvers for the API / frontend."""
        self.discover()
        return [
            {
                "name": p.name,
                "algorithm": p.algorithm,
                "pressure_field": p.pressure_field,
                "is_transient": p.is_transient,
                "is_compressible": p.is_compressible,
                "supports_energy": p.supports_energy,
                "needs_gravity": p.needs_gravity,
                "is_multiphase": p.is_multiphase,
            }
            for p in sorted(self._solvers.values(), key=lambda s: s.name)
        ]


# ── Module-level singleton ────────────────────────────────────────────────

_registry: SolverRegistry | None = None


def get_registry() -> SolverRegistry:
    """Get the global solver registry (lazy singleton)."""
    global _registry
    if _registry is None:
        _registry = SolverRegistry()
        _registry.discover()
    return _registry
