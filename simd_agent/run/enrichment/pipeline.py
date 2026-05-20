# simd_agent/run/enrichment/pipeline.py
"""Composable enrichment pipeline.

The pipeline turns a freshly-linted ``validated_config`` into a fully-
populated config that every downstream consumer (``CaseSpec``,
``RegionSpec``, the BC renderer, the LLM filler, the prompt pack) can
read without re-deriving "what did the user actually mean?".

Adding a new step
-----------------
1. Create ``simd_agent/run/enrichment/<your_step>.py`` exporting an
   ``async def apply(ctx: EnrichmentContext) -> None``.
2. Have it mutate ``ctx.config`` and call ``ctx.add_info`` /
   ``ctx.add_warning`` / ``ctx.add_error`` for diagnostics.
3. Append the step to :data:`DEFAULT_STEPS` below in the right position
   relative to its dependencies (e.g. anything that reads
   ``case_defaults`` must come after :mod:`case_defaults`).
4. Add a unit test under ``tests/test_enrichment_<your_step>.py``.

Failure semantics
-----------------
Each step decides its own.  Transient / recoverable failures should be
caught inside the step and surfaced as a ``warning`` issue; programming
bugs (KeyError, TypeError, …) bubble up so they are visible in logs.
The pipeline halts on the first ``error``-severity issue so the
orchestrator can surface a config-incomplete state without running
the downstream steps against a known-bad config.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from simd_agent.run.enrichment.context import EnrichmentContext, EnrichmentIssue
from simd_agent.run.enrichment import (
    case_defaults,
    inlet_bcs,
    region_details,
    region_fluid_presets,
    region_inits,
    region_topology,
    topology_lint,
    turbulence_bcs,
    wall_bcs,
)

logger = logging.getLogger(__name__)


Step = Callable[[EnrichmentContext], Awaitable[None]]


# Order matters — later steps may depend on earlier ones.
#
#   case_defaults    must precede region_inits (which reads case_defaults).
#   region_topology  must precede region_details (which refines an
#                    existing regions block) and region_inits / inlet_bcs /
#                    turbulence_bcs.
#   region_details   should run before region_inits so the extractor's
#                    prompt-derived values win over the case-level
#                    fallback (region_inits never overrides non-None).
#   region_inits     must run before inlet_bcs (propagates region inits
#                    into per-patch BCs) AND before region_fluid_presets
#                    (which keys its inference off T_init).
#   region_fluid_presets
#                    must run after region_inits.  Reads (T_init, case
#                    fluid name, bulk_density) and replaces the heuristic
#                    "air" default with the real cryogen / water / oil
#                    preset so the deterministic renderer picks the right
#                    Cp / ρ / μ from FLUID_REGION_PRESETS.
#   inlet_bcs        writes T / U on inlets.
#   turbulence_bcs   writes k / ω / ε / ν_t on inlets.  Independent of
#                    inlet_bcs at the field level, grouped after it for
#                    log ordering.
#   wall_bcs         writes T on real (non-CHT-coupled) walls — bridges
#                    the lowercase ``temperature`` user-input key to the
#                    uppercase ``T`` key the renderer reads.  Reads
#                    case_defaults so must come after it.
#   topology_lint    runs last so it sees the fully-populated regions
#                    block; emitting ``error`` issues here halts the
#                    pipeline.
DEFAULT_STEPS: tuple[Step, ...] = (
    case_defaults.apply,
    region_topology.apply,
    region_details.apply,
    region_inits.apply,
    region_fluid_presets.apply,
    inlet_bcs.apply,
    turbulence_bcs.apply,
    wall_bcs.apply,
    topology_lint.apply,
)


async def enrich_validated_config(
    *,
    config: dict[str, Any],
    user_requirements: str,
    steps: tuple[Step, ...] = DEFAULT_STEPS,
) -> list[EnrichmentIssue]:
    """Apply the enrichment pipeline to a validated config.

    Args:
        config: The validated config produced by ``CFDLinter.lint()``.
            Mutated in place.  After this call, every key the
            downstream consumers depend on is populated.
        user_requirements: The user's natural-language prompt.  Consumed
            by LLM-based steps; ``""`` is acceptable for non-LLM tests.
        steps: Override for tests / custom flows.  Defaults to
            :data:`DEFAULT_STEPS`.

    Returns:
        The collected :class:`EnrichmentIssue` list.  The orchestrator
        forwards ``warning`` issues to the UI as ``region_lint_warning``
        events and aborts the run on any ``error`` issue.
    """
    ctx = EnrichmentContext(config=config, user_requirements=user_requirements or "")

    logger.info("[ENRICH] pipeline start (%d steps)", len(steps))
    for step in steps:
        step_name = _short_step_name(step)
        logger.debug("[ENRICH] → %s", step_name)
        await step(ctx)
        if ctx.has_errors():
            logger.warning(
                "[ENRICH] halting after %s (error issues present)", step_name,
            )
            break
    logger.info(
        "[ENRICH] pipeline done — %d issue(s): %s",
        len(ctx.issues),
        [(i.severity, i.step, i.code) for i in ctx.issues],
    )
    return ctx.issues


def _short_step_name(step: Step) -> str:
    """Pull the step's module basename for logging (``case_defaults`` etc.)."""
    module = getattr(step, "__module__", "?")
    return module.rsplit(".", 1)[-1]
