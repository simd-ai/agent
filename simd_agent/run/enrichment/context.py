# simd_agent/run/enrichment/context.py
"""Shared context + diagnostic types passed through the enrichment pipeline.

Every step receives one :class:`EnrichmentContext` instance and mutates
``ctx.config`` in place, optionally appending :class:`EnrichmentIssue`
records for diagnostics.  No step ever creates a new config — the
caller (typically the orchestrator) owns the dict.

Why a mutable context instead of returning patches?
---------------------------------------------------
Steps need to read previous steps' outputs (e.g. ``region_inits``
reads ``case_defaults``).  A patch-then-apply style would force every
step to merge against the running config anyway; the mutable context
makes that bookkeeping the pipeline's job, not each step's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class EnrichmentIssue:
    """Diagnostic record produced by an enrichment step.

    Attributes:
        severity: ``"info"`` for routine progress, ``"warning"`` for
            recoverable problems (LLM failed, signal missing),
            ``"error"`` for fatal config problems (topology lint).
        step: Short identifier of the step that emitted the issue,
            typically the module's basename (e.g. ``"region_details"``).
            Used by the pipeline log prefix and by the orchestrator
            when forwarding warnings to the UI.
        code: Machine-readable code (e.g. ``"EXTRACTOR_FAILED"``,
            ``"REGION_LINT_DANGLING_INTERFACE"``).  Stable string —
            don't rewrite without updating consumers.
        message: Human-readable explanation.
        payload: Optional structured details (LLM response excerpt,
            patch list, …) — kept out of ``message`` so the UI can
            render it separately.
    """

    severity: Severity
    step: str
    code: str
    message: str
    payload: dict[str, Any] | None = None


@dataclass
class EnrichmentContext:
    """Mutable context threaded through every enrichment step.

    Steps consume ``config`` (read + write) and ``user_requirements``
    (read only).  Diagnostic output goes through the ``add_*`` helpers
    so the issue list stays append-only and easy to inspect after the
    pipeline finishes.
    """

    config: dict[str, Any]
    user_requirements: str
    issues: list[EnrichmentIssue] = field(default_factory=list)

    # ── Issue helpers — keep call sites in step modules terse ──

    def add_info(
        self,
        step: str,
        *,
        code: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.issues.append(EnrichmentIssue("info", step, code, message, payload))

    def add_warning(
        self,
        step: str,
        *,
        code: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.issues.append(EnrichmentIssue("warning", step, code, message, payload))

    def add_error(
        self,
        step: str,
        *,
        code: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.issues.append(EnrichmentIssue("error", step, code, message, payload))

    def add_issue(
        self,
        *,
        severity: Severity,
        step: str,
        code: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Lower-level helper for steps that propagate severities from a
        wrapped module (e.g. :func:`topology_lint.apply` mirrors what
        ``lint_regions`` already classified)."""
        self.issues.append(EnrichmentIssue(severity, step, code, message, payload))

    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)
