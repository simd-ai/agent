"""Anonymized usage telemetry via Umami.

Captures high-level simulation events (precheck, run start/complete/fail)
so we can understand what solvers, physics configs, and fluids are used.

Opt-out: set TELEMETRY_ENABLED=false in .env or environment.
No telemetry is sent if UMAMI_WEBSITE_ID is not configured.

No PII is captured — only solver names, physics flags, mesh cell counts,
durations, and success/failure status.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_telemetry: "Telemetry | None" = None


# ---------------------------------------------------------------------------
# Event definitions
# ---------------------------------------------------------------------------

@dataclass
class TelemetryEvent:
    """Base telemetry event. Subclasses set `name` and add fields."""
    name: str = ""

    @property
    def properties(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "name" and v is not None}


@dataclass
class PrecheckCompleted(TelemetryEvent):
    name: str = field(default="precheck_completed", init=False)
    solver: str | None = None
    flow_regime: str | None = None
    time_scheme: str | None = None
    compressibility: str | None = None
    heat_transfer: bool | None = None
    gravity: bool | None = None
    fluid: str | None = None
    turbulence_model: str | None = None
    patch_count: int | None = None
    phase_change_detected: bool | None = None
    duration_s: float | None = None


@dataclass
class RunStarted(TelemetryEvent):
    name: str = field(default="run_started", init=False)
    solver: str | None = None
    flow_regime: str | None = None
    time_scheme: str | None = None
    compressibility: str | None = None
    heat_transfer: bool | None = None
    gravity: bool | None = None
    fluid: str | None = None
    turbulence_model: str | None = None
    mesh_cells: int | None = None
    patch_count: int | None = None


@dataclass
class RunCompleted(TelemetryEvent):
    name: str = field(default="run_completed", init=False)
    solver: str | None = None
    success: bool | None = None
    duration_s: float | None = None
    retry_count: int | None = None
    mesh_cells: int | None = None


@dataclass
class RunFailed(TelemetryEvent):
    name: str = field(default="run_failed", init=False)
    solver: str | None = None
    error_type: str | None = None
    error_summary: str | None = None
    retry_count: int | None = None
    duration_s: float | None = None


@dataclass
class SolverSelected(TelemetryEvent):
    name: str = field(default="solver_selected", init=False)
    solver: str | None = None
    confidence: float | None = None
    was_fallback: bool | None = None
    flow_regime: str | None = None
    heat_transfer: bool | None = None
    compressibility: str | None = None


@dataclass
class ChatQuery(TelemetryEvent):
    name: str = field(default="chat_query", init=False)
    mode: str | None = None
    has_simulation: bool | None = None


@dataclass
class ReportGenerated(TelemetryEvent):
    name: str = field(default="report_generated", init=False)
    solver: str | None = None
    flow_regime: str | None = None
    has_results: bool | None = None


@dataclass
class UserSignedUp(TelemetryEvent):
    name: str = field(default="user_signed_up", init=False)


@dataclass
class ProjectCreated(TelemetryEvent):
    name: str = field(default="project_created", init=False)


@dataclass
class MeshUploaded(TelemetryEvent):
    name: str = field(default="mesh_uploaded", init=False)
    cell_count: int | None = None
    patch_count: int | None = None


@dataclass
class CaseExported(TelemetryEvent):
    name: str = field(default="case_exported", init=False)
    solver: str | None = None


@dataclass
class ResultsViewed(TelemetryEvent):
    name: str = field(default="results_viewed", init=False)


@dataclass
class RunCancelled(TelemetryEvent):
    name: str = field(default="run_cancelled", init=False)


@dataclass
class UsageLimitHit(TelemetryEvent):
    name: str = field(default="usage_limit_hit", init=False)
    limit_type: str | None = None
    current_count: int | None = None


@dataclass
class ProjectDeleted(TelemetryEvent):
    name: str = field(default="project_deleted", init=False)


@dataclass
class SimulationResubmitted(TelemetryEvent):
    name: str = field(default="simulation_resubmitted", init=False)
    solver: str | None = None


# ---------------------------------------------------------------------------
# Telemetry service (singleton)
# ---------------------------------------------------------------------------

class Telemetry:
    """Umami telemetry client. No-op when disabled or unconfigured."""

    def __init__(self) -> None:
        from simd_agent.settings import get_settings
        settings = get_settings()

        self._website_id: str | None = None
        self._api_url: str | None = None

        if not settings.telemetry_enabled:
            logger.debug("[Telemetry] Disabled via TELEMETRY_ENABLED=false")
            return

        if not settings.umami_website_id:
            logger.debug("[Telemetry] No UMAMI_WEBSITE_ID configured, telemetry inactive")
            return

        self._website_id = settings.umami_website_id
        self._api_url = f"{settings.umami_host_url.rstrip('/')}/api/send"
        logger.info("[Telemetry] Umami telemetry enabled → %s", self._api_url)

    def capture(self, event: TelemetryEvent, user_id: str | None = None) -> None:
        """Send a telemetry event to Umami."""
        if self._website_id is None:
            return
        try:
            payload = {
                "type": "event",
                "payload": {
                    "website": self._website_id,
                    "name": event.name,
                    "url": f"/api/{event.name}",
                    "hostname": "api.simd.space",
                    "data": event.properties,
                },
            }
            resp = httpx.post(
                self._api_url,
                json=payload,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
                timeout=5.0,
            )
            if resp.status_code != 200:
                logger.debug("[Telemetry] Umami returned %s for %s", resp.status_code, event.name)
        except Exception as e:
            logger.debug("[Telemetry] Failed to capture %s: %s", event.name, e)

    def flush(self) -> None:
        """No-op — events are sent synchronously."""
        pass

    def shutdown(self) -> None:
        """No-op — no background threads."""
        pass


def get_telemetry() -> Telemetry:
    """Get or create the singleton telemetry instance."""
    global _telemetry
    if _telemetry is None:
        _telemetry = Telemetry()
    return _telemetry
