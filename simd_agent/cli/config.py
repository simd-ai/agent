"""CLI configuration — XDG file + env-var overrides.

Persistent state lives at ``~/.config/simd/config.toml`` (or
``$XDG_CONFIG_HOME/simd/config.toml`` if set).  The file is created
on the first ``simd run`` and updated automatically as the cached
identifiers change.  No login command; no account.

Environment-variable overrides:

  - ``SIMD_AGENT``      → ``agent_url``       (e.g. http://localhost:8000)
  - ``SIMD_TOKEN``      → ``token``           (reserved — Neon Auth)
  - ``SIMD_USER_ID``    → ``user_id``         (open-mode user UUID)
  - ``SIMD_PROJECT``    → ``simulation_id``   (active simulation UUID)

Command-line flags (``--agent``, ``--user``, ``--project``) override
both file and env-var values.  Lookup precedence:

    flag > env-var > file > built-in default
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


_DEFAULT_AGENT_URL = "http://localhost:8000"


def _config_dir() -> Path:
    """Where the CLI keeps its state.  Honours ``$XDG_CONFIG_HOME``."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "simd"


def _config_path() -> Path:
    return _config_dir() / "config.toml"


@dataclass(slots=True)
class CliConfig:
    """In-memory view of the CLI's persistent state."""

    agent_url: str = _DEFAULT_AGENT_URL
    token: str | None = None
    user_id: str | None = None
    simulation_id: str | None = None
    last_run_id: str | None = None
    # Free-form extras for forward-compatibility (e.g. provider hint).
    extras: dict[str, Any] = field(default_factory=dict)

    # ── persistence ───────────────────────────────────────────

    @classmethod
    def load(cls) -> "CliConfig":
        """Read the config file, then apply env-var overrides.

        Returns a default-valued ``CliConfig`` when the file doesn't
        exist — callers should still call :meth:`save` after mutating.
        """
        path = _config_path()
        data: dict[str, Any] = {}
        if path.is_file():
            try:
                data = tomllib.loads(path.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                # Corrupt or unreadable — start fresh rather than error out.
                data = {}

        cfg = cls(
            agent_url=     data.get("agent_url",     _DEFAULT_AGENT_URL),
            token=         data.get("token"),
            user_id=       data.get("user_id"),
            simulation_id= data.get("simulation_id"),
            last_run_id=   data.get("last_run_id"),
            extras=        {k: v for k, v in data.items()
                            if k not in {
                                "agent_url", "token", "user_id",
                                "simulation_id", "last_run_id",
                            }},
        )

        # env-var overrides (one level up from the file)
        if env := os.environ.get("SIMD_AGENT"):
            cfg.agent_url = env
        if env := os.environ.get("SIMD_TOKEN"):
            cfg.token = env
        if env := os.environ.get("SIMD_USER_ID"):
            cfg.user_id = env
        if env := os.environ.get("SIMD_PROJECT"):
            cfg.simulation_id = env

        return cfg

    def save(self) -> None:
        """Persist to ``~/.config/simd/config.toml`` (creates parent dirs)."""
        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Hand-roll a minimal TOML writer — the stdlib ``tomllib`` is
        # read-only, and dragging in ``tomli_w`` for two dozen lines of
        # output isn't worth a new dependency.
        lines: list[str] = []
        data = asdict(self)
        extras = data.pop("extras", {})
        for key, value in {**data, **extras}.items():
            if value is None:
                continue
            if isinstance(value, str):
                # Escape backslashes and quotes for safe TOML output.
                v = value.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{key} = "{v}"')
            else:
                lines.append(f"{key} = {value!r}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # 0600 — token isn't a real secret in open mode, but treat it
        # as one anyway.
        try:
            path.chmod(0o600)
        except OSError:
            pass

    # ── small convenience methods ─────────────────────────────

    def auth_header(self) -> dict[str, str]:
        """Build the Authorization header (if any) for HTTP calls."""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    @property
    def ws_url(self) -> str:
        """Derive the WebSocket base URL from ``agent_url``."""
        return self.agent_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
