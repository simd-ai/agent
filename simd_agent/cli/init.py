"""``simd init`` — interactive wizard that sets up the CLI.

Asks the user where each component runs (agent + simulation runner +
storage) and which LLM provider to use, then writes the resulting
configuration to two places:

  - ``~/.simd/.env``              env vars for the bundled stack
  - ``~/.config/simd/config.toml`` lookup state for every subcommand

The wizard is fully interactive by design — surfacing the choices
once is the whole point of having a setup command.  For non-
interactive (CI / scripting) use, set ``$SIMD_AGENT`` and the related
env vars and skip ``simd init`` entirely; the implicit-init path in
``simd run`` handles the bare minimum silently.

This module is also callable from ``simd run`` to inline the wizard
when no config exists yet (see :func:`maybe_init`).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from simd_agent.cli.config import CliConfig
from simd_agent.cli.display import console, err
from simd_agent.cli.process import (
    DockerMissing,
    docker_daemon_alive,
    env_file,
    port_in_use,
    simd_home,
    compose_file as _compose_file,
)


# ── Entry points ─────────────────────────────────────────────


async def run(args: argparse.Namespace, config: CliConfig) -> int:
    """``simd init`` subcommand handler."""
    console.print("\n[bold]simd init[/]")
    console.print(
        "let's set up where the agent + simulation runner live.  "
        "you can re-run `simd init` later to change any of these.\n"
    )

    answers = _prompt_all()
    if answers is None:
        return 1

    _write_env(answers)
    _copy_compose_file()
    _persist_config(config, answers)
    _print_summary(answers)
    return 0


async def maybe_init(config: CliConfig) -> bool:
    """Run the wizard inline if no config exists.

    Called by ``simd run`` when the user hasn't ``simd init``'d yet.
    Returns True on success (and as a side effect, the config has
    been updated).  Returns False if the user aborted or anything
    crashed — caller should propagate the failure.
    """
    if env_file().is_file() and config.user_id:
        return True
    console.print(
        "  [yellow]first run — let's set things up.  "
        "you can re-run `simd init` later to change these answers.[/]"
    )
    rc = await run(argparse.Namespace(), config)
    return rc == 0


# ── The wizard ───────────────────────────────────────────────


def _prompt_all() -> dict[str, Any] | None:
    """Walk every wizard question.  Returns the answer dict, or None if
    the user aborted."""
    try:
        agent_mode, agent_url = _prompt_agent_backend()
        runner_mode, runner_url = _prompt_runner_backend()
        provider, provider_secret = _prompt_llm_provider()
        storage = _prompt_storage()
    except (KeyboardInterrupt, EOFError):
        console.print("\n  cancelled.")
        return None

    return {
        "agent_mode":      agent_mode,
        "agent_url":       agent_url,
        "runner_mode":     runner_mode,
        "runner_url":      runner_url,
        "provider":        provider,
        "provider_secret": provider_secret,
        "storage":         storage,
    }


def _prompt_agent_backend() -> tuple[str, str]:
    """Question 1: where does the FastAPI agent run?"""
    console.print("[bold]where should the agent run?[/]")
    console.print("  1) local-docker      (recommended — bundled, simd manages it)")
    console.print("  2) local-bare-metal  (you run `uvicorn` yourself)")
    console.print("  3) remote            (point at an existing agent)")
    choice = _ask_choice("  > ", {"1", "2", "3"}, default="1")

    if choice == "1":
        if not _docker_ready_or_warn():
            console.print(
                "  [yellow]docker isn't ready — falling back to "
                "local-bare-metal.  install docker to use the bundled mode.[/]"
            )
            return "local-bare-metal", _ask_url(
                "  agent URL", default="http://localhost:8000"
            )
        port_warn(8000)
        return "local-docker", "http://localhost:8000"

    if choice == "2":
        return "local-bare-metal", _ask_url(
            "  agent URL", default="http://localhost:8000"
        )

    return "remote", _ask_url("  remote agent URL")


def _prompt_runner_backend() -> tuple[str, str]:
    """Question 2: where does the OpenFOAM runner run?"""
    console.print("\n[bold]where should the simulation runner (OpenFOAM) run?[/]")
    console.print("  1) local-docker      (recommended — bundled, simd manages it)")
    console.print("  2) local-bare-metal  (OpenFOAM v2406 on this machine)")
    console.print("  3) remote            (point at an existing runner)")
    choice = _ask_choice("  > ", {"1", "2", "3"}, default="1")

    if choice == "1":
        if not _docker_ready_or_warn():
            console.print(
                "  [yellow]docker isn't ready — falling back to "
                "local-bare-metal.[/]"
            )
            choice = "2"
        else:
            port_warn(9000)
            return "local-docker", "http://localhost:9000"

    if choice == "2":
        console.print(
            "  [yellow]heads up:[/] local-bare-metal mode needs OpenFOAM v2406 "
            "installed and the simulation-runner FastAPI app running.\n"
            "  install OpenFOAM v2406:  https://www.openfoam.com/news/main-news/openfoam-v2406\n"
            "  start the runner:         see simd-ai/simd-agent-simulation README"
        )
        return "local-bare-metal", _ask_url(
            "  runner URL", default="http://localhost:9000"
        )

    return "remote", _ask_url("  remote runner URL (e.g. http://1.2.3.4:9000)")


def _prompt_llm_provider() -> tuple[str, str | None]:
    """Question 3: which LLM provider?"""
    console.print("\n[bold]which LLM provider?[/]")
    console.print("  1) gemini  (Google AI Studio — easiest, has a daily cap)")
    console.print("  2) vertex  (GCP Vertex AI — no daily cap, needs SA JSON)")
    console.print("  3) ollama  (local — no API key, runs on this machine)")
    choice = _ask_choice("  > ", {"1", "2", "3"}, default="1")

    if choice == "1":
        key = _ask("  GEMINI_API_KEY (paste): ", masked=True)
        return "gemini", key
    if choice == "2":
        path = _ask("  path to service-account JSON: ")
        return "vertex", path
    return "ollama", None


def _prompt_storage() -> str:
    """Question 4: where do simulations land?"""
    console.print("\n[bold]where do simulations live?[/]")
    console.print("  1) local    (filesystem, fine for everyone)")
    console.print("  2) gcs      (Google Cloud Storage bucket)")
    choice = _ask_choice("  > ", {"1", "2"}, default="1")
    if choice == "2":
        bucket = _ask("  GCS bucket name: ")
        return f"gcs:{bucket}"
    return "local"


# ── Prompt helpers ───────────────────────────────────────────


def _ask(prompt: str, *, default: str | None = None, masked: bool = False) -> str:
    """One free-form input with an optional default."""
    if default is not None:
        prompt = f"{prompt}[{default}] "
    if masked:
        import getpass
        value = getpass.getpass(prompt).strip()
    else:
        value = input(prompt).strip()
    if not value and default is not None:
        return default
    if not value:
        # Keep asking — empty isn't a valid answer here.
        return _ask(prompt.split("[")[0], default=default, masked=masked)
    return value


def _ask_choice(prompt: str, valid: set[str], default: str) -> str:
    """One-character menu with a default on enter."""
    raw = input(prompt).strip() or default
    while raw not in valid:
        console.print(f"  unknown choice {raw!r}.  pick one of {sorted(valid)}.")
        raw = input(prompt).strip() or default
    return raw


def _ask_url(prompt: str, default: str | None = None) -> str:
    """Validate that the user typed a URL we can parse."""
    while True:
        raw = _ask(f"{prompt}: ", default=default)
        if raw.startswith(("http://", "https://")):
            return raw.rstrip("/")
        console.print(f"  {raw!r} doesn't look like a URL.  prefix http:// or https://")


# ── Side-effect helpers ──────────────────────────────────────


def _docker_ready_or_warn() -> bool:
    """Confirm docker is installed AND the daemon is running."""
    try:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(docker_daemon_alive())
    except RuntimeError:
        # Already inside an event loop — use a separate thread.
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(docker_daemon_alive())
        finally:
            loop.close()


def port_warn(port: int) -> None:
    """Soft-warn the user when a port we plan to use is already taken."""
    if port_in_use("127.0.0.1", port):
        console.print(
            f"  [yellow]warning:[/] port {port} on this machine is already "
            f"in use.  the bundled stack may fail to start until you free it "
            f"(`lsof -i :{port}`)."
        )


def _write_env(answers: dict[str, Any]) -> None:
    """Render ~/.simd/.env from the wizard answers."""
    home = simd_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "storage").mkdir(exist_ok=True)

    lines: list[str] = [
        "# ─── written by `simd init` ────────────────────────────────",
        "# Edit by hand or re-run `simd init` to regenerate.",
        "",
        "# ── Database (bundled postgres) ────────────────────────────",
        "DATABASE_URL=postgresql+asyncpg://simd:simd@postgres:5432/simd",
        "",
        "# ── Simulation runner ───────────────────────────────────────",
        f"SIMULATION_SERVER_URL={answers['runner_url']}",
        "",
        "# ── LLM provider ────────────────────────────────────────────",
    ]
    provider = answers["provider"]
    secret = answers.get("provider_secret")
    if provider == "gemini":
        lines.append("DEFAULT_PROVIDER=gemini")
        lines.append(f"GEMINI_API_KEY={secret}")
    elif provider == "vertex":
        lines.append("DEFAULT_PROVIDER=vertex")
        lines.append("VERTEX_PROJECT=  # fill in")
        lines.append("VERTEX_LOCATION=us-central1")
        lines.append(f"GOOGLE_APPLICATION_CREDENTIALS={secret}")
    elif provider == "ollama":
        lines.append("DEFAULT_PROVIDER=ollama")
        lines.append("OLLAMA_HOST=http://host.docker.internal:11434")

    lines.extend([
        "",
        "# ── Storage ─────────────────────────────────────────────────",
    ])
    storage = answers["storage"]
    if storage == "local":
        lines.append("STORAGE_BACKEND=local")
        lines.append("STORAGE_LOCAL_DIR=/app/storage")
    elif storage.startswith("gcs:"):
        bucket = storage.split(":", 1)[1]
        lines.append("STORAGE_BACKEND=gcs")
        lines.append(f"STORAGE_BUCKET={bucket}")

    lines.extend([
        "",
        "# ── Auth (open mode, no signup) ────────────────────────────",
        "# leave NEON_AUTH_BASE_URL unset",
        "",
        "# ── Self-healing ───────────────────────────────────────────",
        "MAX_RETRIES=7",
        "",
    ])

    env_file().write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        env_file().chmod(0o600)
    except OSError:
        pass


def _copy_compose_file() -> None:
    """Copy the repo's docker-compose.yml into ~/.simd/.

    The bundled-mode CLI runs ``docker compose -f ~/.simd/docker-compose.yml``
    so the user can ``pip install`` simd from a wheel without needing
    the source tree present.
    """
    source = _find_compose_source()
    if source is None:
        # When installed from PyPI without the source tree alongside,
        # the compose file ships with the package data.  Skip for now;
        # `simd up` will surface the missing file.
        return
    target = _compose_file()
    if target.exists():
        return  # don't clobber user edits
    shutil.copy(source, target)


def _find_compose_source() -> Path | None:
    """Locate ``docker/docker-compose.yml`` in the installed package's tree."""
    # When developing from the source tree, the file lives at the repo
    # root alongside ``simd_agent/``.
    import simd_agent
    pkg = Path(simd_agent.__file__).resolve().parent
    candidate = pkg.parent / "docker" / "docker-compose.yml"
    if candidate.is_file():
        return candidate
    return None


def _persist_config(config: CliConfig, answers: dict[str, Any]) -> None:
    """Stamp the agent URL + backend modes into ``~/.config/simd/config.toml``."""
    config.agent_url = answers["agent_url"]
    config.extras["runner_url"]   = answers["runner_url"]
    config.extras["agent_mode"]   = answers["agent_mode"]
    config.extras["runner_mode"]  = answers["runner_mode"]
    config.save()


def _print_summary(answers: dict[str, Any]) -> None:
    """One-screen confirmation of what got written."""
    console.print("\n[bold green]✓ setup saved[/]")
    console.print(f"  agent       {answers['agent_url']}  ({answers['agent_mode']})")
    console.print(f"  runner      {answers['runner_url']}  ({answers['runner_mode']})")
    console.print(f"  provider    {answers['provider']}")
    console.print(f"  storage     {answers['storage']}")
    console.print(f"  env file    {env_file()}")

    if answers["agent_mode"] == "local-docker":
        console.print(
            "\n  next: `simd up` to start the stack, or just run "
            "`simd run …` — it'll auto-start."
        )
    elif answers["agent_mode"] == "local-bare-metal":
        console.print(
            "\n  next: start the agent yourself "
            "(`uvicorn simd_agent.main:app --port 8000`), then `simd run …`."
        )
    else:
        console.print(
            f"\n  next: `simd run …` — talks to your remote agent at "
            f"{answers['agent_url']}."
        )
