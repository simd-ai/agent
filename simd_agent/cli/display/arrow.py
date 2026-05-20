"""Arrow-key menu for interactive CLI prompts.

Mirrors the bash implementation in ``install.sh`` — ↑/↓ to move,
Enter to confirm, ``q`` or Ctrl-C to cancel, ``1``–``9`` for
direct hotkeys, vim's ``j``/``k`` also work.  No external deps;
uses ``termios`` for raw stdin and ANSI escapes for cursor
control.

The visual style (bold cyan caret + bold selected text + dim
help hint) deliberately matches the bash wizard so the two
flows feel like one tool.
"""

from __future__ import annotations

import sys
from typing import Sequence

from simd_agent.cli.display import console


__all__ = ["arrow_choice"]


def arrow_choice(prompt: str, options: Sequence[str]) -> int:
    """Show an arrow-key menu, return the 0-based index of the choice.

    Raises ``KeyboardInterrupt`` if the user cancels (``q`` or Ctrl-C).
    Falls back to plain ``input()`` numbered selection on non-TTY
    stdin so the function still works in pipes / CI.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return _numbered_fallback(prompt, options)

    # Import here so non-TTY paths don't require termios (Windows
    # users see the fallback instead of an ImportError).
    import termios
    import tty

    n = len(options)
    selected = 0

    console.print(f"[bold]{prompt}[/]")
    console.print("[dim]  (↑/↓ to move, Enter to select, q to quit)[/]")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        sys.stdout.write("\x1b[?25l")  # hide cursor
        sys.stdout.flush()

        _draw(options, selected)
        tty.setraw(fd)

        while True:
            ch = sys.stdin.read(1)

            if ch == "\x1b":
                # ESC sequence — could be arrow key or just ESC alone.
                # Read two more chars; if they don't form an arrow,
                # ignore (e.g. lone ESC).
                rest = sys.stdin.read(2)
                if rest == "[A":
                    selected = (selected - 1) % n
                elif rest == "[B":
                    selected = (selected + 1) % n
                # other escape sequences: ignore
            elif ch in ("\r", "\n"):
                break
            elif ch == "k":
                selected = (selected - 1) % n
            elif ch == "j":
                selected = (selected + 1) % n
            elif ch.isdigit() and 1 <= int(ch) <= n:
                selected = int(ch) - 1
                break
            elif ch in ("q", "\x03"):  # q or Ctrl-C
                raise KeyboardInterrupt()
            else:
                continue  # don't redraw on no-op keys

            # Move cursor back to the top of the menu and redraw.
            sys.stdout.write(f"\x1b[{n}A")
            sys.stdout.flush()
            _draw(options, selected)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\x1b[?25h")  # show cursor
        sys.stdout.flush()

    return selected


def _draw(options: Sequence[str], selected: int) -> None:
    for i, opt in enumerate(options):
        # \x1b[2K clears the current line so longer-then-shorter
        # strings don't leave trailing garbage on re-draw.
        sys.stdout.write("\x1b[2K")
        if i == selected:
            console.print(f"  [bold cyan]❯[/] [bold]{opt}[/]")
        else:
            console.print(f"    {opt}")


def _numbered_fallback(prompt: str, options: Sequence[str]) -> int:
    """Plain numbered prompt for non-TTY contexts (CI, pipes)."""
    console.print(f"[bold]{prompt}[/]")
    for i, opt in enumerate(options, start=1):
        console.print(f"  {i}) {opt}")
    while True:
        try:
            raw = input("  > ").strip()
        except EOFError:
            raise KeyboardInterrupt()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        console.print(f"  unknown choice {raw!r}")
