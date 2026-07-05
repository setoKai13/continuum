"""Colorized live viewer for the Continuum trace stream (debug/demo console).

Launched automatically in its own Terminal window by main.py (DEBUG_CONSOLE),
or by hand:

    .venv/bin/python scripts/trace_view.py [path/to/continuum-trace.log]

Tails the file from the top (main.py truncates it at boot, so the window
narrates exactly this session) and colors each event by its tag.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console  # noqa: E402
from rich.text import Text  # noqa: E402

TAG_STYLES = {
    "BOOT": "bold white on blue",
    "HEARD": "bold cyan",
    "PLAN": "bold magenta",
    "THINK": "yellow",
    "ACTION": "bold green",
    "ROUTE": "green",
    "VERIFY": "blue",
    "OVERRIDE": "bold magenta",
    "REFUSE": "bold red",
    "ERROR": "bold red",
    "WARN": "red",
    "MODEL": "yellow",
    "LOOP": "dim",
}


def follow(path: Path, console: Console) -> None:
    """Prints existing lines then follows the file forever (Ctrl+C to quit)."""
    console.print(f"[bold]Continuum trace[/bold] — {path} (Ctrl+C to quit)\n")
    while not path.exists():
        time.sleep(0.3)
    with open(path, "r", encoding="utf-8") as stream:
        while True:
            line = stream.readline()
            if not line:
                time.sleep(0.2)
                continue
            render(line.rstrip("\n"), console)


def render(line: str, console: Console) -> None:
    """Renders one `HH:MM:SS | TAG | message` line with its tag color.

    Built from styled Text spans, never markup: trace messages routinely
    contain brackets ("[turn 3]", boxes) that Rich markup would misparse.
    """
    parts = line.split(" | ", 2)
    if len(parts) != 3:
        console.print(Text(line, style="dim"))
        return
    timestamp, tag, message = parts
    style = TAG_STYLES.get(tag.strip(), "white")
    text = Text()
    text.append(f"{timestamp} ", style="dim")
    text.append(f"{tag.strip():<8} ", style=style)
    text.append(message)
    console.print(text)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("continuum-trace.log")
    try:
        follow(target, Console())
    except KeyboardInterrupt:
        pass
