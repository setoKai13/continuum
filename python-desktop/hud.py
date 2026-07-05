"""Headless-UI HUD: two Rich terminal panels, no graphical window.

Left panel renders the evolving `TaskState` (the visible proof of
hold-state); right panel streams tool calls as they happen. The real Mac
moving on screen is the actual action surface -- this HUD is a live window
into *why* it moves.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_STATUS_MARKERS = {"done": "[x]", "doing": "[~]", "blocked": "[!]", "todo": "[ ]"}


class Hud:
    """Rich `Live` dashboard: TaskState panel + tool-call stream panel.

    Usage:
        with Hud() as hud:
            hud.update(task.render(), "turn 1: acted click on s1")
    """

    def __init__(self, max_log_lines: int = 24) -> None:
        """Builds the two-panel layout without starting the live render yet.

        Args:
            max_log_lines: How many of the most recent log lines to keep
                visible in the right-hand panel.
        """
        self._console = Console()
        self._log: deque[str] = deque(maxlen=max_log_lines)
        self._layout = Layout()
        self._layout.split_row(Layout(name="state"), Layout(name="log"))
        self._live: Live | None = None

    def __enter__(self) -> "Hud":
        self._live = Live(self._layout, console=self._console, refresh_per_second=8)
        self._live.__enter__()
        self._render(None)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._live is not None:
            self._live.__exit__(*exc_info)
            self._live = None

    def update(self, task_snapshot: dict[str, Any] | None, log_line: str) -> None:
        """Pushes one event line and redraws both panels.

        Args:
            task_snapshot: The result of `TaskState.render()`, or None
                before any task is loaded.
            log_line: A short human-readable description of what just happened.
        """
        self._log.append(log_line)
        self._render(task_snapshot)

    def _render(self, snapshot: dict[str, Any] | None) -> None:
        self._layout["state"].update(
            Panel(self._render_state_table(snapshot), title="TaskState (hold-state)")
        )
        self._layout["log"].update(Panel(Text("\n".join(self._log)), title="Tool calls"))
        if self._live is not None:
            self._live.refresh()

    def _render_state_table(self, snapshot: dict[str, Any] | None) -> Table:
        table = Table(show_header=False, expand=True, pad_edge=False)
        if not snapshot:
            table.add_row("(no task loaded -- ready)")
            return table

        table.add_row("task_id", str(snapshot.get("task_id", "")))
        table.add_row("goal", str(snapshot.get("goal", "")))
        table.add_row("status", str(snapshot.get("status", "")))
        table.add_row("session_count", str(snapshot.get("session_count", "")))
        table.add_row("progress", str(snapshot.get("progress", "")))

        for step in snapshot.get("steps", []):
            marker = _STATUS_MARKERS.get(step.get("status", "todo"), "[ ]")
            note = f" -- {step['note']}" if step.get("note") else ""
            table.add_row(f"{marker} {step.get('id', '')}", f"{step.get('desc', '')}{note}")

        for override in snapshot.get("overrides", []):
            applied = "applied" if override.get("applied") else "no-op"
            table.add_row("override", f"{override.get('when')} -> {override.get('rule')} ({applied})")

        return table
