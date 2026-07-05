"""Live trace stream for the debug/demo console.

The Rich HUD owns the main console, so anything we want to WATCH live --
what the model heard, planned, reasoned about, decided, and why a step is
stuck -- goes through this Tracer into a plain append-only file. A second
terminal (auto-opened by main.py, see `launch_debug_console`) tails it
with colors via `scripts/trace_view.py`.

Pure stdlib and always safe: a tracing failure must never touch the run,
so every write is wrapped and degrades to silence.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime
from pathlib import Path

# One trace line stays one line: reasoning blobs are collapsed and capped so
# the viewer's tail loop never has to reassemble multi-line records.
_MAX_MESSAGE_CHARS = 600
_WHITESPACE_RE = re.compile(r"\s+")


def _clean(message: str) -> str:
    """Collapses a message to a single capped line."""
    flat = _WHITESPACE_RE.sub(" ", str(message)).strip()
    if len(flat) > _MAX_MESSAGE_CHARS:
        flat = flat[: _MAX_MESSAGE_CHARS - 1] + "…"
    return flat


class Tracer:
    """Appends timestamped, tagged events to the trace stream file."""

    def __init__(self, path: str | Path) -> None:
        """Opens (truncating) the trace file for this session.

        Truncation is deliberate: the debug window narrates THIS run, not
        an ever-growing archive (continuum.log keeps the full history).

        Args:
            path: Filesystem path of the trace stream file.
        """
        self._path = Path(path)
        self._lock = threading.Lock()
        self._file = open(self._path, "w", encoding="utf-8")  # noqa: SIM115 - long-lived stream

    def event(self, tag: str, message: str) -> None:
        """Writes one trace line: `HH:MM:SS | TAG | message`.

        Args:
            tag: Short uppercase category (HEARD/PLAN/THINK/ACTION/...).
            message: Free-text detail; newlines collapsed, length capped.
        """
        try:
            line = f"{datetime.now().strftime('%H:%M:%S')} | {tag:<8} | {_clean(message)}\n"
            with self._lock:
                self._file.write(line)
                self._file.flush()
        except Exception:  # noqa: BLE001 - tracing must never hurt the run
            pass

    def close(self) -> None:
        """Closes the underlying file (best-effort)."""
        try:
            with self._lock:
                self._file.close()
        except Exception:  # noqa: BLE001
            pass


class TraceLogHandler(logging.Handler):
    """Bridges a stdlib logger into the trace stream.

    Attached to the `vision` logger in main.py so its INFO/WARNING lines
    (grounding escalation, Gemini retries, circuit breaker) appear in the
    debug window without any extra plumbing at the call sites.
    """

    def __init__(self, tracer: Tracer) -> None:
        super().__init__(level=logging.INFO)
        self._tracer = tracer

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102 - stdlib contract
        tag = "WARN" if record.levelno >= logging.WARNING else "MODEL"
        self._tracer.event(tag, record.getMessage())


class NullTracer:
    """No-op stand-in so call sites never need `if tracer is not None`."""

    def event(self, tag: str, message: str) -> None:  # noqa: D102
        return None

    def close(self) -> None:  # noqa: D102
        return None
