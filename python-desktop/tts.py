"""Text-to-speech: macOS `say` first, `pyttsx3` as a portable fallback.

`say` is a subprocess call (stdlib `subprocess`, always safe to import);
`pyttsx3` is imported lazily since it is only needed on the fallback path
(e.g. running the agent loop on a non-macOS box during development).
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# `say` speaks a short confirmation in ~1-2s; the timeout only guards a hung
# child process, never a normal sentence.
_SAY_TIMEOUT_S = 30.0


class Speaker:
    """Speaks text aloud, preferring the native macOS `say` command."""

    def __init__(self) -> None:
        """Prepares a speaker; the pyttsx3 engine loads lazily if ever needed."""
        self._fallback_engine: Any | None = None

    def say(self, text: str) -> None:
        """Speaks `text`, trying `say` first and falling back to pyttsx3.

        Speech is a nicety: if BOTH paths fail (no `say` binary and no
        pyttsx3 installed), the failure is logged and swallowed -- a missing
        voice must never crash the agent loop mid-turn.

        Args:
            text: The sentence to speak. A falsy value is a no-op.
        """
        if not text:
            return
        try:
            subprocess.run(["say", text], check=True, timeout=_SAY_TIMEOUT_S)
            return
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            self._say_fallback(text)
        except Exception:  # noqa: BLE001 - voice is optional, the loop is not
            logger.warning("TTS unavailable (`say` failed and pyttsx3 fallback missing)")

    def _say_fallback(self, text: str) -> None:
        """Speaks via pyttsx3 when the native `say` binary is unavailable.

        Args:
            text: The sentence to speak.
        """
        import pyttsx3  # lazy: only needed off macOS or if `say` is missing

        if self._fallback_engine is None:
            self._fallback_engine = pyttsx3.init()
        self._fallback_engine.say(text)
        self._fallback_engine.runAndWait()
