"""Centralized runtime configuration for Continuum.

Every tunable value (API keys, hotkeys, language, storage paths, turn limits)
flows through a single pydantic `Settings` object loaded from environment
variables / a `.env` file. No module outside this file should call
`os.getenv` directly -- this keeps configuration auditable and testable
(the test suite overrides `Settings` fields via constructor kwargs instead
of mutating the process environment).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Goal sentinel for a task created before the operator has spoken: the first
# voice instruction replaces it (agent._maybe_plan checks against this exact
# value, main.parse_args uses it as the --goal default).
DEFAULT_GOAL = "Awaiting instructions"

# Resolve .env next to this file, not the CWD, so `python python-desktop/main.py`
# from anywhere still reads the right environment.
_ENV_FILE = Path(__file__).resolve().parent / ".env"


class Settings(BaseSettings):
    """Typed application settings, sourced from `.env` and env vars.

    Attributes:
        gemini_api_key: Required key for the Gemini vision/grounding calls.
        openrouter_api_key: Optional key for the Nemotron selector (bonus path).
        gradium_api_key: Optional key for the Gradium voice sponsor path.
        model_name: Gemini model used for vision grounding and per-turn
            screen judgement (Computer Use tier: fast, cheap, called every
            turn).
        planner_model_name: Optional stronger "mastermind" model used for
            the reasoning-heavy, once-per-instruction calls (decomposing an
            instruction into steps, extracting a voice-correction override).
            Empty (default) means model_name handles everything.
        planner_thinking_level: Gemini 3 thinking depth (minimal/low/medium/
            high) for the planner calls. Empty (default) leaves the model's
            dynamic default; only set on models that support thinking_level.
        ground_samples: CAP on grounding self-consistency samples. The model
            reports a confidence with each action: confident answers act on
            the first (fast) call, only low-confidence ones escalate to
            extra samples and a majority vote. 1 disables escalation.
        ground_confidence: Confidence threshold below which grounding
            escalates to extra samples (when ground_samples > 1).
        push_to_talk_key: Keyboard key held down to record voice input.
        kill_switch_key: Keyboard key that raises the kill-switch event.
        language: STT priming language code (e.g. "fr", "en").
        whisper_model: faster-whisper model size ("small" default; "base"
            trades accuracy for speed on demo day).
        keep_alive: When True (default), finishing every step does not end
            the live run: the agent announces completion, keeps listening,
            and the next spoken instruction starts a fresh plan in the same
            session. Set to False to exit as soon as the task completes.
        db_path: Filesystem path to the SQLite hold-state database.
        log_path: File the process logs to (the console belongs to the HUD).
        trace_path: File the live trace stream is written to (truncated at
            each boot; the debug console window tails it).
        debug_console: When True (default), main.py opens a second Terminal
            window at launch showing the live trace: what the model heard,
            planned, reasoned, decided, and every loop event.
        log_level: Logging level name (DEBUG/INFO/WARNING/ERROR).
        max_turns: Ceiling on WORK turns (turns that carry an observation);
            idle waiting does not consume this budget.
        max_idle_turns: Ceiling on CONSECUTIVE turns with nothing to observe
            before the run ends on its own (state persists either way).
        loop_idle_sleep_s: Sleep between empty OBSERVE polls, so waiting for
            voice does not busy-spin the CPU.
        max_step_attempts: Actions attempted on one step before it is marked
            blocked and the loop moves on.
        max_context_screenshots: How many recent screenshots stay in context.
        pyautogui_pause_s: Delay pyautogui inserts between actions.
        ui_settle_s: Pause after each real action so the next screenshot is
            not taken mid-animation.
        gemini_timeout_ms: Per-request HTTP timeout for the per-turn Gemini
            calls (grounding): tight, they run every turn.
        planner_timeout_ms: Per-request timeout for the mastermind calls
            (planning, override extraction): generous, they run once per
            instruction and a thinking pro model can exceed 30s (504
            DEADLINE_EXCEEDED otherwise).
        gemini_max_attempts: Attempts per Gemini call (transient errors only).
        gemini_backoff_s: Base backoff between Gemini retries (linear).
        gemini_breaker_failures: Consecutive failed calls before the circuit
            breaker opens.
        gemini_breaker_cooldown_s: How long an open breaker rejects calls.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    gradium_api_key: str | None = Field(default=None, alias="GRADIUM_API_KEY")

    model_name: str = Field(default="gemini-3.5-flash", alias="MODEL_NAME")
    planner_model_name: str = Field(default="", alias="PLANNER_MODEL")
    planner_thinking_level: str = Field(default="", alias="PLANNER_THINKING_LEVEL")
    ground_samples: int = Field(default=1, alias="GROUND_SAMPLES")
    ground_confidence: float = Field(default=0.75, alias="GROUND_CONFIDENCE")

    push_to_talk_key: str = Field(default="f8", alias="PTT_KEY")
    kill_switch_key: str = Field(default="esc", alias="KILL_KEY")
    language: str = Field(default="fr", alias="LANGUAGE")
    whisper_model: str = Field(default="small", alias="WHISPER_MODEL")

    db_path: str = Field(default="continuum.db", alias="DB_PATH")
    log_path: str = Field(default="continuum.log", alias="LOG_PATH")
    trace_path: str = Field(default="continuum-trace.log", alias="TRACE_PATH")
    debug_console: bool = Field(default=True, alias="DEBUG_CONSOLE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    keep_alive: bool = Field(default=True, alias="KEEP_ALIVE")

    max_turns: int = Field(default=120, alias="MAX_TURNS")
    max_idle_turns: int = Field(default=1500, alias="MAX_IDLE_TURNS")
    loop_idle_sleep_s: float = Field(default=0.4, alias="LOOP_IDLE_SLEEP_S")
    max_step_attempts: int = Field(default=3, alias="MAX_STEP_ATTEMPTS")
    max_context_screenshots: int = Field(default=3, alias="MAX_CONTEXT_SCREENSHOTS")

    pyautogui_pause_s: float = Field(default=0.1, alias="PYAUTOGUI_PAUSE_S")
    ui_settle_s: float = Field(default=0.8, alias="UI_SETTLE_S")

    gemini_timeout_ms: int = Field(default=30_000, alias="GEMINI_TIMEOUT_MS")
    planner_timeout_ms: int = Field(default=120_000, alias="PLANNER_TIMEOUT_MS")
    gemini_max_attempts: int = Field(default=3, alias="GEMINI_MAX_ATTEMPTS")
    gemini_backoff_s: float = Field(default=0.5, alias="GEMINI_BACKOFF_S")
    gemini_breaker_failures: int = Field(default=4, alias="GEMINI_BREAKER_FAILURES")
    gemini_breaker_cooldown_s: float = Field(default=20.0, alias="GEMINI_BREAKER_COOLDOWN_S")


class MissingApiKeyError(Exception):
    """Raised when a live call requires a real API key that is absent."""


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Loads and memoizes application settings for the process lifetime.

    Returns:
        The cached `Settings` instance.
    """
    return Settings()


def is_placeholder_key(value: str | None) -> bool:
    """Detects the bootstrap placeholder so live calls can refuse early.

    Args:
        value: The candidate API key string.

    Returns:
        True if the value looks like the bootstrap placeholder or is empty.
    """
    if not value:
        return True
    return "REPLACE_WITH_REAL_KEY" in value or "TODO" in value.upper()
