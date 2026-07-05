"""CERVEAU (Route 1): official Gemini Computer Use via the Interactions API.

This is the DeepMind-track path: instead of re-deriving a click from a
`generate_content` box (that is `vision.py`, Route 2), it literally calls
`client.interactions.create` with a `computer_use` tool bound to
`ENVIRONMENT_DESKTOP`. The model drives the desktop by emitting predefined
function calls (`click`, `type`, `hotkey`, `scroll`, ...), each carrying
0..CU_NORM_MAX normalized `{x, y}` POINTS (not the 0-1000 boxes Route 2
returns). We execute exactly one such call per turn, then feed the resulting
screenshot back as a `function_result`, reusing the same interaction as a
session (`previous_interaction_id`).

`InteractionsComputerUse.ground(task, step, screenshot)` mirrors
`GeminiVision.ground`'s contract, except it returns a `CuAction | None`:
`None` means "stall this turn" (unsupported call, safety refusal, or any
failure). Like `vision.py`, a failure NEVER raises out of a turn -- the loop
keeps living. `google-genai` is imported lazily so the module stays
importable without the SDK or a real key.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Any

from config import MissingApiKeyError, Settings, is_placeholder_key
from state import Step, TaskState
from vision import _is_transient_error as is_transient_error  # shared retry classifier

logger = logging.getLogger(__name__)

# CU predefined function name -> our ActionPlan kind. Several aliases per kind
# so the mapping survives naming differences between the desktop and browser
# environments (e.g. `click` vs `click_at`, `hotkey` vs `key_combination`):
# whichever the live API emits, we still recognize it. Anything not listed
# here (double_click, right_click, move, mouse_down, wait, ...) is left
# UNSUPPORTED on purpose -> the turn stalls rather than mis-acting.
_CLICK_FUNCTIONS = frozenset({"click", "click_at", "left_click"})
_TYPE_FUNCTIONS = frozenset({"type", "type_text_at", "type_text"})
_HOTKEY_FUNCTIONS = frozenset({"hotkey", "key_combination"})
_PRESS_KEY_FUNCTIONS = frozenset({"press_key", "key_press"})
_SCROLL_FUNCTIONS = frozenset({"scroll", "scroll_document", "scroll_at"})

# CU key token -> pyautogui key name (lower-cased first). Single characters and
# already-correct names pass straight through, so this only patches the aliases.
_KEY_ALIASES = {
    "return": "enter",
    "cmd": "command",
    "control": "ctrl",
    "opt": "option",
    "escape": "esc",
    "del": "delete",
}

_SYSTEM_INSTRUCTION = (
    "You control a real macOS desktop through the Computer Use tool. Each turn "
    "you receive the current screen and the single step to progress; respond "
    "with exactly one predefined desktop action (click, type, hotkey, scroll) "
    "that makes real progress on that step, or stop if it is already done. "
    "Never attempt destructive or irreversible actions."
)


class InteractionsError(Exception):
    """Raised internally when an Interactions CU call cannot be completed."""


@dataclass
class CuAction:
    """One decision from the Computer Use brain for the current step.

    Attributes:
        kind: "click" | "type" | "hotkey" | "scroll" | "done" | "noop".
            "done" means the model issued no function call (the step looks
            satisfied); the caller marks the step complete.
        point: Normalized (x, y) target in 0..CU_NORM_MAX, for a click.
        text: Text to type, for a "type" action.
        keys: Normalized pyautogui key chord, for a "hotkey" action.
        amount: Signed wheel clicks (positive = up), for a "scroll" action.
        reasoning: The model's `intent` string (for the HUD/log).
    """

    kind: str
    point: tuple[float, float] | None = None
    text: str | None = None
    keys: list[str] | None = None
    amount: int | None = None
    reasoning: str | None = None


class InteractionsComputerUse:
    """Drives ENVIRONMENT_DESKTOP Computer Use through `interactions.create`.

    The session (one `interaction.id` reused via `previous_interaction_id`) is
    created on the first call and continued on every later turn, so the model
    keeps its own view of what it has already done. The id is exposed via
    `session_id` for persistence and reseeded on resume via `restore_session`,
    which lets `--resume` reattach to the live session (or renew cleanly if it
    has expired). Hardened exactly like `vision.py`: explicit timeout, retry on
    transient errors only, and a consecutive-failure circuit breaker.
    """

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        """Stores settings; does not build the genai client yet.

        Args:
            settings: Application settings (model, environment, timeouts,
                breaker knobs -- shared with the grounding path).
            client: Optional pre-built client (tests inject a fake exposing
                `interactions.create`); None builds a real `genai.Client`
                lazily on first use.
        """
        self._settings = settings
        self._client = client
        self._interaction_id: str | None = None
        self._pending_call: dict[str, str] | None = None
        self._step_id: str | None = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    @property
    def session_id(self) -> str | None:
        """The current interaction id (the `previous_interaction_id` handle)."""
        return self._interaction_id

    def restore_session(self, session_id: str | None) -> None:
        """Reseeds the session from a persisted id (used by `--resume`).

        Args:
            session_id: The `interactions_session_id` reloaded from TaskState,
                or None to start a fresh session.
        """
        self._interaction_id = session_id or None
        self._pending_call = None

    def ground(self, task: TaskState, step: Step, screenshot: Any) -> CuAction | None:
        """Grounds ONE desktop action for `step`, or None to stall the turn.

        Args:
            task: The current hold-state (its goal frames the instruction).
            step: The step selected by `TaskState.next_actionable_step()`.
            screenshot: A PIL image / bytes of the current screen.

        Returns:
            A `CuAction` to execute, or None to stall (unsupported call,
            safety refusal, or any failure). Never raises out of a turn.
        """
        try:
            return self._ground(task, step, screenshot)
        except Exception as error:  # noqa: BLE001 - a turn degrades to a stall, never crashes
            logger.warning("CU grounding failed, stalling this turn: %s", error)
            return None

    def _ground(self, task: TaskState, step: Step, screenshot: Any) -> CuAction | None:
        """The real grounding body (may raise; `ground` is the safe wrapper)."""
        image_b64 = _encode_screenshot(screenshot)
        if image_b64 is None:
            return None

        step_changed = step.id != self._step_id
        self._step_id = step.id
        continuing = bool(self._interaction_id and self._pending_call and not step_changed)
        if continuing:
            input_ = [self._function_result(self._pending_call, image_b64)]
        else:
            input_ = [_user_turn(_instruction(task, step), image_b64)]
            self._pending_call = None

        try:
            response = self._create(input_, self._interaction_id)
        except InteractionsError as error:
            if self._interaction_id is not None and _looks_expired(error):
                return self._renew(task, step, image_b64, error)
            logger.warning("CU call failed: %s", error)
            return None

        self._interaction_id = _field(response, "id") or self._interaction_id
        return self._interpret(response)

    def _renew(self, task: TaskState, step: Step, image_b64: str, cause: Exception) -> CuAction | None:
        """Starts a fresh session after the previous interaction id expired.

        Args:
            task: The current hold-state.
            step: The step being grounded.
            image_b64: The already-encoded current screenshot.
            cause: The error that flagged the session as expired (for the log).

        Returns:
            A `CuAction` from the renewed session, or None on failure.
        """
        logger.info("CU session %s expired (%s); renewing", self._interaction_id, cause)
        self._interaction_id = None
        self._pending_call = None
        try:
            response = self._create([_user_turn(_instruction(task, step), image_b64)], None)
        except InteractionsError as error:
            logger.warning("CU session renewal failed: %s", error)
            return None
        self._interaction_id = _field(response, "id")
        return self._interpret(response)

    def _interpret(self, response: Any) -> CuAction | None:
        """Turns an Interaction response into a CuAction (or None to stall)."""
        function_call = _first_function_call(response)
        if function_call is None:
            # No action requested: the model judged the step satisfied (it
            # replies with output_text instead of a function call).
            self._pending_call = None
            return CuAction(kind="done", reasoning=_field(response, "output_text"))

        name = str(_field(function_call, "name") or "")
        arguments = _field(function_call, "arguments")
        arguments = arguments if isinstance(arguments, dict) else {}
        call_id = str(_field(function_call, "id") or "")

        if _is_safety_confirmation(arguments):
            logger.warning("CU safety confirmation requested for %r -> refusing", name)
            self._pending_call = None
            return None

        action = _map_function(name, arguments, self._settings)
        if action is None:
            logger.info("CU function %r unsupported -> stall", name)
            self._pending_call = None
            return None

        self._pending_call = {"call_id": call_id, "name": name}
        return action

    def _create(self, input_: list[Any], previous_interaction_id: str | None) -> Any:
        """Calls `interactions.create` with a timeout, retry and breaker.

        Mirrors `vision._generate`: one explicit per-request timeout, retries
        only on transient errors (linear backoff), and a consecutive-failure
        breaker so a dead endpoint fails fast instead of hanging the loop.

        Args:
            input_: The `input` payload (a user Turn or a function_result Step).
            previous_interaction_id: The session id to continue, or None to
                open a new session.

        Returns:
            The raw `Interaction` response.

        Raises:
            InteractionsError: If the breaker is open or every attempt failed.
            MissingApiKeyError: If no real API key is configured.
        """
        if self._circuit_open_until and time.monotonic() < self._circuit_open_until:
            raise InteractionsError("Computer Use circuit breaker is open; skipping call")

        client = self._ensure_client()
        kwargs = self._request_kwargs(input_, previous_interaction_id)

        last_error: Exception | None = None
        for attempt in range(1, self._settings.gemini_max_attempts + 1):
            try:
                response = client.interactions.create(**kwargs)
                self._consecutive_failures = 0
                return response
            except Exception as error:  # noqa: BLE001 - normalized into InteractionsError
                last_error = error
                if not is_transient_error(error):
                    logger.warning("CU call failed (non-transient, no retry): %s", error)
                    break
                logger.warning(
                    "CU call attempt %d/%d failed: %s",
                    attempt, self._settings.gemini_max_attempts, error,
                )
                if attempt < self._settings.gemini_max_attempts:
                    time.sleep(self._settings.gemini_backoff_s * attempt)

        self._trip_breaker()
        # Chain the cause so `_looks_expired` can still read its status code
        # through the wrapper and decide whether to renew the session.
        raise InteractionsError(f"CU call failed: {last_error}") from last_error

    def _trip_breaker(self) -> None:
        """Counts a failed call and opens the breaker past the threshold."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._settings.gemini_breaker_failures:
            self._circuit_open_until = time.monotonic() + self._settings.gemini_breaker_cooldown_s
            logger.error(
                "CU circuit breaker OPEN for %.0fs after %d consecutive failures",
                self._settings.gemini_breaker_cooldown_s, self._consecutive_failures,
            )

    def _request_kwargs(self, input_: list[Any], previous_interaction_id: str | None) -> dict[str, Any]:
        """Builds the `interactions.create` keyword arguments for one call."""
        kwargs: dict[str, Any] = {
            "model": self._settings.cu_model_name or self._settings.model_name,
            "input": input_,
            "tools": [self._tool()],
            "system_instruction": _SYSTEM_INSTRUCTION,
            "store": True,  # required so previous_interaction_id resolves later
            "timeout": self._settings.cu_timeout_ms / 1000.0,
        }
        if previous_interaction_id:
            kwargs["previous_interaction_id"] = previous_interaction_id
        return kwargs

    def _tool(self) -> dict[str, Any]:
        """Builds the `computer_use` tool declaration from settings."""
        return {
            "type": "computer_use",
            "environment": self._settings.cu_environment,
            "enable_prompt_injection_detection": self._settings.cu_prompt_injection_detection,
        }

    def _function_result(self, pending: dict[str, str], image_b64: str) -> dict[str, Any]:
        """Builds the function_result step feeding back the executed action's screen."""
        return {
            "type": "function_result",
            "call_id": pending["call_id"],
            "name": pending["name"],
            "result": [
                {"type": "text", "text": "Action executed. Current screen attached."},
                {"type": "image", "data": image_b64, "mime_type": "image/png"},
            ],
        }

    def _ensure_client(self) -> Any:
        """Lazily builds the `google.genai` client, refusing placeholder keys.

        Returns:
            A constructed `genai.Client` (or the injected test client).

        Raises:
            MissingApiKeyError: If the configured key is empty or a placeholder.
        """
        if self._client is not None:
            return self._client
        if is_placeholder_key(self._settings.gemini_api_key):
            raise MissingApiKeyError(
                "GEMINI_API_KEY is missing or a placeholder. "
                "Replace it in .env before a live CU run (see RUNBOOK.md)."
            )
        from google import genai  # lazy: network/SDK dependency

        self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client


def _encode_screenshot(screenshot: Any) -> str | None:
    """Encodes a screenshot to base64 PNG for the CU image content.

    Args:
        screenshot: A PIL image (has `.save`), raw bytes, or a token (tests).

    Returns:
        The base64-encoded PNG string, or None if encoding failed (the turn
        then stalls rather than crashing).
    """
    try:
        if screenshot is None:
            return None
        if hasattr(screenshot, "save"):  # PIL.Image
            buffer = io.BytesIO()
            screenshot.save(buffer, format="PNG")
            raw = buffer.getvalue()
        elif isinstance(screenshot, (bytes, bytearray)):
            raw = bytes(screenshot)
        else:
            raw = str(screenshot).encode("utf-8")
        return base64.b64encode(raw).decode("ascii")
    except Exception as error:  # noqa: BLE001 - encoding must never crash a turn
        logger.warning("could not encode screenshot for CU: %s", error)
        return None


def _user_turn(text: str, image_b64: str) -> dict[str, Any]:
    """Builds a user Turn carrying the instruction text and the screenshot.

    A Turn (role + content) is used rather than a bare content list because
    the SDK's input union resolves a bare `[{type:text},{type:image}]` to
    UNKNOWN steps; wrapping them in a Turn serializes them correctly.
    """
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image", "data": image_b64, "mime_type": "image/png"},
        ],
    }


def _instruction(task: TaskState, step: Step) -> str:
    """Builds the per-step instruction text sent alongside the screenshot."""
    history = "; ".join(step.history[-5:]) or "none yet"
    return (
        f"Goal: {task.goal}\n"
        f"Current step: {step.desc}\n"
        f"Notes so far: {step.note or 'none'}\n"
        f"Actions ALREADY executed for this step: {history}\n"
        "Perform the SINGLE next desktop action to progress THIS step on the "
        "current screen. If the step is already satisfied, do not act."
    )


def _map_function(name: str, arguments: dict[str, Any], settings: Settings) -> CuAction | None:
    """Maps a CU predefined function call onto a CuAction, or None if unsupported.

    Args:
        name: The predefined function name returned by the model.
        arguments: Its argument dict (coordinates, text, keys, ...).
        settings: Application settings (scroll conversion knob).

    Returns:
        A `CuAction` for a supported click/type/hotkey/scroll, or None so the
        caller stalls on anything else (double_click, move, wait, ...).
    """
    key = name.lower()
    intent = str(arguments.get("intent") or "").strip() or None

    if key in _CLICK_FUNCTIONS:
        point = _as_point(arguments.get("x"), arguments.get("y"))
        return CuAction(kind="click", point=point, reasoning=intent) if point else None
    if key in _TYPE_FUNCTIONS:
        text = arguments.get("text")
        if isinstance(text, str) and text:
            return CuAction(kind="type", text=text, reasoning=intent)
        return None
    if key in _HOTKEY_FUNCTIONS:
        keys = _normalize_keys(arguments.get("keys"))
        return CuAction(kind="hotkey", keys=keys, reasoning=intent) if keys else None
    if key in _PRESS_KEY_FUNCTIONS:
        keys = _normalize_keys([arguments.get("key")])
        return CuAction(kind="hotkey", keys=keys, reasoning=intent) if keys else None
    if key in _SCROLL_FUNCTIONS:
        amount = _scroll_amount(arguments, settings)
        return CuAction(kind="scroll", amount=amount, reasoning=intent) if amount is not None else None
    return None


def _as_point(x: Any, y: Any) -> tuple[float, float] | None:
    """Parses (x, y) coordinate arguments into a float point, or None."""
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def _normalize_keys(raw: Any) -> list[str]:
    """Normalizes CU key tokens to pyautogui key names (empty list if none)."""
    if not isinstance(raw, (list, tuple)):
        return []
    keys: list[str] = []
    for token in raw:
        text = (str(token) if token is not None else "").strip().lower()
        if not text:
            continue
        keys.append(_KEY_ALIASES.get(text, text))
    return keys


def _scroll_amount(arguments: dict[str, Any], settings: Settings) -> int | None:
    """Converts a CU scroll (direction + magnitude) into signed wheel clicks.

    Args:
        arguments: The scroll call's arguments (direction, magnitude_in_pixels).
        settings: Application settings (cu_scroll_pixels_per_click divisor).

    Returns:
        Signed wheel clicks (positive = up), or None for an unrecognized
        direction (horizontal scroll has no pyautogui vertical equivalent).
    """
    direction = str(arguments.get("direction") or "").lower()
    if direction in ("up", "haut"):
        sign = 1
    elif direction in ("down", "bas"):
        sign = -1
    else:
        return None
    try:
        pixels = float(arguments.get("magnitude_in_pixels"))
    except (TypeError, ValueError):
        pixels = float(settings.cu_scroll_pixels_per_click)
    clicks = max(1, round(pixels / settings.cu_scroll_pixels_per_click))
    return sign * clicks


def _is_safety_confirmation(arguments: dict[str, Any]) -> bool:
    """True if the function call asks for a safety confirmation (we refuse)."""
    decision = arguments.get("safety_decision")
    if isinstance(decision, dict):
        return str(decision.get("decision") or "").lower() == "require_confirmation"
    return False


def _first_function_call(response: Any) -> Any | None:
    """Returns the first `function_call` step of an Interaction, or None."""
    steps = _field(response, "steps") or []
    for step in steps:
        if _field(step, "type") == "function_call":
            return step
    return None


def _looks_expired(error: Exception) -> bool:
    """Heuristic: does the error mean the previous interaction id is gone?

    Inspects both the raised `InteractionsError` and its chained cause (the
    original SDK error carries the status code), so a 404 on a continuation
    is recognized through the wrapper.

    Args:
        error: The failure from `_create` (already retry-exhausted).

    Returns:
        True for a 404 / not-found / stale-session shape, so the caller can
        renew the session cleanly instead of stalling forever on a dead id.
    """
    marks = ("not found", "previous_interaction", "expired", "does not exist")
    for candidate in (error, getattr(error, "__cause__", None)):
        if candidate is None:
            continue
        code = getattr(candidate, "code", None)
        if not isinstance(code, int):
            code = getattr(candidate, "status_code", None)
        if code == 404:
            return True
        if any(mark in str(candidate).lower() for mark in marks):
            return True
    return False


def _field(obj: Any, key: str) -> Any:
    """Reads `key` from a pydantic object or a plain dict (SDK or test fake)."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
