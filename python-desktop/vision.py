"""CERVEAU: Gemini vision grounding -- screenshot -> action + normalized target.

Reimplements the "vision grounding, not the browser Computer Use tool"
path documented in CONNECTEURS.md: we send a screenshot plus the current
step to `generate_content`, and parse a normalized `[ymin, xmin, ymax,
xmax]` box (or explicit x/y) out of the response, which `mac_control.py`
then denormalizes and clicks.

`google-genai` is imported lazily so this module (and everything that
imports it) stays importable without the package installed -- only calling
`GeminiVision.ground()` requires it, and only with a real (non-placeholder)
API key.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from config import MissingApiKeyError, Settings, is_placeholder_key
from state import Step, TaskState

logger = logging.getLogger(__name__)

_BOX_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")

# One cap for the planner, used in BOTH the prompt and the parser so the two
# can never drift apart.
_MAX_PLAN_STEPS = 6


class VisionError(Exception):
    """Raised when Gemini grounding fails or returns an unparseable result."""


@dataclass
class GroundedAction:
    """One grounded decision returned by the vision brain for the current step.

    Attributes:
        kind: "click" | "type" | "scroll" | "hotkey" | "noop".
        box: Normalized [ymin, xmin, ymax, xmax] target, if any (0..1000).
        text: Text to type, app/url to open, or keys to send, if any.
        keys: Hotkey chord (e.g. ["command", "v"]), if any.
        amount: Scroll wheel clicks (positive = up), if any.
        reasoning: The model's free-text rationale (for the HUD/log).
        confidence: The model's self-reported probability (0..1) that this
            action is right; None when the model omitted it. Drives the
            adaptive escalation in `ground()`.
    """

    kind: str
    box: tuple[float, float, float, float] | None = None
    text: str | None = None
    keys: list[str] | None = None
    amount: int | None = None
    reasoning: str | None = None
    confidence: float | None = None


class GeminiVision:
    """Wraps `google-genai` vision calls for desktop grounding.

    Keeps only the last N screenshots in context (token budget), matching
    the "3 derniers tours" constraint from the spec.
    """

    def __init__(self, settings: Settings) -> None:
        """Stores settings; does not import/construct the genai client yet.

        Args:
            settings: Application settings (model name, API key, context budget).
        """
        self._settings = settings
        self._client = None
        self._recent_screenshots: list[Any] = []
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _ensure_client(self) -> Any:
        """Lazily builds the `google.genai` client, refusing placeholder keys.

        Returns:
            A constructed `genai.Client`.

        Raises:
            MissingApiKeyError: If the configured key is empty or still the
                bootstrap placeholder.
        """
        if self._client is not None:
            return self._client
        if is_placeholder_key(self._settings.gemini_api_key):
            raise MissingApiKeyError(
                "GEMINI_API_KEY is missing or a placeholder. "
                "Replace it in .env before a live run (see RUNBOOK.md, TODO(tom))."
            )
        from google import genai  # lazy: network/SDK dependency

        self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    def _remember(self, screenshot: Any) -> None:
        """Appends a screenshot to context, trimming to the configured budget."""
        self._recent_screenshots.append(screenshot)
        budget = self._settings.max_context_screenshots
        if len(self._recent_screenshots) > budget:
            self._recent_screenshots = self._recent_screenshots[-budget:]

    def _planner_model(self) -> str:
        """Returns the mastermind model for reasoning-heavy calls.

        Falls back to the grounding model when PLANNER_MODEL is unset, so a
        single-model configuration keeps working unchanged.
        """
        return self._settings.planner_model_name or self._settings.model_name

    def _generate(
        self, contents: list[Any], model: str | None = None, thinking_level: str | None = None
    ) -> Any:
        """Calls Gemini `generate_content` with a timeout, retry and breaker.

        This is the single outbound network dependency the agent hits every
        turn, so it honors the project rule (timeout + retry + circuit
        breaker): each attempt carries an explicit timeout, transient failures
        are retried with linear backoff, and after too many consecutive
        failures the breaker trips so a dead endpoint fails fast instead of
        hanging the loop.

        Args:
            contents: The `generate_content` contents (screenshots + prompt).
            model: Model override for this call (the planner/mastermind
                calls pass one); None uses the grounding `model_name`.
            thinking_level: Optional Gemini 3 thinking depth for this call
                (the planner calls pass the configured level); None leaves
                the model's dynamic default.

        Returns:
            The raw SDK response.

        Raises:
            VisionError: If the breaker is open or every attempt failed.
            MissingApiKeyError: If no real API key is configured.
        """
        import time

        if self._circuit_open_until and time.monotonic() < self._circuit_open_until:
            raise VisionError("Gemini circuit breaker is open; skipping call")

        client = self._ensure_client()
        config = self._generate_config(thinking_level)
        extra = {"config": config} if config is not None else {}

        max_attempts = self._settings.gemini_max_attempts
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=model or self._settings.model_name, contents=contents, **extra
                )
                self._consecutive_failures = 0
                return response
            except Exception as error:  # noqa: BLE001 - normalized into VisionError
                last_error = error
                if not _is_transient_error(error):
                    logger.warning("Gemini call failed (non-transient, no retry): %s", error)
                    break
                logger.warning("Gemini call attempt %d/%d failed: %s", attempt, max_attempts, error)
                if attempt < max_attempts:
                    time.sleep(self._settings.gemini_backoff_s * attempt)

        self._consecutive_failures += 1
        if self._consecutive_failures >= self._settings.gemini_breaker_failures:
            self._circuit_open_until = time.monotonic() + self._settings.gemini_breaker_cooldown_s
            logger.error(
                "Gemini circuit breaker OPEN for %.0fs after %d consecutive failures",
                self._settings.gemini_breaker_cooldown_s,
                self._consecutive_failures,
            )
        raise VisionError(f"Gemini call failed: {last_error}")

    def _generate_config(self, thinking_level: str | None = None) -> Any:
        """Builds a request config carrying an explicit timeout, defensively.

        Args:
            thinking_level: Optional Gemini 3 thinking depth to request
                (e.g. "high" for the mastermind planner calls).

        Returns:
            A `GenerateContentConfig` with an HTTP timeout (and thinking
            config when requested), or None if the installed SDK does not
            expose that surface (so the call still runs).
        """
        try:
            from google.genai import types  # lazy: SDK dependency

            kwargs: dict[str, Any] = {
                "http_options": types.HttpOptions(timeout=self._settings.gemini_timeout_ms)
            }
            if thinking_level:
                kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level.upper()
                )
            return types.GenerateContentConfig(**kwargs)
        except Exception as error:  # noqa: BLE001 - timeout is best-effort, never fatal
            logger.warning("could not build a per-request timeout config: %s", error)
            return None

    def ground(self, task: TaskState, step: Step, screenshot: Any) -> GroundedAction:
        """Asks Gemini where to act next for `step`, given the latest screenshot.

        Args:
            task: The current hold-state (used for goal context).
            step: The step selected by `TaskState.next_actionable_step()`.
            screenshot: A PIL image (or genai-compatible image part).

        Returns:
            A `GroundedAction` describing what to do and where.

        Raises:
            MissingApiKeyError: If no real API key is configured.
            VisionError: If the model response has no usable candidate.
        """
        self._remember(screenshot)

        prompt = (
            f"Goal: {task.goal}\n"
            f"Current step: {step.desc}\n"
            f"Notes so far: {step.note or 'none'}\n"
            "You control a real macOS desktop. The LAST image is the CURRENT "
            "screen; any earlier images are recent history for context only. "
            "Decide the SINGLE next action to progress this step, targeting "
            "the CURRENT screen, and reply with ONE JSON object, nothing else:\n"
            '  {"action":"click","box":[ymin,xmin,ymax,xmax]}   (box normalized 0-1000)\n'
            '  {"action":"type","text":"the text to type"}\n'
            '  {"action":"hotkey","keys":["command","v"]}\n'
            '  {"action":"scroll","amount":-5}   (wheel clicks, positive = up)\n'
            '  {"action":"noop"}\n'
            'Also include a "confidence" field (0.0-1.0): your probability '
            "that this exact action is the right next move."
        )
        contents = [*self._recent_screenshots, prompt]

        # Adaptive self-consistency: the model reports its own confidence.
        # A confident first answer acts immediately (single fast call); only
        # a low-confidence one escalates to extra samples + majority vote,
        # so the latency cost is paid exactly on the turns that need it.
        first = self._parse_response(self._generate(contents))
        samples_cap = max(1, self._settings.ground_samples)
        if samples_cap == 1 or not self._is_uncertain(first):
            return first

        logger.info(
            "grounding confidence %.2f below %.2f -> escalating to %d-sample vote",
            first.confidence if first.confidence is not None else -1.0,
            self._settings.ground_confidence,
            samples_cap,
        )
        actions = [first]
        for _ in range(samples_cap - 1):
            try:
                response = self._generate(contents)
            except VisionError as error:
                logger.warning("extra grounding sample failed: %s", error)
                continue
            actions.append(self._parse_response(response))
        return _vote_grounded(actions)

    def _is_uncertain(self, action: GroundedAction) -> bool:
        """True when a grounded action's self-reported confidence is too low.

        A missing confidence (model ignored the instruction) counts as
        confident: escalating on every non-compliant reply would silently
        triple latency, and the parse fallbacks already handle those cases.
        """
        if action.confidence is None:
            return False
        return action.confidence < self._settings.ground_confidence

    def _parse_response(self, response: Any) -> GroundedAction:
        """Extracts a GroundedAction out of a `generate_content` response.

        Args:
            response: The raw SDK response object.

        Returns:
            A best-effort `GroundedAction`; falls back to kind="noop" with
            the raw text as reasoning if no box can be parsed.

        Raises:
            VisionError: If the response has no text/candidates at all.
        """
        text = getattr(response, "text", None)
        if text is None:
            raise VisionError("Gemini response had no text content")

        action = _extract_json_object(text)
        if action is not None:
            grounded = _grounded_from_action(action, text)
            if grounded is not None:
                return grounded

        match = _BOX_RE.search(text)
        if match:
            box = tuple(float(g) for g in match.groups())
            return GroundedAction(kind="click", box=box, reasoning=text.strip())
        return GroundedAction(kind="noop", reasoning=text.strip())

    def plan_steps(self, task: TaskState, instruction: str, screenshot: Any | None) -> list[str]:
        """Decomposes a natural-language instruction into ordered UI steps.

        This is the CERVEAU as planner: it turns "assign the network bugs to
        infra" into the concrete step list the hold-state needs before the
        loop can ground+act anything. Called by `agent.py` when a fresh
        instruction arrives and the plan has no actionable step.

        Args:
            task: The current hold-state (its goal frames the decomposition).
            instruction: The raw operator instruction (voice/text).
            screenshot: Optional current screen, so the plan can reference
                what is actually visible.

        Returns:
            An ordered list of short step descriptions (possibly empty if the
            model returned nothing usable -- the loop then simply stalls).

        Raises:
            MissingApiKeyError: If no real API key is configured.
        """
        visual_parts: list[Any] = []
        if screenshot is not None:
            self._remember(screenshot)
            visual_parts = list(self._recent_screenshots)

        prompt = (
            f"Goal: {task.goal}\n"
            f"Operator instruction: {instruction}\n"
            "You control a real macOS desktop. Break this instruction into a "
            f"short ordered list (max {_MAX_PLAN_STEPS}) of concrete, "
            "single-action UI steps, e.g. 'open Slack', 'click the #general "
            "channel', 'type the message'. Respond ONLY with a JSON array of "
            "short strings."
        )
        contents = [*visual_parts, prompt]

        response = self._generate(
            contents,
            model=self._planner_model(),
            thinking_level=self._settings.planner_thinking_level or None,
        )
        return self._parse_steps(response)

    def verify_step_done(self, task: TaskState, step: Step, screenshot: Any) -> bool:
        """Asks Gemini whether `step` looks complete on the current screenshot.

        This is what advances the loop across steps live: after acting, the
        next turn feeds the fresh screen here, and a YES marks the step done.

        Args:
            task: The current hold-state (its goal frames the judgement).
            step: The step the agent just acted on.
            screenshot: The current screen (PIL image / genai image part).

        Returns:
            True only if the model answers affirmatively; any ambiguity is
            treated as "not done" so the agent keeps working the step.

        Raises:
            MissingApiKeyError: If no real API key is configured.
        """
        prompt = (
            f"Goal: {task.goal}\n"
            f"Step under check: {step.desc}\n"
            "Looking ONLY at the current screenshot, is this step now complete? "
            "Answer with a single word: YES or NO."
        )
        response = self._generate([screenshot, prompt])
        return _parse_yes_no(getattr(response, "text", None))

    def extract_override(self, task: TaskState, instruction: str) -> tuple[str, str] | None:
        """Extracts a (when, rule) override pair from a spoken correction.

        This is the live half of the override path: `router.is_correction`
        (zero-LLM) decides the phrase SOUNDS like a correction, then this one
        Gemini call decides whether it really is one and which remaining
        steps it recalibrates. Text-only call: the correction is about the
        plan, not the pixels, so no screenshot is attached.

        Args:
            task: The current hold-state (goal + remaining steps give the
                model the vocabulary to anchor `when` on).
            instruction: The raw operator phrase flagged as a correction.

        Returns:
            A (when, rule) tuple ready for `TaskState.apply_override`, or
            None if the model judges the phrase is not actually a correction
            (the caller then treats it as a plain instruction).

        Raises:
            MissingApiKeyError: If no real API key is configured.
            VisionError: If the breaker is open or every attempt failed.
        """
        remaining = [
            f"- {step.id}: {step.desc} [{step.status.value}]"
            for step in task.steps
            if step.status.value != "done"
        ]
        prompt = (
            f"Goal: {task.goal}\n"
            f"Remaining plan steps:\n" + "\n".join(remaining) + "\n"
            f"The operator just said: \"{instruction}\"\n"
            "Is this a CORRECTION of how the remaining steps should be done "
            "(a changed rule, target, or destination), rather than a new task?\n"
            "Reply with ONE JSON object, nothing else:\n"
            '  {"correction":true,"when":"<short substring of the affected step '
            'descriptions>","rule":"<the corrected rule, one imperative sentence>"}\n'
            '  {"correction":false}'
        )
        response = self._generate(
            [prompt],
            model=self._planner_model(),
            thinking_level=self._settings.planner_thinking_level or None,
        )
        return _parse_override(getattr(response, "text", None))

    def _parse_steps(self, response: Any) -> list[str]:
        """Extracts an ordered list of step strings from a model response.

        Args:
            response: The raw `generate_content` response object.

        Returns:
            A cleaned, de-blanked list of at most `_MAX_PLAN_STEPS` step
            descriptions; empty if the response carried no parseable JSON array.
        """
        text = getattr(response, "text", None)
        if not text:
            return []
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [str(item).strip() for item in data if str(item).strip()][:_MAX_PLAN_STEPS]


# Two click samples whose centers are both within this normalized distance
# (out of 1000) count as votes for the SAME target; farther apart, they are
# different UI elements and must not be averaged into a point between them.
_CLICK_CLUSTER_TOLERANCE = 50.0


def _vote_grounded(actions: list[GroundedAction]) -> GroundedAction:
    """Majority-votes N grounding samples into one action (self-consistency).

    The winning `kind` is the most common one. Within the winning kind:
    clicks are clustered by proximity (the densest cluster wins and its
    boxes are averaged -- never averaging across distinct targets), text
    and hotkeys go to the most common value, scroll takes the median.

    Args:
        actions: One parsed GroundedAction per sample (at least one).

    Returns:
        The consensus action.
    """
    if len(actions) == 1:
        return actions[0]

    winning_kind = Counter(a.kind for a in actions).most_common(1)[0][0]
    cluster = [a for a in actions if a.kind == winning_kind]

    if winning_kind == "click":
        boxes = [a.box for a in cluster if a.box is not None]
        if not boxes:
            return cluster[0]
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]

        def neighborhood(i: int) -> list[int]:
            return [
                j
                for j, (cy, cx) in enumerate(centers)
                if abs(cy - centers[i][0]) <= _CLICK_CLUSTER_TOLERANCE
                and abs(cx - centers[i][1]) <= _CLICK_CLUSTER_TOLERANCE
            ]

        anchor = max(range(len(boxes)), key=lambda i: len(neighborhood(i)))
        group = neighborhood(anchor)
        consensus_box = tuple(
            sum(boxes[j][k] for j in group) / len(group) for k in range(4)
        )
        return GroundedAction(
            kind="click", box=consensus_box, reasoning=cluster[anchor].reasoning
        )
    if winning_kind == "type":
        text = Counter(a.text for a in cluster).most_common(1)[0][0]
        return next(a for a in cluster if a.text == text)
    if winning_kind == "hotkey":
        keys = Counter(tuple(a.keys or ()) for a in cluster).most_common(1)[0][0]
        return next(a for a in cluster if tuple(a.keys or ()) == keys)
    if winning_kind == "scroll":
        amounts = sorted(a.amount or 0 for a in cluster)
        median = amounts[len(amounts) // 2]
        return GroundedAction(
            kind="scroll", amount=median, reasoning=cluster[0].reasoning
        )
    return cluster[0]


def _is_transient_error(error: Exception) -> bool:
    """Decides whether a failed Gemini call is worth retrying.

    Transient: timeouts, connection drops, HTTP 5xx and 429 rate limits.
    Not transient: other 4xx (invalid key, permission denied, bad request) --
    retrying those burns live demo seconds to get the exact same answer.
    Errors with no recognizable status code default to transient, since a
    false "permanent" would drop a recoverable call.

    Args:
        error: The exception raised by the SDK call.

    Returns:
        True if the call should be retried.
    """
    code = getattr(error, "code", None)
    if not isinstance(code, int):
        code = getattr(error, "status_code", None)
    if isinstance(code, int) and 400 <= code < 500 and code != 429:
        return False
    return True


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Finds and parses the first JSON object in a text blob.

    Args:
        text: Raw model text that may contain a `{...}` action object.

    Returns:
        The first parseable dict, or None if no valid JSON object is present.
        Scanning brace by brace (instead of one greedy `{.*}` regex) keeps a
        reply containing TWO objects, or an object followed by prose and a
        stray `}`, parseable -- a greedy span would be invalid JSON.
    """
    decoder = json.JSONDecoder()
    for brace in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, brace.start())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _grounded_from_action(action: dict[str, Any], reasoning: str) -> GroundedAction | None:
    """Maps a parsed `{action: ...}` object to a GroundedAction.

    Args:
        action: The parsed action dict from the model.
        reasoning: The full model text, kept for the HUD/log.

    Returns:
        A `GroundedAction` for a well-formed click/type/hotkey/scroll/noop,
        or None if the object is malformed -- including a box/amount that is
        not numeric (the caller then falls back to the box regex). Malformed
        model output must NEVER crash the loop.
    """
    kind = str(action.get("action", "")).lower()
    confidence = _parse_confidence(action.get("confidence"))
    if kind == "click":
        box = action.get("box")
        if isinstance(box, list) and len(box) == 4:
            try:
                parsed_box = tuple(float(v) for v in box)
            except (TypeError, ValueError):
                return None
            return GroundedAction(
                kind="click", box=parsed_box, reasoning=reasoning.strip(), confidence=confidence
            )
        return None
    if kind == "type":
        text = action.get("text")
        if isinstance(text, str) and text:
            return GroundedAction(
                kind="type", text=text, reasoning=reasoning.strip(), confidence=confidence
            )
        return None
    if kind == "hotkey":
        keys = action.get("keys")
        if isinstance(keys, list) and keys:
            return GroundedAction(
                kind="hotkey",
                keys=[str(k) for k in keys],
                reasoning=reasoning.strip(),
                confidence=confidence,
            )
        return None
    if kind == "scroll":
        try:
            amount = int(action.get("amount"))
        except (TypeError, ValueError):
            return None
        return GroundedAction(
            kind="scroll", amount=amount, reasoning=reasoning.strip(), confidence=confidence
        )
    if kind == "noop":
        return GroundedAction(kind="noop", reasoning=reasoning.strip(), confidence=confidence)
    return None


def _parse_confidence(value: Any) -> float | None:
    """Parses a model-reported confidence, tolerating junk.

    Args:
        value: The raw "confidence" JSON value (number, string, or garbage).

    Returns:
        The confidence clamped to [0.0, 1.0], or None if unparseable --
        a malformed confidence must never invalidate an otherwise good
        action, it just opts out of the escalation heuristic.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(parsed, 0.0), 1.0)


def _parse_override(text: str | None) -> tuple[str, str] | None:
    """Parses the JSON verdict of `extract_override` into a (when, rule) pair.

    Args:
        text: The model's answer text.

    Returns:
        (when, rule) if the model confirmed a correction with both fields
        non-blank; None on `correction: false`, missing fields, or any
        malformed payload (the safe default: treat as a plain instruction).
    """
    if not text:
        return None
    data = _extract_json_object(text)
    if not data or not data.get("correction"):
        return None
    when = str(data.get("when") or "").strip()
    rule = str(data.get("rule") or "").strip()
    if not when or not rule:
        return None
    return when, rule


def _parse_yes_no(text: str | None) -> bool:
    """Interprets a YES/NO (or OUI/NON) completion answer, defaulting to False.

    Args:
        text: The model's answer text.

    Returns:
        True only on an explicit affirmative; anything else is False so the
        agent keeps working the step rather than declaring it done on doubt.
    """
    if not text:
        return False
    head = text.strip().lower()
    return head.startswith("yes") or head.startswith("oui")
