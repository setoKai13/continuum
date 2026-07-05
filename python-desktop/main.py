"""Continuum entrypoint: wires config/state/memory/router/vision/mac_control/
stt/tts/hud into one live agent process.

Launch: `python main.py` (new task) or `python main.py --resume <task_id>`
(reloads the hold-state, proving it survives process restarts). See
RUNBOOK.md for the exact live command, the 3 required macOS permissions,
and the TODO(tom) items this headless build cannot complete on its own.

Nothing in this module touches a GUI/native/network library at import
time -- every heavy dependency lives behind a lazy import inside
`mac_control.py`, `vision.py`, or `stt.py`/`tts.py`, so
`python -c "import main"` succeeds even without pyautogui/mss/sounddevice/
faster-whisper/pynput/google-genai installed.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import uuid
from typing import Any

from agent import ActFn, ActionPlan, AgentLoop, GroundFn, Observation, ObserveFn, OverrideFn, PlanFn, VerifyFn
from config import DEFAULT_GOAL, Settings, is_placeholder_key, load_settings
from hud import Hud
from mac_control import MacController, check_macos_permissions, denormalize_box
from memory import MemoryStore
from router import IntentRouter, is_correction
from state import Step, TaskState
from tts import Speaker
from vision import GeminiVision, VisionError

logger = logging.getLogger(__name__)


def configure_logging(settings: Settings) -> None:
    """Routes logs to a file, keeping the console clean for the Rich HUD.

    A stderr line emitted while `rich.Live` is drawing tears the HUD apart
    mid-demo, so INFO/WARNING go to `settings.log_path` (tail it in a second
    terminal) and only ERROR-level events still reach the console.

    Args:
        settings: Application settings (log path and level).
    """
    logging.basicConfig(
        filename=settings.log_path,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.ERROR)
    logging.getLogger().addHandler(console)


def build_new_task(task_id: str, goal: str) -> TaskState:
    """Creates a fresh, empty-plan TaskState for a brand-new run.

    Args:
        task_id: Stable identifier for this task (used by `--resume` later).
        goal: The operator's stated objective.

    Returns:
        A new `TaskState` with no steps yet. The operator's first voice
        instruction populates the plan live: `agent.AgentLoop` calls the
        injected `plan_fn` (Gemini `plan_steps`) to decompose it into steps,
        after which `next_actionable_step()` has something to act on.
    """
    return TaskState(task_id=task_id, goal=goal)


def build_observe_fn(stt_queue: "queue.Queue[str]", mac: MacController, task: TaskState) -> ObserveFn:
    """Builds the OBSERVE callable: new voice transcript and/or fresh screenshot.

    Two ways a turn produces an observation:
      * a new voice transcript arrived (the operator spoke), or
      * there is an actionable step to progress (the agent keeps working the
        current plan by looking at the screen each turn).

    If neither holds -- no new voice AND no plan to advance -- it returns None:
    the loop stalls. That preserves the causal story ("no intent, no action":
    with nothing spoken and no plan, nothing happens) while letting an
    already-planned task drive itself forward instead of freezing after the
    first action.

    Args:
        stt_queue: Queue fed by `stt.PushToTalkListener` on each transcript.
        mac: Controller used to grab the current screenshot.
        task: The live hold-state, checked for an actionable step.

    Returns:
        A zero-arg callable suitable for `AgentLoop(observe_fn=...)`.
    """

    def _observe() -> Observation | None:
        try:
            instruction: str | None = stt_queue.get_nowait()
        except queue.Empty:
            instruction = None
        if instruction is None and task.next_actionable_step() is None:
            return None
        try:
            screenshot = mac.capture_screenshot_logical()
        except Exception:  # pragma: no cover - live hardware path
            logger.warning("screenshot capture failed; grounding this turn without one")
            screenshot = None
        return Observation(instruction=instruction, screenshot=screenshot)

    return _observe


def build_ground_fn(vision: GeminiVision, mac: MacController, router: IntentRouter) -> GroundFn:
    """Builds the DECIDE->target callable: router fast-path, else Gemini box.

    Two-tier DECIDE: first the zero-LLM `IntentRouter` fast-paths (a step
    like "open Slack" becomes an `open_app`/`open_url` action with no Gemini
    call at all); only steps that do not match a fast path fall through to
    vision grounding, where Gemini returns a normalized box.

    NOTE(tom): `mac.capture_screenshot_logical()` already resizes the
    screenshot to the logical (pyautogui) resolution before it reaches
    Gemini, so the Retina SCALE factor is 1.0 here and `denormalize_box`
    maps straight onto `mac.screen_size_logical()` -- no extra division
    needed (see mac_control.py docstrings for the alternative path).

    Args:
        vision: The Gemini vision-grounding client.
        mac: Controller used to read the current logical screen size.
        router: Zero-LLM intent router for cheap "open app/URL" steps.

    Returns:
        A callable suitable for `AgentLoop(ground_fn=...)`.
    """

    def _ground(task: TaskState, step: Step, observation: Observation) -> ActionPlan | None:
        routed = router.route(step.desc)
        if routed is not None:
            return ActionPlan(kind=routed.kind, step_id=step.id, text=routed.target)
        if observation.screenshot is None:
            return None
        try:
            grounded = vision.ground(task, step, observation.screenshot)
        except VisionError as error:
            # A failed/opened-breaker Gemini call degrades into a stall turn:
            # the loop keeps living and retries next turn, instead of dying
            # with a traceback in the middle of the demo.
            logger.warning("grounding failed for step %s: %s", step.id, error)
            return None
        if grounded.kind == "click" and grounded.box is not None:
            width, height = mac.screen_size_logical()
            target = denormalize_box(grounded.box, width, height)
            return ActionPlan(kind="click", step_id=step.id, target=target, text=grounded.reasoning)
        if grounded.kind == "type" and grounded.text:
            return ActionPlan(kind="type", step_id=step.id, text=grounded.text)
        if grounded.kind == "hotkey" and grounded.keys:
            return ActionPlan(kind="hotkey", step_id=step.id, target=tuple(grounded.keys))
        if grounded.kind == "scroll" and grounded.amount is not None:
            return ActionPlan(kind="scroll", step_id=step.id, target=grounded.amount)
        return None

    return _ground


def build_plan_fn(vision: GeminiVision) -> PlanFn:
    """Builds the planner callable: instruction -> ordered step descriptions.

    Args:
        vision: The Gemini client (its `plan_steps` does the decomposition).

    Returns:
        A callable suitable for `AgentLoop(plan_fn=...)`. A failed Gemini
        call returns an empty plan (the loop stalls and the operator can
        simply repeat the instruction) instead of crashing the run.
    """

    def _plan(task: TaskState, instruction: str, screenshot: Any) -> list[str]:
        try:
            return vision.plan_steps(task, instruction, screenshot)
        except VisionError as error:
            logger.warning("planning failed for instruction %r: %s", instruction, error)
            return []

    return _plan


def build_verify_fn(vision: GeminiVision) -> VerifyFn:
    """Builds the step-completion judge, degrading failures to "not done".

    Args:
        vision: The Gemini client (its `verify_step_done` does the judging).

    Returns:
        A callable suitable for `AgentLoop(verify_fn=...)`. On a failed
        Gemini call the step simply stays in progress -- the safe answer.
    """

    def _verify(task: TaskState, step: Step, screenshot: Any) -> bool:
        try:
            return vision.verify_step_done(task, step, screenshot)
        except VisionError as error:
            logger.warning("verification failed for step %s: %s", step.id, error)
            return False

    return _verify


def build_act_fn(mac: MacController, settings: Settings) -> ActFn:
    """Builds the actuator callable, adding a UI-settle pause after actions.

    Without the pause, the next turn's screenshot is taken while the UI is
    still animating (menu opening, app launching) and both grounding and
    verification judge a transitional frame.

    Args:
        mac: The real Mac controller.
        settings: Application settings (ui_settle_s).

    Returns:
        A callable suitable for `AgentLoop(act_fn=...)`.
    """

    def _act(plan: ActionPlan) -> dict[str, Any]:
        result = mac.execute(plan)
        if plan.kind != "noop" and settings.ui_settle_s > 0:
            mac.wait(settings.ui_settle_s)
        return result

    return _act


def build_override_fn(vision: GeminiVision) -> OverrideFn:
    """Builds the live correction detector: marker gate, then Gemini extraction.

    Two-tier, mirroring DECIDE: `router.is_correction` (zero-LLM markers like
    "non, ...", "en fait ...") decides WHETHER to spend a model call, then
    `vision.extract_override` decides whether the phrase really corrects the
    remaining steps and extracts the (when, rule) pair. Any vision failure
    degrades to "plain instruction" instead of crashing the turn: losing one
    correction is recoverable live (the operator repeats it), a dead loop is not.

    Args:
        vision: The Gemini client (its `extract_override` does the extraction).

    Returns:
        A callable suitable for `AgentLoop(override_fn=...)`.
    """

    def _override(task: TaskState, instruction: str) -> tuple[str, str] | None:
        if not is_correction(instruction):
            return None
        try:
            return vision.extract_override(task, instruction)
        except Exception:  # pragma: no cover - live network path
            logger.warning("override extraction failed; treating as plain instruction")
            return None

    return _override


def check_permissions_or_exit() -> None:
    """Probes the 3 macOS permissions and exits clearly if any is missing.

    Constraint from the spec: "Permission manquante -> arret clair avec
    message (pas d'echec silencieux)."
    """
    results = check_macos_permissions()
    missing = [name for name, ok in results.items() if not ok]
    if missing:
        logger.error(
            "Missing macOS permission(s): %s. Grant Accessibility, Screen "
            "Recording and Microphone to your terminal/python (System "
            "Settings > Privacy & Security), then relaunch. See RUNBOOK.md.",
            ", ".join(missing),
        )
        sys.exit(1)


def build_kill_switch(settings: Settings, stop_event: threading.Event) -> Any:
    """Builds a start/stop-able listener that sets `stop_event` on the kill key.

    Args:
        settings: Application settings (kill_switch_key).
        stop_event: Event the agent loop polls at the top of every turn.

    Returns:
        An object with `.start()` / `.stop()` methods.
    """

    class _KillSwitch:
        def __init__(self) -> None:
            self._listener = None
            self._target_key = None

        def _on_press(self, key: Any) -> None:
            if key == self._target_key:
                stop_event.set()

        def start(self) -> None:
            from pynput import keyboard  # lazy: native input dependency
            from stt import resolve_key

            self._target_key = resolve_key(settings.kill_switch_key)
            self._listener = keyboard.Listener(on_press=self._on_press)
            self._listener.start()

        def stop(self) -> None:
            if self._listener is not None:
                self._listener.stop()
                self._listener = None

    return _KillSwitch()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments for the Continuum entrypoint.

    Args:
        argv: Argument list (defaults to `sys.argv[1:]`).

    Returns:
        Parsed namespace with `resume` and `goal` attributes.
    """
    parser = argparse.ArgumentParser(description="Continuum -- hold-state Mac agent")
    parser.add_argument("--resume", metavar="TASK_ID", default=None, help="Resume a paused/active task by id")
    parser.add_argument("--goal", default=DEFAULT_GOAL, help="Goal for a brand-new task")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Boots Continuum: checks permissions, loads/resumes state, runs the loop.

    Args:
        argv: Optional CLI argument override (used by tests); defaults to
            `sys.argv[1:]`.

    Returns:
        Process exit code (0 on a clean run).
    """
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(settings)

    # Fail loud at boot, not at the first voice command: a live run without a
    # real key can only crash later inside the loop (headless proofs go
    # through pytest/dry_run.py, never through this entrypoint).
    if is_placeholder_key(settings.gemini_api_key):
        logger.error(
            "GEMINI_API_KEY is missing or a placeholder. Copy .env.example to "
            ".env and set a real key (RUNBOOK.md section 2), then relaunch."
        )
        return 1

    check_permissions_or_exit()

    memory = MemoryStore(settings.db_path)
    mac = MacController(pyautogui_pause_s=settings.pyautogui_pause_s)
    vision = GeminiVision(settings)
    router = IntentRouter()  # DECIDE tier 1: zero-LLM fast-paths, wired into build_ground_fn
    speaker = Speaker()

    if args.resume:
        task = memory.resume_task_state(args.resume)
        if task is None:
            logger.error("No task found for id=%s; cannot resume.", args.resume)
            return 1
        logger.info(
            "Resuming %s (session_count=%s): %s",
            task.task_id,
            task.session_count,
            task.render()["progress"],
        )
    else:
        task_id = uuid.uuid4().hex[:8]
        task = build_new_task(task_id, args.goal)
        memory.save_task_state(task)
        logger.info("Starting new task %s: %s", task.task_id, task.goal)

    stop_event = threading.Event()
    stt_queue: "queue.Queue[str]" = queue.Queue()

    kill_switch = build_kill_switch(settings, stop_event)
    from stt import PushToTalkListener  # lazy: pulls in pynput/sounddevice at start()

    ptt_listener = PushToTalkListener(settings, on_transcript=stt_queue.put)

    summary = None
    with Hud() as hud:
        ptt_listener.start()
        kill_switch.start()
        try:
            loop = AgentLoop(
                task=task,
                memory=memory,
                observe_fn=build_observe_fn(stt_queue, mac, task),
                ground_fn=build_ground_fn(vision, mac, router),
                act_fn=build_act_fn(mac, settings),
                speak_fn=speaker.say,
                hud=hud,
                stop_event=stop_event,
                max_turns=settings.max_turns,
                plan_fn=build_plan_fn(vision),
                verify_fn=build_verify_fn(vision),
                override_fn=build_override_fn(vision),
                idle_sleep_s=settings.loop_idle_sleep_s,
                max_idle_turns=settings.max_idle_turns,
                max_step_attempts=settings.max_step_attempts,
            )
            summary = loop.run()
        finally:
            ptt_listener.stop()
            kill_switch.stop()

    logger.info("Run summary: %s", summary)
    memory.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
