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
from pathlib import Path
from typing import Any

from agent import ActFn, ActionPlan, AgentLoop, GroundFn, Observation, ObserveFn, OverrideFn, PlanFn
from config import DEFAULT_GOAL, Settings, is_placeholder_key, load_settings
from hud import Hud
from interactions_cu import InteractionsComputerUse
from mac_control import MacController, check_macos_permissions, denormalize_box, denormalize_point
from memory import MemoryStore
from router import IntentRouter, is_correction
from state import Step, TaskState
from trace import NullTracer, TraceLogHandler, Tracer
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


def build_observe_fn(
    stt_queue: "queue.Queue[str]",
    mac: MacController,
    task: TaskState,
    tracer: Any = None,
) -> ObserveFn:
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

    trace = tracer or NullTracer()

    def _observe() -> Observation | None:
        try:
            instruction: str | None = stt_queue.get_nowait()
        except queue.Empty:
            instruction = None
        if instruction is not None:
            trace.event("HEARD", instruction)
        if instruction is None and task.next_actionable_step() is None:
            return None
        try:
            screenshot = mac.capture_screenshot_logical()
        except Exception:  # pragma: no cover - live hardware path
            logger.warning("screenshot capture failed; grounding this turn without one")
            screenshot = None
        return Observation(instruction=instruction, screenshot=screenshot)

    return _observe


def build_ground_fn(
    vision: GeminiVision,
    mac: MacController,
    router: IntentRouter,
    settings: Settings,
    cu: InteractionsComputerUse | None = None,
    tracer: Any = None,
) -> GroundFn:
    """Builds the DECIDE->target callable: router fast-path, else model grounding.

    Two-tier DECIDE: first the zero-LLM `IntentRouter` fast-paths (a step
    like "open Slack" becomes an `open_app`/`open_url` action with no model
    call at all); only steps that do not match a fast path fall through to
    model grounding. The fast-path tier runs upstream in BOTH grounding
    routes -- only the fallback differs, selected by `settings.cu_mode`:

      * "grounding" (default): `vision.ground` returns a normalized
        [ymin,xmin,ymax,xmax] box (`denormalize_box`). Unchanged.
      * "interactions": the official Computer Use path (`cu.ground` ->
        `interactions.create`) returns a 0..CU_NORM_MAX {x,y} point
        (`denormalize_point`).

    NOTE(tom): `mac.capture_screenshot_logical()` already resizes the
    screenshot to the logical (pyautogui) resolution before it reaches
    Gemini, so the Retina SCALE factor is 1.0 here and `denormalize_box`/
    `denormalize_point` map straight onto `mac.screen_size_logical()` -- no
    extra division needed (see mac_control.py docstrings).

    Args:
        vision: The Gemini vision-grounding client (Route 2 fallback).
        mac: Controller used to read the current logical screen size.
        router: Zero-LLM intent router for cheap "open app/URL" steps.
        settings: Application settings (its `cu_mode` picks the fallback).
        cu: The Interactions Computer Use client (Route 1); required when
            `cu_mode == "interactions"`, ignored otherwise.

    Returns:
        A callable suitable for `AgentLoop(ground_fn=...)`.
    """

    trace = tracer or NullTracer()

    def _ground(task: TaskState, step: Step, observation: Observation) -> ActionPlan | None:
        routed = router.route(step.desc)
        if routed is not None:
            trace.event("ROUTE", f"[{step.id}] fast-path {routed.kind}: {routed.target}")
            # completes_step: a literal keyboard/open step IS its own
            # execution -- done on success, no model judgement spent.
            if routed.kind == "type":
                return ActionPlan(kind="type", step_id=step.id, text=routed.target, completes_step=True)
            if routed.kind == "hotkey":
                return ActionPlan(kind="hotkey", step_id=step.id, target=routed.keys, completes_step=True)
            if routed.kind == "scroll":
                return ActionPlan(kind="scroll", step_id=step.id, target=int(routed.target), completes_step=True)
            return ActionPlan(kind=routed.kind, step_id=step.id, text=routed.target, completes_step=True)
        if observation.screenshot is None:
            return None
        if settings.cu_mode == "interactions" and cu is not None:
            return _ground_via_interactions(task, step, observation.screenshot, cu, mac, settings, trace)
        try:
            grounded = vision.ground(task, step, observation.screenshot)
        except VisionError as error:
            # A failed/opened-breaker Gemini call degrades into a stall turn:
            # the loop keeps living and retries next turn, instead of dying
            # with a traceback in the middle of the demo.
            logger.warning("grounding failed for step %s: %s", step.id, error)
            trace.event("ERROR", f"[{step.id}] grounding failed: {error}")
            return None
        confidence = f"{grounded.confidence:.2f}" if grounded.confidence is not None else "?"
        trace.event(
            "THINK",
            f"[{step.id}] {step.desc} -> {grounded.kind} (conf {confidence}) :: "
            f"{grounded.reasoning or ''}",
        )
        if grounded.kind == "done":
            trace.event("VERIFY", f"[{step.id}] {step.desc} -> already satisfied")
            return ActionPlan(kind="done", step_id=step.id)
        plan: ActionPlan | None = None
        if grounded.kind == "click" and grounded.box is not None:
            width, height = mac.screen_size_logical()
            target = denormalize_box(grounded.box, width, height)
            plan = ActionPlan(kind="click", step_id=step.id, target=target, text=grounded.reasoning)
        elif grounded.kind == "type" and grounded.text:
            plan = ActionPlan(kind="type", step_id=step.id, text=grounded.text)
        elif grounded.kind == "hotkey" and grounded.keys:
            plan = ActionPlan(kind="hotkey", step_id=step.id, target=tuple(grounded.keys))
        elif grounded.kind == "scroll" and grounded.amount is not None:
            plan = ActionPlan(kind="scroll", step_id=step.id, target=grounded.amount)
        if plan is not None:
            detail = plan.target if plan.kind != "type" else plan.text
            trace.event("ACTION", f"[{step.id}] {plan.kind}: {detail}")
        else:
            trace.event("LOOP", f"[{step.id}] model returned {grounded.kind} -> no actionable plan")
        return plan

    return _ground


def _ground_via_interactions(
    task: TaskState,
    step: Step,
    screenshot: Any,
    cu: InteractionsComputerUse,
    mac: MacController,
    settings: Settings,
    trace: Any,
) -> ActionPlan | None:
    """Grounds one step through the Computer Use path, mapping a point to a plan.

    Mirrors the vision branch of `build_ground_fn` but for Route 1: the CU
    model returns a 0..CU_NORM_MAX {x,y} point (`denormalize_point`) instead
    of a box, and signals "step already satisfied" by issuing no function
    call (mapped to `ActionPlan(kind="done")`, exactly like the grounding
    path's done-or-act). The live session id is copied back onto the task so
    the agent loop's next save persists it for `--resume`. A None from
    `cu.ground` (unsupported call, safety refusal, or any failure) stalls.

    Args:
        task: The current hold-state (its session id is updated here).
        step: The step being grounded.
        screenshot: The current logical-resolution screenshot.
        cu: The Interactions Computer Use client.
        mac: Controller used to read the logical screen size.
        settings: Application settings (cu_norm_max for denormalization).
        trace: The trace sink for HUD/debug events.

    Returns:
        An `ActionPlan` to execute, `ActionPlan(kind="done")` if the step is
        satisfied, or None to stall.
    """
    action = cu.ground(task, step, screenshot)
    task.interactions_session_id = cu.session_id  # persisted by the loop's save
    if action is None:
        trace.event("LOOP", f"[{step.id}] CU returned no action -> stall")
        return None
    trace.event(
        "THINK", f"[{step.id}] {step.desc} -> CU {action.kind} :: {action.reasoning or ''}"
    )
    if action.kind == "done":
        trace.event("VERIFY", f"[{step.id}] {step.desc} -> already satisfied (CU)")
        return ActionPlan(kind="done", step_id=step.id)
    plan: ActionPlan | None = None
    if action.kind == "click" and action.point is not None:
        width, height = mac.screen_size_logical()
        target = denormalize_point(action.point[0], action.point[1], width, height, settings.cu_norm_max)
        plan = ActionPlan(kind="click", step_id=step.id, target=target, text=action.reasoning)
    elif action.kind == "type" and action.text:
        plan = ActionPlan(kind="type", step_id=step.id, text=action.text)
    elif action.kind == "hotkey" and action.keys:
        plan = ActionPlan(kind="hotkey", step_id=step.id, target=tuple(action.keys))
    elif action.kind == "scroll" and action.amount is not None:
        plan = ActionPlan(kind="scroll", step_id=step.id, target=action.amount)
    if plan is not None:
        detail = plan.target if plan.kind != "type" else plan.text
        trace.event("ACTION", f"[{step.id}] {plan.kind}: {detail}")
    else:
        trace.event("LOOP", f"[{step.id}] CU returned {action.kind} -> no actionable plan")
    return plan


def build_plan_fn(vision: GeminiVision, tracer: Any = None) -> PlanFn:
    """Builds the planner callable: instruction -> ordered step descriptions.

    Args:
        vision: The Gemini client (its `plan_steps` does the decomposition).

    Returns:
        A callable suitable for `AgentLoop(plan_fn=...)`. A failed Gemini
        call returns an empty plan (the loop stalls and the operator can
        simply repeat the instruction) instead of crashing the run.
    """

    trace = tracer or NullTracer()

    def _plan(task: TaskState, instruction: str, screenshot: Any) -> list[str]:
        trace.event("PLAN", f"decomposing: {instruction}")
        try:
            steps = vision.plan_steps(task, instruction, screenshot)
        except VisionError as error:
            logger.warning("planning failed for instruction %r: %s", instruction, error)
            trace.event("ERROR", f"planning failed: {error}")
            return []
        trace.event("PLAN", f"steps: {steps}" if steps else "planner returned no steps")
        return steps

    return _plan


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


def build_override_fn(vision: GeminiVision, tracer: Any = None) -> OverrideFn:
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

    trace = tracer or NullTracer()

    def _override(task: TaskState, instruction: str) -> tuple[str, str] | None:
        if not is_correction(instruction):
            return None
        trace.event("OVERRIDE", f"correction marker heard: {instruction}")
        try:
            decision = vision.extract_override(task, instruction)
        except Exception:  # pragma: no cover - live network path
            logger.warning("override extraction failed; treating as plain instruction")
            trace.event("ERROR", "override extraction failed; treated as plain instruction")
            return None
        if decision is not None:
            trace.event("OVERRIDE", f"when={decision[0]!r} rule={decision[1]!r}")
        return decision

    return _override


class _TracingHud:
    """Tees every HUD update line into the trace stream (LOOP events)."""

    def __init__(self, inner: Hud, tracer: Any) -> None:
        self._inner = inner
        self._tracer = tracer

    def update(self, task_snapshot: dict[str, Any], log_line: str) -> None:
        self._tracer.event("LOOP", log_line)
        self._inner.update(task_snapshot, log_line)


def launch_debug_console(settings: Settings) -> None:
    """Opens a second Terminal window tailing the live trace stream.

    Best-effort by design: on the first run macOS may ask to allow the
    terminal to control Terminal.app (Automation permission); a refusal or
    any osascript failure just means no debug window -- the trace file is
    still written and can be tailed by hand:

        .venv/bin/python scripts/trace_view.py continuum-trace.log
    """
    import shlex
    import subprocess

    here = Path(__file__).resolve().parent
    python = here / ".venv" / "bin" / "python"
    viewer = here / "scripts" / "trace_view.py"
    trace_file = (here / settings.trace_path).resolve()
    command = (
        f"cd {shlex.quote(str(here))} && "
        f"{shlex.quote(str(python))} {shlex.quote(str(viewer))} {shlex.quote(str(trace_file))}"
    )
    # `activate` brings the window to the FRONT: without it the new window
    # opens behind whatever is focused and nobody ever sees it.
    script = (
        'tell application "Terminal"\n'
        f'  do script "{command}"\n'
        "  activate\n"
        "end tell"
    )
    try:
        subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, timeout=10.0, check=True
        )
    except Exception as error:  # noqa: BLE001 - a missing debug window is never fatal
        logger.warning("could not open the debug console window: %s", error)


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

    tracer: Any = Tracer(settings.trace_path) if settings.debug_console else NullTracer()
    if settings.debug_console:
        # Vision's own log lines (escalations, retries, breaker) join the
        # trace stream so the debug window tells the whole story.
        logging.getLogger("vision").addHandler(TraceLogHandler(tracer))
        launch_debug_console(settings)

    memory = MemoryStore(settings.db_path)
    mac = MacController(pyautogui_pause_s=settings.pyautogui_pause_s)
    vision = GeminiVision(settings)
    # Route 1 (official Computer Use) is only built when selected; in the
    # default grounding mode it stays None and build_ground_fn never touches it.
    cu = InteractionsComputerUse(settings) if settings.cu_mode == "interactions" else None
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
        tracer.event(
            "BOOT",
            f"resumed task {task.task_id} (session {task.session_count}, "
            f"{task.render()['progress']} steps done): {task.goal}",
        )
    else:
        task_id = uuid.uuid4().hex[:8]
        task = build_new_task(task_id, args.goal)
        memory.save_task_state(task)
        logger.info("Starting new task %s: %s", task.task_id, task.goal)
        tracer.event("BOOT", f"new task {task.task_id} -- hold the PTT key and speak")

    # Reattach the Computer Use session to whatever `--resume` reloaded (a new
    # task carries None, so this opens a fresh session on the first CU call).
    if cu is not None:
        cu.restore_session(task.interactions_session_id)

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
                observe_fn=build_observe_fn(stt_queue, mac, task, tracer),
                ground_fn=build_ground_fn(vision, mac, router, settings, cu, tracer),
                act_fn=build_act_fn(mac, settings),
                speak_fn=speaker.say,
                hud=_TracingHud(hud, tracer),
                stop_event=stop_event,
                max_turns=settings.max_turns,
                plan_fn=build_plan_fn(vision, tracer),
                # No separate verify_fn: completion is decided by the SAME
                # grounding call that picks the next action ({"action":"done"}),
                # so done-vs-act can never contradict and each turn costs one
                # model round-trip instead of two. Fast-path steps complete
                # deterministically on execution (completes_step).
                override_fn=build_override_fn(vision, tracer),
                idle_sleep_s=settings.loop_idle_sleep_s,
                max_idle_turns=settings.max_idle_turns,
                max_step_attempts=settings.max_step_attempts,
                keep_alive=settings.keep_alive,
            )
            summary = loop.run()
        finally:
            ptt_listener.stop()
            kill_switch.stop()

    logger.info("Run summary: %s", summary)
    tracer.event("BOOT", f"run ended: {summary}")
    tracer.close()
    memory.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
