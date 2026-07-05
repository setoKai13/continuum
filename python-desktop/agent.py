"""The OBSERVE -> UPDATE_STATE -> DECIDE -> ACT loop.

This is the causal chain the whole product hinges on: an action is only
ever selected via `TaskState.next_actionable_step()`, and a turn with no
new observation produces no action (stall), which is what proves the
hold-state -- not a screenshot snapshot -- is what drives behavior.

Every collaborator (how to observe, how to ground a step into a concrete
target, how to act on the Mac, how to speak, how to render the HUD) is
injected as a plain callable/object so this module has zero hard
dependency on pyautogui/mss/google-genai and can be exercised entirely by
`scripts/dry_run.py` with fakes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from config import DEFAULT_GOAL
from memory import MemoryStore
from router import is_dangerous
from state import Step, StepStatus, TaskState, TaskStatus


@dataclass
class Observation:
    """One OBSERVE result: what changed in the world since the last turn.

    Attributes:
        instruction: A new voice/text instruction, if the operator spoke or
            typed since the last turn.
        screenshot: Opaque screenshot handle (a real mss/PIL image in the
            live path, or any fake token in tests/dry-run).
        step_completed: Step id the observation confirms as finished
            (e.g. the screenshot shows the expected end state).
    """

    instruction: str | None = None
    screenshot: Any | None = None
    step_completed: str | None = None


@dataclass
class ActionPlan:
    """One ACT directive: what the actuator (mac_control) should perform.

    Attributes:
        kind: "click" | "type" | "open_app" | "open_url" | "hotkey" | "noop".
        step_id: The step this action serves.
        target: Action-specific target (e.g. (x, y) pixel tuple).
        text: Action-specific text payload (typed text, app name, URL...).
    """

    kind: str
    step_id: str
    target: Any | None = None
    text: str | None = None


@dataclass
class RunSummary:
    """Recap of one `AgentLoop.run()` call, printed at the end of a session.

    Attributes:
        turns: Total loop iterations executed.
        tool_calls: Number of real actuator calls made (ACT succeeded).
        stalled_turns: Turns where causality blocked any action.
        refused_turns: Turns where a dangerous instruction/plan was refused.
        steps_done: Steps marked done at the end of the run.
        steps_total: Total steps in the plan.
        status: Final TaskState status value.
    """

    turns: int
    tool_calls: int
    stalled_turns: int
    refused_turns: int
    steps_done: int
    steps_total: int
    status: str


class Hud(Protocol):
    """Minimal HUD contract: render a state snapshot plus a log line."""

    def update(self, task_snapshot: dict[str, Any], log_line: str) -> None: ...


ObserveFn = Callable[[], Observation | None]
GroundFn = Callable[[TaskState, Step, Observation], ActionPlan | None]
ActFn = Callable[[ActionPlan], dict[str, Any]]
SpeakFn = Callable[[str], None]
# Turns a natural-language instruction (+ optional screenshot) into an ordered
# list of concrete step descriptions. This is what makes a brand-new task
# actionable: without it, a fresh TaskState has no steps and the loop stalls.
PlanFn = Callable[[TaskState, str, Any], list[str]]
# Judges, from the current screenshot, whether the step the agent just acted on
# is now complete. This is what lets the loop advance across steps on its own
# instead of waiting for an external `step_completed` signal that never comes
# live.
VerifyFn = Callable[[TaskState, Step, Any], bool]
# Decides whether a mid-task instruction is a live CORRECTION and, if so,
# extracts the (when, rule) pair to feed `TaskState.apply_override`. Returning
# None means "plain instruction" (live: router.is_correction gate + Gemini
# extraction, wired by main.build_override_fn; a scripted fake in tests).
OverrideFn = Callable[[TaskState, str], tuple[str, str] | None]

# Safety net so a step the agent cannot complete (bad grounding, wrong screen)
# does not loop forever: after this many acts on the same step without it being
# verified done, the step is marked BLOCKED and the loop moves on.
MAX_STEP_ATTEMPTS = 3


class AgentLoop:
    """Runs the OBSERVE->UPDATE_STATE->DECIDE->ACT loop for one task.

    Attributes:
        task: The persistent hold-state driving this run.
        memory: SQLite-backed store for persistence, logs and trajectories.
    """

    def __init__(
        self,
        task: TaskState,
        memory: MemoryStore,
        observe_fn: ObserveFn,
        ground_fn: GroundFn,
        act_fn: ActFn,
        speak_fn: SpeakFn,
        hud: Hud,
        stop_event: threading.Event,
        max_turns: int,
        plan_fn: PlanFn | None = None,
        verify_fn: VerifyFn | None = None,
        override_fn: OverrideFn | None = None,
        idle_sleep_s: float = 0.0,
        max_idle_turns: int | None = None,
        max_step_attempts: int = MAX_STEP_ATTEMPTS,
        keep_alive: bool = False,
    ) -> None:
        """Wires an agent loop with all its collaborators injected.

        Args:
            task: The TaskState to drive (already loaded/resumed).
            memory: Persistence layer for state/log/trajectory writes.
            observe_fn: Zero-arg callable returning an Observation or None.
            ground_fn: Turns (task, step, observation) into an ActionPlan
                (this is where vision.py's Gemini call plugs in live).
            act_fn: Executes an ActionPlan on the real Mac (mac_control.py
                live, or a logging fake in the dry-run).
            speak_fn: One-arg callable used for spoken confirmations/refusals.
            hud: Object with an `update(snapshot, log_line)` method.
            stop_event: Kill-switch; checked at the START of every turn.
            max_turns: Hard ceiling on loop iterations.
            plan_fn: Optional planner turning a new instruction into steps
                (vision.py's Gemini planner live, a fake in tests). When
                omitted, an instruction only updates facts/overrides and the
                plan must already contain steps.
            verify_fn: Optional judge that, given the current screenshot,
                decides whether the step just acted on is complete. When
                omitted, the loop relies on external `step_completed`
                observations instead (used by the dry-run's scripted sensor).
            override_fn: Optional detector that decides whether a mid-task
                instruction is a live correction, returning the (when, rule)
                pair to apply. When omitted, every instruction is treated as
                a plain instruction (facts + planning only).
            idle_sleep_s: Sleep between empty OBSERVE polls (0 in tests so
                scripted runs stay instant; ~0.4s live so waiting for voice
                does not busy-spin).
            max_idle_turns: Ceiling on CONSECUTIVE observation-less turns
                before the run ends on its own. None defaults to `max_turns`
                (the historical behavior scripted tests rely on); live wiring
                passes a large value so the agent can wait minutes for voice.
            max_step_attempts: Acts allowed on one step before it is marked
                blocked (reset when an override recalibrates the step).
            keep_alive: When True, completing every step does NOT end the
                run: the task is marked done (and announced once), the loop
                keeps listening, and the next spoken instruction plans a
                fresh set of steps in the same session. The idle budget
                still bounds how long the agent waits in silence.
        """
        self.task = task
        self.memory = memory
        self._observe_fn = observe_fn
        self._ground_fn = ground_fn
        self._act_fn = act_fn
        self._speak_fn = speak_fn
        self._hud = hud
        self._stop_event = stop_event
        self._max_turns = max_turns
        self._plan_fn = plan_fn
        self._verify_fn = verify_fn
        self._override_fn = override_fn
        self._idle_sleep_s = idle_sleep_s
        self._max_idle_turns = max_idle_turns
        self._max_step_attempts = max_step_attempts
        self._keep_alive = keep_alive
        self._attempts: dict[str, int] = {}

    def run(self) -> RunSummary:
        """Executes the loop until done, paused (kill-switch), or budget spent.

        Budget semantics: `max_turns` caps WORK turns (turns that carry an
        observation -- the ones that may hit Gemini or the actuator). Turns
        with nothing to observe are free but sleep `idle_sleep_s` and are
        bounded by `max_idle_turns` CONSECUTIVE occurrences, so the live
        agent can wait quietly for the operator's voice without busy-spinning
        through its budget in milliseconds.

        Returns:
            A `RunSummary` recap suitable for printing/asserting on.
        """
        tool_calls = 0
        stalled_turns = 0
        refused_turns = 0
        turn = 0
        work_turns = 0
        idle_streak = 0
        max_idle = self._max_idle_turns if self._max_idle_turns is not None else self._max_turns

        self.memory.open_trajectory(self.task.task_id)
        try:
            while work_turns < self._max_turns:
                turn += 1

                if self._stop_event.is_set():
                    self.task.status = TaskStatus.PAUSED
                    self.memory.save_task_state(self.task)
                    self._log(turn, f"kill-switch engaged -> paused ({self.task.task_id})")
                    break

                if self.task.is_complete():
                    if self.task.status != TaskStatus.DONE:
                        self.task.status = TaskStatus.DONE
                        self.memory.save_task_state(self.task)
                        self._log(turn, "all steps done -> task complete")
                        if self._keep_alive:
                            self._speak_fn("Task complete. Listening for the next one.")
                    if not self._keep_alive:
                        break

                observation = self._observe_fn()
                self.memory.append_trajectory(
                    self.task.task_id,
                    {"turn": turn, "type": "observe", "has_observation": observation is not None},
                )

                if observation is None:
                    stalled_turns += 1
                    idle_streak += 1
                    self._log(turn, "no new observation -> stall (causality holds)")
                    if idle_streak >= max_idle:
                        self._log(turn, f"idle for {idle_streak} turn(s) -> ending run (state persists)")
                        break
                    if self._idle_sleep_s > 0:
                        time.sleep(self._idle_sleep_s)
                    continue

                idle_streak = 0
                work_turns += 1
                outcome = self._run_work_turn(turn, observation)
                if outcome == "refused":
                    refused_turns += 1
                elif outcome == "stalled":
                    stalled_turns += 1
                elif outcome == "acted":
                    tool_calls += 1
        finally:
            self.memory.close_trajectory(self.task.task_id)

        steps_done, steps_total = self.task.progress()
        return RunSummary(
            turns=turn,
            tool_calls=tool_calls,
            stalled_turns=stalled_turns,
            refused_turns=refused_turns,
            steps_done=steps_done,
            steps_total=steps_total,
            status=self.task.status.value,
        )

    def _run_work_turn(self, turn: int, observation: Observation) -> str:
        """Processes one turn that carries a fresh observation.

        Args:
            turn: Current loop iteration (for logs).
            observation: This turn's OBSERVE result.

        Returns:
            The turn outcome: "refused", "stalled", "advanced", "blocked"
            or "acted" (run() maps these onto its counters).
        """
        if self._update_state(observation, turn):
            self._log(turn, "refused dangerous instruction, no action taken")
            return "refused"

        step = self.task.next_actionable_step()
        if step is None:
            self._log(turn, "no actionable step -> stall")
            return "stalled"

        # A step we already acted on: ask whether the screen now shows it
        # done, so the loop advances to the next step on its own instead
        # of re-acting the same one forever.
        if step.status == StepStatus.DOING and self._verify_completed(step, observation):
            self.task.mark_step(step.id, StepStatus.DONE)
            self.memory.save_task_state(self.task)
            self._log(turn, f"step {step.id} verified done -> next")
            return "advanced"

        self._attempts[step.id] = self._attempts.get(step.id, 0) + 1
        if self._attempts[step.id] > self._max_step_attempts:
            self.task.mark_step(step.id, StepStatus.BLOCKED)
            self.task.add_open_question(f"Could not complete step {step.id}: {step.desc}")
            self.memory.save_task_state(self.task)
            self._log(turn, f"step {step.id} blocked after {self._max_step_attempts} attempts -> next")
            return "blocked"

        if step.status == StepStatus.TODO:
            self.task.mark_step(step.id, StepStatus.DOING)

        plan = self._ground_fn(self.task, step, observation)
        if plan is None:
            self._log(turn, f"vision returned no plan for step {step.id} -> stall")
            return "stalled"

        if self._plan_is_dangerous(plan):
            self._speak_fn("Refused: that action looks destructive.")
            self.memory.log_tool(self.task.task_id, turn, "refuse", {"plan_kind": plan.kind})
            self._log(turn, "refused dangerous action plan, no action taken")
            return "refused"

        return self._act(turn, step, plan)

    def _act(self, turn: int, step: Step, plan: ActionPlan) -> str:
        """Executes one ActionPlan, absorbing actuator failures.

        A single failed action (app not found, UI element gone) must degrade
        into a logged, attempt-counted miss -- never a process crash. The
        pyautogui fail-safe corner is the exception: slamming the mouse into
        a screen corner IS the operator's emergency gesture, so it is honored
        as a kill-switch.

        Args:
            turn: Current loop iteration.
            step: The step this action serves.
            plan: The action to execute.

        Returns:
            "acted" on success, "stalled" on an absorbed failure.
        """
        try:
            result = self._act_fn(plan)
        except Exception as error:  # noqa: BLE001 - one failed action must not kill the run
            if type(error).__name__ == "FailSafeException":
                self._stop_event.set()
                self._log(turn, "fail-safe corner hit -> engaging kill-switch")
                return "stalled"
            self.memory.log_tool(
                self.task.task_id, turn, "act_error", {"kind": plan.kind, "error": str(error)}
            )
            self.memory.save_task_state(self.task)
            self._log(turn, f"action [{plan.kind}] failed on step {step.id}: {error}")
            return "stalled"

        self.memory.log_tool(self.task.task_id, turn, plan.kind, result)
        self.memory.append_trajectory(
            self.task.task_id, {"turn": turn, "type": "act", "kind": plan.kind, "step_id": step.id}
        )
        self.memory.save_task_state(self.task)
        self._log(turn, f"acted [{plan.kind}] on step {step.id}")
        return "acted"

    @staticmethod
    def _plan_is_dangerous(plan: ActionPlan) -> bool:
        """Gates the EXECUTABLE payload of a plan, not the model's reasoning.

        Only kinds whose text is actually executed (typed, or handed to
        `open`) are screened: for a click, `text` carries the model's
        free-form rationale, and a mention of "wipe"/"sudo" inside an
        EXPLANATION must not veto a legitimate click.

        Args:
            plan: The action plan to screen.

        Returns:
            True if the plan's executable text matches the blocklist.
        """
        if plan.kind not in ("type", "open_app", "open_url"):
            return False
        return bool(plan.text) and is_dangerous(plan.text)

    def _update_state(self, observation: Observation, turn: int) -> bool:
        """Folds one Observation into the TaskState. Returns True if refused.

        Args:
            observation: The OBSERVE result for this turn.
            turn: Current loop iteration (for the audit log).

        Returns:
            True if the observation carried a dangerous instruction that
            was refused (in which case DECIDE/ACT must not run this turn).
        """
        if observation.instruction:
            if is_dangerous(observation.instruction):
                self._speak_fn("I won't do that, it looks destructive.")
                self.memory.log_tool(
                    self.task.task_id, turn, "refuse", {"instruction": observation.instruction}
                )
                return True
            self.task.add_fact(f"heard: {observation.instruction}")
            if not self._apply_live_override(observation.instruction, turn):
                self._maybe_plan(observation)

        if observation.step_completed:
            try:
                self.task.mark_step(observation.step_completed, StepStatus.DONE)
            except KeyError:
                # A stale/mistyped external completion signal must not kill
                # the run; record it and keep the loop alive.
                self.task.add_fact(f"ignored unknown step_completed: {observation.step_completed}")

        return False

    def _apply_live_override(self, instruction: str, turn: int) -> bool:
        """Folds a mid-task correction into the hold-state. Returns True if applied.

        This is the "learns from human corrections" half of the hold-state:
        when the override detector recognizes the phrase as a correction, the
        (when, rule) pair recalibrates every remaining matching step via
        `TaskState.apply_override`, recalibrated steps get a fresh attempt
        budget (a blocked step returns to `todo` in state.py, so it must not
        instantly re-block on a stale counter), and the plan is NOT re-planned
        from the phrase.

        Args:
            instruction: The instruction heard this turn.
            turn: Current loop iteration (for the audit log).

        Returns:
            True if the instruction was consumed as an override (the caller
            must then skip planning); False to treat it as a plain instruction.
        """
        if self._override_fn is None or not self.task.steps:
            return False
        decision = self._override_fn(self.task, instruction)
        if decision is None:
            return False
        when, rule = decision
        recalibrated = self.task.apply_override(when, rule)
        for step in recalibrated:
            self._attempts.pop(step.id, None)
        self.task.add_fact(f"override: {rule} ({len(recalibrated)} step(s) recalibrated)")
        self.memory.log_tool(
            self.task.task_id,
            turn,
            "override",
            {"when": when, "rule": rule, "recalibrated": len(recalibrated)},
        )
        self.memory.save_task_state(self.task)
        self._speak_fn(f"Correction noted: {rule}")
        return True

    def _maybe_plan(self, observation: Observation) -> None:
        """Grows the plan from a fresh instruction when nothing is actionable.

        This is the bridge from "the operator said something" to "there are
        concrete steps to act on". It only runs when a planner is wired AND
        the plan currently has no actionable step, so a mid-task instruction
        does not silently rewrite an in-progress plan (that path is an
        override, handled separately).

        Args:
            observation: The OBSERVE result carrying the instruction and,
                optionally, the current screenshot for visual planning.
        """
        if self._plan_fn is None or self.task.next_actionable_step() is not None:
            return
        instruction = observation.instruction or ""
        # The first spoken instruction also becomes the task goal, so the
        # operator states the objective live by voice rather than pre-scripting
        # it on the command line (the `--goal` flag is just an optional label).
        if instruction and self.task.goal.strip().lower() in ("", DEFAULT_GOAL.lower()):
            self.task.goal = instruction
        descriptions = self._plan_fn(self.task, instruction, observation.screenshot)
        planned = self.task.add_steps(descriptions)
        if planned:
            self.task.add_fact(f"planned {len(planned)} step(s) from instruction")
            # Keep-alive revival: a completed task that receives a new
            # instruction becomes active again, with that instruction as the
            # new goal (the previous goal is already served, its steps stay
            # done in the history).
            if self.task.status == TaskStatus.DONE:
                self.task.status = TaskStatus.ACTIVE
                if instruction:
                    self.task.goal = instruction

    def _verify_completed(self, step: Step, observation: Observation) -> bool:
        """Returns True if a verifier confirms `step` is done from the screen.

        With no verifier wired (the dry-run's scripted path) or no screenshot
        this turn, completion cannot be judged here and the loop falls back to
        an external `step_completed` observation.

        Args:
            step: The step currently in progress.
            observation: This turn's observation (its screenshot is the frame
                the verifier inspects).

        Returns:
            True only if a verifier is present, a screenshot exists, and the
            verifier judges the step complete.
        """
        if self._verify_fn is None or observation.screenshot is None:
            return False
        return bool(self._verify_fn(self.task, step, observation.screenshot))

    def _log(self, turn: int, message: str) -> None:
        """Pushes one line to the HUD's tool-call/event stream.

        Args:
            turn: Current loop iteration.
            message: Human-readable event description.
        """
        self._hud.update(self.task.render(), f"[turn {turn}] {message}")
