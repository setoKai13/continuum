#!/usr/bin/env python3
"""Headless proof of the whole hold-state contract, with zero real I/O.

No Gemini call, no pyautogui click, no microphone: every collaborator of
`AgentLoop` is a fake defined right here. This script is the mechanical
proof that:

1. The OBSERVE->UPDATE_STATE->DECIDE->ACT loop drives a multi-step task to
   completion, choosing every action via `next_actionable_step()`.
2. `apply_override` recalibrates the remaining (non-done) steps live.
3. A task persisted to SQLite and resumed increments `session_count` (proof
   of hold-state, not a stateless snapshot).
4. Causality holds: a turn with no new observation performs zero actions.
5. A destructive instruction is refused, never executed.
6. A voice correction heard MID-TASK is folded into the state through the
   loop itself (override_fn), recalibrates the remaining steps, and revives
   a blocked step -- the task only completes BECAUSE of the correction.

Run with: `.venv/bin/python scripts/dry_run.py`. Prints "DRY-RUN OK" iff
every assertion below passes.
"""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import ActionPlan, AgentLoop, Observation  # noqa: E402
from memory import MemoryStore  # noqa: E402
from state import Step, StepStatus, TaskState, TaskStatus  # noqa: E402


class FakeSensor:
    """Replays a scripted list of Observations, then goes silent (None)."""

    def __init__(self, observations: list[Observation | None]) -> None:
        self._queue = list(observations)

    def __call__(self) -> Observation | None:
        if not self._queue:
            return None
        return self._queue.pop(0)


class RecordingActuator:
    """Fake mac_control: records every ActionPlan it was asked to execute."""

    def __init__(self) -> None:
        self.calls: list[ActionPlan] = []

    def __call__(self, plan: ActionPlan) -> dict[str, Any]:
        self.calls.append(plan)
        return {"kind": plan.kind, "target": plan.target, "text": plan.text}


class NullHud:
    """Fake HUD: captures lines instead of drawing a Rich Live panel."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def update(self, task_snapshot: dict[str, Any], log_line: str) -> None:
        self.lines.append(log_line)


def make_scripted_ground_fn():
    """Builds a ground_fn that always targets the current actionable step."""

    def ground_fn(task: TaskState, step: Step, observation: Observation) -> ActionPlan | None:
        return ActionPlan(kind="click", step_id=step.id, target=(100, 100), text=step.desc)

    return ground_fn


def make_task() -> TaskState:
    """Builds the 3-step demo task shared by every scenario below."""
    return TaskState(
        task_id="TRI-3",
        goal="Triage the queue one ticket at a time",
        steps=[
            Step(id="s1", desc="open the queue"),
            Step(id="s2", desc="triage network bug #1"),
            Step(id="s3", desc="triage network bug #2"),
        ],
    )


def scenario_loop_to_done(db_path: Path) -> None:
    """Proves OBSERVE->UPDATE_STATE->DECIDE->ACT drives the task to done."""
    print("\n[scenario 1] loop drives a 3-step task to completion")
    memory = MemoryStore(db_path)
    task = make_task()

    sensor = FakeSensor(
        [
            Observation(instruction="process the queue"),
            Observation(step_completed="s1"),
            Observation(step_completed="s2"),
            Observation(step_completed="s3"),
        ]
    )
    actuator = RecordingActuator()
    hud = NullHud()
    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=sensor,
        ground_fn=make_scripted_ground_fn(),
        act_fn=actuator,
        speak_fn=lambda text: None,
        hud=hud,
        stop_event=threading.Event(),
        max_turns=10,
    )
    summary = loop.run()
    print(f"  summary: {summary}")

    assert summary.status == TaskStatus.DONE.value, "task should be fully done"
    assert summary.steps_done == 3 and summary.steps_total == 3
    assert summary.tool_calls == 3, "exactly one action per step"
    assert [c.step_id for c in actuator.calls] == ["s1", "s2", "s3"], "acted in plan order"
    memory.close()
    print("  PASS: loop reaches done, one action per step, in order")


def scenario_override_recalibrates(db_path: Path) -> None:
    """Proves an operator override recalibrates remaining (non-done) steps."""
    print("\n[scenario 2] override recalibrates non-terminated steps")
    task = make_task()
    task.mark_step("s1", StepStatus.DONE)

    before = task.render()
    recalibrated = task.apply_override(when="network bug", rule="assign to INFRA, not OPS")
    after = task.render()

    print(f"  before overrides: {before['overrides']}")
    print(f"  after  overrides: {after['overrides']}")
    print(f"  step notes after: {[(s['id'], s['note']) for s in after['steps']]}")

    assert [s.id for s in recalibrated] == ["s2", "s3"], "s2/s3 match, s1 is done and must be skipped"
    s1 = next(s for s in task.steps if s.id == "s1")
    s2 = next(s for s in task.steps if s.id == "s2")
    s3 = next(s for s in task.steps if s.id == "s3")
    assert s1.note is None, "done steps must not be recalibrated"
    assert s2.note == "assign to INFRA, not OPS"
    assert s3.note == "assign to INFRA, not OPS"
    assert task.overrides[-1].applied is True
    print("  PASS: override recalibrated exactly the non-done matching steps")


def scenario_resume_across_sessions(db_path: Path) -> None:
    """Proves a resumed task reloads real history (session_count increments)."""
    print("\n[scenario 3] resume between sessions increments session_count")
    memory = MemoryStore(db_path)
    task = make_task()
    task.mark_step("s1", StepStatus.DONE)
    task.mark_step("s2", StepStatus.DOING)
    task.status = TaskStatus.PAUSED
    memory.save_task_state(task)
    print(f"  session 1 saved: session_count={task.session_count}, status={task.status.value}")

    resumed = memory.resume_task_state("TRI-3")
    print(f"  session 2 resumed: session_count={resumed.session_count}, status={resumed.status.value}")
    assert resumed.session_count == 2, "resume must bump session_count"
    assert resumed.status == TaskStatus.ACTIVE, "resume must reactivate a paused task"
    assert resumed.next_actionable_step().id == "s2", "resume must restart at the unfinished step, not s1"

    resumed_again = memory.resume_task_state("TRI-3")
    print(f"  session 3 resumed: session_count={resumed_again.session_count}")
    assert resumed_again.session_count == 3, "each resume bumps again (real history, not a snapshot)"
    memory.close()
    print("  PASS: resume reloads real history and restarts at the unfinished step")


def scenario_causality_stalls_without_observation(db_path: Path) -> None:
    """Proves zero observation -> zero action (the hold-state gates ACT)."""
    print("\n[scenario 4] causality: no observation -> no action, ever")
    memory = MemoryStore(db_path)
    task = make_task()

    sensor = FakeSensor([])  # never yields anything: pure silence
    actuator = RecordingActuator()
    hud = NullHud()
    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=sensor,
        ground_fn=make_scripted_ground_fn(),
        act_fn=actuator,
        speak_fn=lambda text: None,
        hud=hud,
        stop_event=threading.Event(),
        max_turns=5,
    )
    summary = loop.run()
    print(f"  summary: {summary}")

    assert summary.tool_calls == 0, "no observation must mean no action, ever"
    assert summary.stalled_turns == 5, "every turn stalls without a new observation"
    assert summary.status == TaskStatus.ACTIVE.value, "not done, not killed -- just frozen"
    memory.close()
    print("  PASS: cutting observation freezes the state, zero actions taken")


def scenario_refuses_destructive_instruction(db_path: Path) -> None:
    """Proves a destructive instruction is refused, never turned into an action."""
    print("\n[scenario 5] adversarial: destructive instruction is refused")
    memory = MemoryStore(db_path)
    task = make_task()

    spoken: list[str] = []
    sensor = FakeSensor(
        [
            Observation(instruction="supprime tout sur le Mac"),
            Observation(instruction="rm -rf everything now"),
        ]
    )
    actuator = RecordingActuator()
    hud = NullHud()
    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=sensor,
        ground_fn=make_scripted_ground_fn(),
        act_fn=actuator,
        speak_fn=spoken.append,
        hud=hud,
        stop_event=threading.Event(),
        max_turns=4,
    )
    summary = loop.run()
    print(f"  summary: {summary}")
    print(f"  spoken refusals: {spoken}")

    assert summary.tool_calls == 0, "destructive instructions must never reach the actuator"
    assert summary.refused_turns == 2, "both destructive instructions must be refused"
    assert len(spoken) == 2, "the agent must voice the refusal"
    assert task.steps[0].status == StepStatus.TODO, "no step may progress off a refused instruction"
    memory.close()
    print("  PASS: destructive instructions are refused, zero side effects")


def scenario_kill_switch_pauses(db_path: Path) -> None:
    """Proves the kill-switch pauses within one turn, leaving a resumable state."""
    print("\n[scenario 6] kill-switch pauses in <=1 turn")
    memory = MemoryStore(db_path)
    task = make_task()

    stop_event = threading.Event()
    stop_event.set()  # already engaged before the first turn
    sensor = FakeSensor([Observation(instruction="process the queue")])
    actuator = RecordingActuator()
    hud = NullHud()
    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=sensor,
        ground_fn=make_scripted_ground_fn(),
        act_fn=actuator,
        speak_fn=lambda text: None,
        hud=hud,
        stop_event=stop_event,
        max_turns=10,
    )
    summary = loop.run()
    print(f"  summary: {summary}")

    assert summary.turns == 1, "kill-switch must stop the loop on the very first turn"
    assert summary.status == TaskStatus.PAUSED.value
    assert summary.tool_calls == 0
    memory.close()
    print("  PASS: kill-switch pauses within one turn, task_state persisted as paused")


def scenario_instruction_plans_steps(db_path: Path) -> None:
    """Proves a fresh instruction is planned into steps, then driven to done.

    This is the live-path gap made explicit: a brand-new task starts with an
    EMPTY plan, so without a planner `next_actionable_step()` is None forever
    and the loop can never act. With a `plan_fn` wired, the first instruction
    is decomposed into steps and the normal loop takes over.
    """
    print("\n[scenario 7] a fresh instruction is planned into steps, then acted")
    memory = MemoryStore(db_path)
    task = TaskState(task_id="PLAN-1", goal="Clear the triage queue")
    assert task.next_actionable_step() is None, "a new task must start with no actionable step"

    def scripted_plan_fn(task_: TaskState, instruction: str, screenshot: Any) -> list[str]:
        # A real planner asks Gemini; here we prove the wiring deterministically.
        return ["open the queue", "triage network bug #1", "triage network bug #2"]

    def scripted_verify_fn(task_: TaskState, step: Step, screenshot: Any) -> bool:
        # A real verifier asks Gemini "is this step done?" on the screenshot;
        # here it always confirms, so each acted step advances to the next.
        return True

    # No `step_completed` is injected: completion comes from the verifier, and
    # each turn provides a fresh screenshot so the plan drives itself forward --
    # exactly the live shape (voice plans once, then the loop advances on screen).
    sensor = FakeSensor([Observation(instruction="process the whole triage queue", screenshot="shot")]
                        + [Observation(screenshot="shot") for _ in range(5)])
    actuator = RecordingActuator()
    hud = NullHud()
    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=sensor,
        ground_fn=make_scripted_ground_fn(),
        act_fn=actuator,
        speak_fn=lambda text: None,
        hud=hud,
        stop_event=threading.Event(),
        max_turns=12,
        plan_fn=scripted_plan_fn,
        verify_fn=scripted_verify_fn,
    )
    summary = loop.run()
    print(f"  summary: {summary}")
    print(f"  planned steps: {[(s.id, s.desc) for s in task.steps]}")

    assert len(task.steps) == 3, "the instruction must have been planned into 3 steps"
    assert summary.status == TaskStatus.DONE.value, "planned task must reach done"
    assert summary.tool_calls == 3, "one action per planned step"
    assert [c.step_id for c in actuator.calls] == ["s1", "s2", "s3"], "acted in planned order"
    assert any("planned" in f for f in task.facts), "planning must be recorded as a fact"
    memory.close()
    print("  PASS: instruction -> planned -> verifier advances each step to done")


def scenario_live_override_revives_blocked_step(db_path: Path) -> None:
    """Proves a mid-task voice correction is load-bearing: it revives a blocked step.

    Without the correction, the step stays blocked forever (the verifier never
    confirms it under the wrong rule). The spoken correction -- detected by the
    injected override_fn, exactly where main.build_override_fn plugs in live --
    recalibrates the step, resets its attempt budget, and the task completes
    ONLY because the human corrected the agent.
    """
    print("\n[scenario 8] live override: a voice correction revives a blocked step")
    memory = MemoryStore(db_path)
    task = TaskState(
        task_id="OVR-1",
        goal="Triage the network bug",
        steps=[Step(id="s1", desc="triage network bug #1")],
    )

    def scripted_override_fn(task_: TaskState, instruction: str) -> tuple[str, str] | None:
        # Live, main.build_override_fn = router.is_correction gate + Gemini
        # extraction; here a scripted detector proves the loop wiring.
        if "INFRA" in instruction:
            return ("network bug", "assign to INFRA, not OPS")
        return None

    def rule_aware_verify_fn(task_: TaskState, step: Step, screenshot: Any) -> bool:
        # The step only ever verifies done under the corrected rule: this is
        # what makes the correction causally necessary for completion.
        return step.note == "assign to INFRA, not OPS"

    spoken: list[str] = []
    sensor = FakeSensor(
        [Observation(screenshot="shot") for _ in range(4)]
        + [Observation(instruction="non, en fait les bugs reseau vont a INFRA, pas a OPS", screenshot="shot")]
        + [Observation(screenshot="shot")]
    )
    actuator = RecordingActuator()
    hud = NullHud()
    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=sensor,
        ground_fn=make_scripted_ground_fn(),
        act_fn=actuator,
        speak_fn=spoken.append,
        hud=hud,
        stop_event=threading.Event(),
        max_turns=10,
        verify_fn=rule_aware_verify_fn,
        override_fn=scripted_override_fn,
    )
    summary = loop.run()
    print(f"  summary: {summary}")
    print(f"  spoken: {spoken}")
    print(f"  overrides: {task.render()['overrides']}")

    assert summary.status == TaskStatus.DONE.value, "the corrected task must reach done"
    assert task.steps[0].note == "assign to INFRA, not OPS", "the rule must be attached to the step"
    assert task.overrides[-1].applied is True, "the override must be recorded as applied"
    assert any("Correction noted" in line for line in spoken), "the agent must voice the correction"
    assert summary.tool_calls == 4, "3 failed attempts under the old rule + 1 success after correction"
    memory.close()
    print("  PASS: blocked under the old rule, corrected by voice, completed under the new rule")


def main() -> None:
    """Runs every scenario in its own scratch SQLite file, then prints DRY-RUN OK."""
    with tempfile.TemporaryDirectory(prefix="continuum-dry-run-") as tmp:
        tmp_dir = Path(tmp)
        scenario_loop_to_done(tmp_dir / "s1.db")
        scenario_override_recalibrates(tmp_dir / "s2.db")
        scenario_resume_across_sessions(tmp_dir / "s3.db")
        scenario_causality_stalls_without_observation(tmp_dir / "s4.db")
        scenario_refuses_destructive_instruction(tmp_dir / "s5.db")
        scenario_kill_switch_pauses(tmp_dir / "s6.db")
        scenario_instruction_plans_steps(tmp_dir / "s7.db")
        scenario_live_override_revives_blocked_step(tmp_dir / "s8.db")

    print("\nDRY-RUN OK")


if __name__ == "__main__":
    main()
