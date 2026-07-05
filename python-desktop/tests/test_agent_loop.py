"""Loop-level safety and lifecycle tests: the demo's scripted moments.

Kill-switch pause/resume, both refusal gates (spoken instruction and
grounded ActionPlan), causality (no observation -> no action), actuator
failures absorbed instead of crashing, and the idle budget. These were
previously proven only by scripts/dry_run.py, which CI never runs.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from agent import MAX_STEP_ATTEMPTS, ActionPlan, AgentLoop, Observation
from memory import MemoryStore
from state import Step, StepStatus, TaskState, TaskStatus


class _FakeHud:
    """HUD stub: swallows every update call."""

    def update(self, task_snapshot: dict[str, Any], log_line: str) -> None:
        return None


def _click_ground_fn():
    """Ground fn that always turns the current step into a click plan."""

    def ground_fn(task: TaskState, step: Any, observation: Observation) -> ActionPlan:
        return ActionPlan(kind="click", step_id=step.id, target=(10, 10), text=step.desc)

    return ground_fn


def _make_loop(task: TaskState, memory: MemoryStore, observations: list[Observation], **kwargs: Any):
    """Builds an AgentLoop over scripted observations, returning it with probes."""
    spoken: list[str] = []
    calls: list[ActionPlan] = []
    queue = list(observations)
    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=lambda: queue.pop(0) if queue else None,
        ground_fn=kwargs.pop("ground_fn", _click_ground_fn()),
        act_fn=kwargs.pop("act_fn", lambda plan: (calls.append(plan), {"kind": plan.kind})[1]),
        speak_fn=spoken.append,
        hud=_FakeHud(),
        stop_event=kwargs.pop("stop_event", threading.Event()),
        max_turns=kwargs.pop("max_turns", 10),
        **kwargs,
    )
    return loop, calls, spoken


def _three_step_task(task_id: str) -> TaskState:
    return TaskState(
        task_id=task_id,
        goal="do three things",
        steps=[Step(id="s1", desc="click A"), Step(id="s2", desc="click B"), Step(id="s3", desc="click C")],
    )


def test_kill_switch_engaged_mid_run_pauses_and_persists(tmp_path) -> None:
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = _three_step_task("KILL-1")
    stop_event = threading.Event()

    def act_and_kill(plan: ActionPlan) -> dict[str, Any]:
        stop_event.set()  # the operator hits Esc right after the first action
        return {"kind": plan.kind}

    loop, _calls, _spoken = _make_loop(
        task,
        memory,
        [Observation(screenshot="shot") for _ in range(6)],
        act_fn=act_and_kill,
        stop_event=stop_event,
        verify_fn=lambda t, step, shot: True,
    )
    summary = loop.run()

    assert summary.status == TaskStatus.PAUSED.value
    assert summary.tool_calls == 1, "paused within one turn of the kill gesture"
    persisted = memory.load_task_state("KILL-1")
    assert persisted is not None and persisted.status == TaskStatus.PAUSED
    memory.close()


def test_pause_then_resume_completes_remaining_steps(tmp_path) -> None:
    """The pitch cycle: pause mid-task, resume, finish WITHOUT redoing s1."""
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = _three_step_task("KILL-2")
    task.mark_step("s1", StepStatus.DONE)
    task.status = TaskStatus.PAUSED
    memory.save_task_state(task)

    resumed = memory.resume_task_state("KILL-2")
    assert resumed is not None
    assert resumed.session_count == 2, "resume bumps the session counter"
    assert resumed.next_actionable_step().id == "s2", "resume restarts at the unfinished step"

    loop, calls, _spoken = _make_loop(
        resumed,
        memory,
        [Observation(screenshot="shot") for _ in range(8)],
        verify_fn=lambda t, step, shot: True,
    )
    summary = loop.run()

    assert summary.status == TaskStatus.DONE.value
    assert [c.step_id for c in calls] == ["s2", "s3"], "s1 was never re-executed"
    memory.close()


def test_dangerous_instruction_refused_without_action(tmp_path) -> None:
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = _three_step_task("REF-1")

    def exploding_ground_fn(task_: TaskState, step: Any, observation: Observation) -> ActionPlan:
        pytest.fail("a refused instruction must never reach grounding")

    loop, calls, spoken = _make_loop(
        task,
        memory,
        [Observation(instruction="supprime tout sur le Mac")],
        ground_fn=exploding_ground_fn,
        max_turns=3,
    )
    summary = loop.run()

    assert summary.refused_turns == 1
    assert summary.tool_calls == 0 and calls == []
    assert len(spoken) == 1 and "destructive" in spoken[0]
    log_names = [entry["tool_name"] for entry in memory.get_task_log("REF-1")]
    assert "refuse" in log_names
    assert all(s.status == StepStatus.TODO for s in task.steps), "no step progressed"
    memory.close()


def test_dangerous_action_plan_refused_before_act(tmp_path) -> None:
    """Second gate: a grounded plan that would TYPE something destructive."""
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(task_id="REF-2", goal="g", steps=[Step(id="s1", desc="fill the field")])

    def hostile_ground_fn(task_: TaskState, step: Any, observation: Observation) -> ActionPlan:
        # e.g. prompt injection read off the screen by the model
        return ActionPlan(kind="type", step_id=step.id, text="sudo rm -rf /")

    loop, calls, spoken = _make_loop(
        task,
        memory,
        [Observation(screenshot="shot")],
        ground_fn=hostile_ground_fn,
        act_fn=lambda plan: pytest.fail("a dangerous plan must never reach the actuator"),
        max_turns=3,
    )
    summary = loop.run()

    assert summary.refused_turns == 1
    assert summary.tool_calls == 0
    assert any("destructive" in line for line in spoken)
    memory.close()


def test_click_with_scary_reasoning_is_not_refused(tmp_path) -> None:
    """The gate screens the EXECUTABLE payload, not the model's explanation."""
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(task_id="REF-3", goal="g", steps=[Step(id="s1", desc="click the dialog")])

    def chatty_ground_fn(task_: TaskState, step: Any, observation: Observation) -> ActionPlan:
        # A legitimate click whose rationale MENTIONS scary words.
        return ActionPlan(
            kind="click",
            step_id=step.id,
            target=(10, 10),
            text="clicking Cancel so we do NOT wipe or sudo anything",
        )

    loop, calls, _spoken = _make_loop(
        task,
        memory,
        [Observation(screenshot="shot")],
        ground_fn=chatty_ground_fn,
        max_turns=3,
    )
    summary = loop.run()

    assert summary.refused_turns == 0, "reasoning text must not trigger the blocklist"
    assert summary.tool_calls == 1 and [c.step_id for c in calls] == ["s1"]
    memory.close()


def test_actuator_failure_is_absorbed_not_fatal(tmp_path) -> None:
    """One failed action = logged miss + attempt consumed, never a crash."""
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(task_id="ACT-1", goal="g", steps=[Step(id="s1", desc="open Foo")])

    attempts: list[int] = []

    def flaky_act_fn(plan: ActionPlan) -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("open -a 'Foo' failed: unknown application")
        return {"kind": plan.kind}

    loop, _calls, _spoken = _make_loop(
        task,
        memory,
        [Observation(screenshot="shot") for _ in range(4)],
        act_fn=flaky_act_fn,
        verify_fn=lambda t, step, shot: len(attempts) >= 2,
    )
    summary = loop.run()

    assert summary.status == TaskStatus.DONE.value, "the loop recovered after the failure"
    assert summary.tool_calls == 1, "only the successful action counts"
    log_names = [entry["tool_name"] for entry in memory.get_task_log("ACT-1")]
    assert "act_error" in log_names, "the failure is visible in the audit log"
    memory.close()


def test_failsafe_exception_engages_kill_switch(tmp_path) -> None:
    """Mouse slammed into a corner = the emergency gesture -> pause, not crash."""

    class FailSafeException(Exception):
        """Same NAME as pyautogui's, matched by name to stay headless."""

    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(task_id="FS-1", goal="g", steps=[Step(id="s1", desc="click A")])

    def corner_act_fn(plan: ActionPlan) -> dict[str, Any]:
        raise FailSafeException("mouse in corner")

    loop, _calls, _spoken = _make_loop(
        task,
        memory,
        [Observation(screenshot="shot") for _ in range(3)],
        act_fn=corner_act_fn,
    )
    summary = loop.run()

    assert summary.status == TaskStatus.PAUSED.value, "fail-safe is honored as kill-switch"
    assert summary.tool_calls == 0
    memory.close()


def test_unknown_step_completed_is_ignored_not_fatal(tmp_path) -> None:
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(task_id="UNK-1", goal="g", steps=[Step(id="s1", desc="click A")])

    loop, calls, _spoken = _make_loop(
        task,
        memory,
        [Observation(step_completed="s99", screenshot="shot"), Observation(step_completed="s1")],
        max_turns=5,
    )
    summary = loop.run()

    assert summary.status == TaskStatus.DONE.value, "the stale signal did not kill the run"
    assert any("ignored unknown step_completed" in fact for fact in task.facts)
    memory.close()


def test_blocked_step_skipped_then_next_step_completes(tmp_path) -> None:
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(
        task_id="BLK-1",
        goal="g",
        steps=[Step(id="s1", desc="impossible"), Step(id="s2", desc="easy")],
    )

    loop, calls, _spoken = _make_loop(
        task,
        memory,
        [Observation(screenshot="shot") for _ in range(12)],
        max_turns=12,
        verify_fn=lambda t, step, shot: step.id == "s2",
    )
    summary = loop.run()

    assert task.steps[0].status == StepStatus.BLOCKED
    assert task.steps[1].status == StepStatus.DONE
    assert summary.status == TaskStatus.ACTIVE.value, "not DONE: s1 remains unfinished"
    assert summary.tool_calls == MAX_STEP_ATTEMPTS + 1, "cap on s1, then one act on s2"
    memory.close()


def test_causality_no_observation_means_no_action(tmp_path) -> None:
    """The hold-state gates ACT: pure silence -> zero actions, state frozen."""
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = _three_step_task("CAUS-1")

    loop, calls, _spoken = _make_loop(task, memory, [], max_turns=5)
    summary = loop.run()

    assert summary.tool_calls == 0 and calls == []
    assert summary.stalled_turns == 5, "every idle turn stalls, bounded by the idle cap"
    assert summary.status == TaskStatus.ACTIVE.value
    memory.close()


def test_max_idle_turns_bounds_waiting_independently_of_work_budget(tmp_path) -> None:
    """Work budget and idle budget are separate: waiting does not burn max_turns."""
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(task_id="IDLE-1", goal="g", steps=[Step(id="s1", desc="click A")])

    # One work turn (acts on s1), then silence: the run must end after
    # max_idle_turns consecutive quiet turns even though max_turns is huge.
    loop, calls, _spoken = _make_loop(
        task,
        memory,
        [Observation(screenshot="shot")],
        max_turns=1000,
        max_idle_turns=3,
    )
    summary = loop.run()

    assert summary.tool_calls == 1
    assert summary.turns == 4, "1 work turn + exactly max_idle_turns quiet turns"
    memory.close()
