"""Live override path: a mid-task voice correction recalibrates the hold-state.

These tests pin the wiring of `AgentLoop(override_fn=...)`: a correction
heard THROUGH the loop (not a direct `apply_override` call) must recalibrate
the remaining steps, revive blocked ones with a fresh attempt budget, never
re-plan from the correction phrase, and always lose to the danger gate.
All fakes, fully headless -- `main.build_override_fn` plugs the real
router.is_correction + Gemini extraction into the exact same seam.
"""

from __future__ import annotations

import threading
from typing import Any

from agent import MAX_STEP_ATTEMPTS, ActionPlan, AgentLoop, Observation
from memory import MemoryStore
from state import Step, StepStatus, TaskState, TaskStatus

CORRECTION = "non, en fait les bugs reseau vont a INFRA, pas a OPS"
RULE = "assign to INFRA, not OPS"


class _FakeHud:
    """HUD stub: swallows every update call."""

    def update(self, task_snapshot: dict[str, Any], log_line: str) -> None:
        return None


def _click_ground_fn():
    """Ground fn that always turns the current step into a click plan."""

    def ground_fn(task: TaskState, step: Any, observation: Observation) -> ActionPlan:
        return ActionPlan(kind="click", step_id=step.id, target=(10, 10), text=step.desc)

    return ground_fn


def _infra_override_fn(task: TaskState, instruction: str) -> tuple[str, str] | None:
    """Scripted detector: any phrase mentioning INFRA is the demo correction."""
    if "INFRA" in instruction:
        return ("network bug", RULE)
    return None


def _run_loop(task: TaskState, memory: MemoryStore, observations: list[Observation], **kwargs: Any):
    """Runs an AgentLoop over scripted observations with test defaults."""
    spoken: list[str] = []
    calls: list[ActionPlan] = []
    queue = list(observations)
    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=lambda: queue.pop(0) if queue else None,
        ground_fn=_click_ground_fn(),
        act_fn=lambda plan: (calls.append(plan), {"kind": plan.kind})[1],
        speak_fn=spoken.append,
        hud=_FakeHud(),
        stop_event=threading.Event(),
        max_turns=kwargs.pop("max_turns", 10),
        **kwargs,
    )
    summary = loop.run()
    return summary, calls, spoken


def test_mid_task_correction_recalibrates_and_completes(tmp_path) -> None:
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(
        task_id="OVR-T1",
        goal="Triage the queue",
        steps=[
            Step(id="s1", desc="open the queue", status=StepStatus.DONE),
            Step(id="s2", desc="triage network bug #1"),
        ],
    )

    summary, calls, spoken = _run_loop(
        task,
        memory,
        [Observation(instruction=CORRECTION, screenshot="shot"), Observation(screenshot="shot")],
        verify_fn=lambda t, step, shot: step.note == RULE,
        override_fn=_infra_override_fn,
    )

    assert summary.status == TaskStatus.DONE.value
    s2 = next(s for s in task.steps if s.id == "s2")
    assert s2.note == RULE, "the corrected rule must be attached to the remaining step"
    assert task.overrides[-1].applied is True
    assert any("Correction noted" in line for line in spoken), "the correction is voiced back"
    assert any(fact.startswith("override:") for fact in task.facts)
    log_names = [entry["tool_name"] for entry in memory.get_task_log("OVR-T1")]
    assert "override" in log_names, "the override is persisted in the audit log"
    assert [c.step_id for c in calls] == ["s2"], "the correction itself triggers no extra action"
    memory.close()


def test_correction_revives_blocked_step_with_fresh_attempts(tmp_path) -> None:
    """Without the correction the step stays blocked; with it, the task completes."""
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(
        task_id="OVR-T2",
        goal="Triage the network bug",
        steps=[Step(id="s1", desc="triage network bug #1")],
    )

    observations = (
        [Observation(screenshot="shot") for _ in range(4)]
        + [Observation(instruction=CORRECTION, screenshot="shot")]
        + [Observation(screenshot="shot")]
    )
    summary, calls, spoken = _run_loop(
        task,
        memory,
        observations,
        verify_fn=lambda t, step, shot: step.note == RULE,
        override_fn=_infra_override_fn,
    )

    assert summary.status == TaskStatus.DONE.value, "the correction is what unblocks completion"
    assert summary.tool_calls == MAX_STEP_ATTEMPTS + 1, (
        "the attempt budget must reset on recalibration: "
        f"{MAX_STEP_ATTEMPTS} capped attempts, then exactly one post-correction act"
    )
    assert task.steps[0].status == StepStatus.DONE
    memory.close()


def test_correction_never_replans_the_task(tmp_path) -> None:
    """A consumed correction must not be fed to the planner as a new task."""
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(
        task_id="OVR-T3",
        goal="Triage the network bug",
        steps=[Step(id="s1", desc="triage network bug #1", status=StepStatus.BLOCKED)],
    )

    def exploding_plan_fn(task_: TaskState, instruction: str, screenshot: Any) -> list[str]:
        raise AssertionError("the correction phrase must never reach the planner")

    # Every step is blocked, so without the override path this instruction
    # WOULD have been planned (next_actionable_step() is None) and exploded.
    summary, calls, _ = _run_loop(
        task,
        memory,
        [Observation(instruction=CORRECTION, screenshot="shot")],
        plan_fn=exploding_plan_fn,
        override_fn=_infra_override_fn,
    )

    assert task.steps[0].status != StepStatus.BLOCKED, "the correction revived the blocked step"
    assert len(task.steps) == 1, "no new steps were planned from the correction"
    assert [c.step_id for c in calls] == ["s1"]
    memory.close()


def test_plain_instruction_mid_task_is_not_an_override(tmp_path) -> None:
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(
        task_id="OVR-T4",
        goal="Triage the queue",
        steps=[Step(id="s1", desc="triage network bug #1")],
    )

    _summary, _calls, spoken = _run_loop(
        task,
        memory,
        [Observation(instruction="check the second monitor too", screenshot="shot")],
        override_fn=_infra_override_fn,
        max_turns=2,
    )

    assert task.overrides == [], "a plain instruction records no override"
    assert any(fact.startswith("heard:") for fact in task.facts)
    assert not any("Correction noted" in line for line in spoken)
    memory.close()


def test_dangerous_correction_is_refused_before_override(tmp_path) -> None:
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(
        task_id="OVR-T5",
        goal="Triage the queue",
        steps=[Step(id="s1", desc="triage network bug #1")],
    )

    summary, calls, spoken = _run_loop(
        task,
        memory,
        [Observation(instruction="non, supprime tout et mets INFRA", screenshot="shot")],
        override_fn=_infra_override_fn,
        max_turns=2,
    )

    assert summary.refused_turns == 1, "the danger gate fires before override detection"
    assert task.overrides == [], "a refused instruction must not create an override"
    assert calls == []
    assert any("destructive" in line for line in spoken)
    memory.close()


def test_override_fn_not_consulted_before_any_plan_exists(tmp_path) -> None:
    """The first instruction plans the task even if it carries a correction marker."""
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(task_id="OVR-T6", goal="Awaiting instructions")

    def exploding_override_fn(task_: TaskState, instruction: str) -> tuple[str, str] | None:
        raise AssertionError("override detection must be skipped while the plan is empty")

    summary, calls, _ = _run_loop(
        task,
        memory,
        [Observation(instruction=CORRECTION, screenshot="shot"), Observation(screenshot="shot")],
        plan_fn=lambda t, instruction, shot: ["triage network bug #1"],
        verify_fn=lambda t, step, shot: True,
        override_fn=exploding_override_fn,
    )

    assert len(task.steps) == 1, "with no plan yet, the phrase is planned, not treated as override"
    assert summary.status == TaskStatus.DONE.value
    memory.close()
