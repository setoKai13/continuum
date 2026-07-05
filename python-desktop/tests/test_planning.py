"""Planner path: a natural-language instruction becomes actionable steps.

These tests pin the fix for the live-path gap where a brand-new task had an
empty plan and therefore could never act. They use plain fakes (no Gemini,
no GUI) so they run headless.
"""

from __future__ import annotations

import threading
from typing import Any

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


def test_add_steps_appends_sequential_ids_and_skips_blanks() -> None:
    task = TaskState(task_id="T1", goal="goal")

    created = task.add_steps(["first", "   ", "second"])

    assert [s.id for s in created] == ["s1", "s2"], "blank descriptions are skipped"
    assert [s.desc for s in created] == ["first", "second"]
    # A later call continues the numbering after existing steps (resume-safe).
    more = task.add_steps(["third"])
    assert more[0].id == "s3"
    assert len(task.steps) == 3


def test_instruction_is_planned_into_steps_then_acted(tmp_path) -> None:
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(task_id="T2", goal="clear the queue")
    assert task.next_actionable_step() is None, "new task starts with no actionable step"

    def plan_fn(task_: TaskState, instruction: str, screenshot: Any) -> list[str]:
        assert instruction == "process the queue"
        return ["open the queue", "triage the bug"]

    observations = [
        Observation(instruction="process the queue", screenshot="shot"),
        Observation(step_completed="s1"),
        Observation(step_completed="s2"),
    ]

    def observe_fn() -> Observation | None:
        return observations.pop(0) if observations else None

    calls: list[ActionPlan] = []
    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=observe_fn,
        ground_fn=_click_ground_fn(),
        act_fn=lambda plan: (calls.append(plan), {"kind": plan.kind})[1],
        speak_fn=lambda text: None,
        hud=_FakeHud(),
        stop_event=threading.Event(),
        max_turns=10,
        plan_fn=plan_fn,
    )
    summary = loop.run()
    memory.close()

    assert len(task.steps) == 2, "the instruction was decomposed into two steps"
    assert summary.status == TaskStatus.DONE.value
    assert [c.step_id for c in calls] == ["s1", "s2"], "acted in planned order"
    assert any("planned" in fact for fact in task.facts), "planning is recorded as a fact"


def test_without_plan_fn_a_new_task_never_acts(tmp_path) -> None:
    """Regression guard: no planner + empty plan => permanent stall, zero action."""
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(task_id="T3", goal="goal")

    observations = [Observation(instruction="do the thing", screenshot="shot")]

    def observe_fn() -> Observation | None:
        return observations.pop(0) if observations else None

    calls: list[ActionPlan] = []
    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=observe_fn,
        ground_fn=_click_ground_fn(),
        act_fn=lambda plan: (calls.append(plan), {"kind": plan.kind})[1],
        speak_fn=lambda text: None,
        hud=_FakeHud(),
        stop_event=threading.Event(),
        max_turns=4,
        plan_fn=None,
    )
    summary = loop.run()
    memory.close()

    assert summary.tool_calls == 0, "no planner means nothing to act on"
    assert len(task.steps) == 0


def test_verifier_advances_steps_without_step_completed(tmp_path) -> None:
    """The verifier (not an injected signal) is what moves the loop step to step."""
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(
        task_id="T4",
        goal="do two things",
        steps=[Step(id="s1", desc="click A"), Step(id="s2", desc="click B")],
    )

    # Fresh screenshot every turn (like the live observe when a plan is active).
    def observe_fn() -> Observation:
        return Observation(screenshot="shot")

    calls: list[ActionPlan] = []
    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=observe_fn,
        ground_fn=_click_ground_fn(),
        act_fn=lambda plan: (calls.append(plan), {"kind": plan.kind})[1],
        speak_fn=lambda text: None,
        hud=_FakeHud(),
        stop_event=threading.Event(),
        max_turns=12,
        verify_fn=lambda task_, step, shot: True,  # every acted step verifies done
    )
    summary = loop.run()
    memory.close()

    assert summary.status == TaskStatus.DONE.value
    assert [c.step_id for c in calls] == ["s1", "s2"], "advanced across steps on its own"


def test_step_blocked_after_max_attempts(tmp_path) -> None:
    """A step the verifier never confirms is capped, marked blocked, and skipped."""
    memory = MemoryStore(str(tmp_path / "t.db"))
    task = TaskState(task_id="T5", goal="stuck", steps=[Step(id="s1", desc="click A")])

    calls: list[ActionPlan] = []
    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=lambda: Observation(screenshot="shot"),
        ground_fn=_click_ground_fn(),
        act_fn=lambda plan: (calls.append(plan), {"kind": plan.kind})[1],
        speak_fn=lambda text: None,
        hud=_FakeHud(),
        stop_event=threading.Event(),
        max_turns=10,
        verify_fn=lambda task_, step, shot: False,  # never confirms completion
    )
    summary = loop.run()
    memory.close()

    assert summary.tool_calls == MAX_STEP_ATTEMPTS, "acted exactly the cap, then stopped"
    assert task.steps[0].status == StepStatus.BLOCKED
    assert task.open_questions, "a blocked step records an open question"
