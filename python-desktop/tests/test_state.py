"""Tests for the hold-state core (state.py)."""

from __future__ import annotations

from state import Step, StepStatus, TaskState, TaskStatus


def make_task() -> TaskState:
    """Builds a small 3-step task fixture used across tests."""
    return TaskState(
        task_id="TRI-3",
        goal="Triage the Linear queue",
        steps=[
            Step(id="s1", desc="open Linear"),
            Step(id="s2", desc="triage network bug #1"),
            Step(id="s3", desc="triage network bug #2"),
        ],
    )


def test_next_actionable_step_prefers_doing_then_todo() -> None:
    task = make_task()
    assert task.next_actionable_step().id == "s1"

    task.mark_step("s1", StepStatus.DOING)
    assert task.next_actionable_step().id == "s1"

    task.mark_step("s1", StepStatus.DONE)
    assert task.next_actionable_step().id == "s2"


def test_next_actionable_step_none_when_all_done_or_blocked() -> None:
    task = make_task()
    for step in task.steps:
        task.mark_step(step.id, StepStatus.DONE)
    assert task.next_actionable_step() is None

    task2 = make_task()
    for step in task2.steps:
        task2.mark_step(step.id, StepStatus.BLOCKED)
    assert task2.next_actionable_step() is None


def test_mark_step_unknown_id_raises() -> None:
    task = make_task()
    try:
        task.mark_step("does-not-exist", StepStatus.DONE)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_add_fact_appends() -> None:
    task = make_task()
    task.add_fact("5 tickets open")
    assert task.facts == ["5 tickets open"]


def test_apply_override_recalibrates_only_non_done_matching_steps() -> None:
    task = make_task()
    task.mark_step("s1", StepStatus.DONE)

    recalibrated = task.apply_override(when="network bug", rule="assign to INFRA, not OPS")

    # s1 is done and does not match "network bug" -> untouched.
    s1 = next(s for s in task.steps if s.id == "s1")
    assert s1.note is None
    assert s1.status == StepStatus.DONE

    # s2, s3 match and are not done -> recalibrated.
    s2 = next(s for s in task.steps if s.id == "s2")
    s3 = next(s for s in task.steps if s.id == "s3")
    assert s2.note == "assign to INFRA, not OPS"
    assert s3.note == "assign to INFRA, not OPS"
    assert [s.id for s in recalibrated] == ["s2", "s3"]
    assert task.overrides[-1].applied is True


def test_apply_override_no_match_marks_unapplied() -> None:
    task = make_task()
    recalibrated = task.apply_override(when="does-not-match-anything", rule="noop")
    assert recalibrated == []
    assert task.overrides[-1].applied is False


def test_apply_override_unblocks_recalibrated_blocked_step() -> None:
    task = make_task()
    task.mark_step("s2", StepStatus.BLOCKED)

    recalibrated = task.apply_override(when="network bug", rule="assign to INFRA")

    s2 = next(s for s in task.steps if s.id == "s2")
    assert s2.status == StepStatus.TODO, "a corrected blocked step earns a fresh attempt"
    assert s2.note == "assign to INFRA"
    assert "s2" in [s.id for s in recalibrated]


def test_progress_and_is_complete() -> None:
    task = make_task()
    assert task.progress() == (0, 3)
    assert task.is_complete() is False

    for step in task.steps:
        task.mark_step(step.id, StepStatus.DONE)
    assert task.progress() == (3, 3)
    assert task.is_complete() is True


def test_resume_bumps_session_count_and_reactivates() -> None:
    task = make_task()
    assert task.session_count == 1

    task.status = TaskStatus.PAUSED
    task.resume()

    assert task.session_count == 2
    assert task.status == TaskStatus.ACTIVE


def test_render_is_plain_serializable_dict() -> None:
    task = make_task()
    task.add_fact("fact one")
    task.add_open_question("which app?")
    task.apply_override(when="triage", rule="prefer INFRA")

    snapshot = task.render()

    assert snapshot["task_id"] == "TRI-3"
    assert snapshot["progress"] == "0/3"
    assert snapshot["facts"] == ["fact one"]
    assert snapshot["open_questions"] == ["which app?"]
    assert len(snapshot["overrides"]) == 1
    assert isinstance(snapshot["steps"], list) and len(snapshot["steps"]) == 3
