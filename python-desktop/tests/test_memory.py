"""Tests for the SQLite hold-state store (memory.py)."""

from __future__ import annotations

from pathlib import Path

from memory import MemoryStore
from state import Step, StepStatus, TaskState, TaskStatus


def make_task(task_id: str = "TRI-3") -> TaskState:
    return TaskState(
        task_id=task_id,
        goal="Triage the Linear queue",
        steps=[Step(id="s1", desc="open Linear"), Step(id="s2", desc="triage #1")],
    )


def test_save_then_load_roundtrips(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db")
    task = make_task()
    task.mark_step("s1", StepStatus.DONE)
    task.add_fact("5 tickets open")

    store.save_task_state(task)
    loaded = store.load_task_state("TRI-3")

    assert loaded is not None
    assert loaded.task_id == "TRI-3"
    assert loaded.facts == ["5 tickets open"]
    assert loaded.next_actionable_step().id == "s2"
    store.close()


def test_load_unknown_task_returns_none(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db")
    assert store.load_task_state("does-not-exist") is None
    store.close()


def test_resume_increments_session_count_and_persists(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db")
    task = make_task()
    task.status = TaskStatus.PAUSED
    store.save_task_state(task)

    resumed = store.resume_task_state("TRI-3")
    assert resumed is not None
    assert resumed.session_count == 2
    assert resumed.status == TaskStatus.ACTIVE

    # Persisted, not just mutated in memory: a second resume bumps again.
    resumed_again = store.resume_task_state("TRI-3")
    assert resumed_again.session_count == 3
    store.close()


def test_memory_kv_upsert(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db")
    store.memory_save("preferred_app", "Linear")
    store.memory_save("preferred_app", "Reminders")

    assert store.memory_get("preferred_app") == "Reminders"
    assert store.memory_all() == {"preferred_app": "Reminders"}
    store.close()


def test_log_tool_and_get_task_log(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db")
    store.log_tool("TRI-3", turn=1, tool_name="click", payload={"x": 10, "y": 20})
    store.log_tool("TRI-3", turn=2, tool_name="type", payload={"text": "hello"})

    log = store.get_task_log("TRI-3")
    assert len(log) == 2
    assert log[0]["tool_name"] == "click"
    assert log[0]["payload"] == {"x": 10, "y": 20}
    assert log[1]["turn"] == 2
    store.close()


def test_trajectory_open_append_close(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db")
    store.open_trajectory("TRI-3")
    store.append_trajectory("TRI-3", {"type": "observe", "detail": "screenshot taken"})
    store.append_trajectory("TRI-3", {"type": "act", "detail": "clicked button"})
    store.close_trajectory("TRI-3")

    with store._conn:  # inspect raw row for the test only
        row = store._conn.execute(
            "SELECT events, ended_at FROM trajectories WHERE task_id = ?", ("TRI-3",)
        ).fetchone()
    import json

    events = json.loads(row["events"])
    assert len(events) == 2
    assert row["ended_at"] is not None
    store.close()


def test_load_startup_context_reinjects_resumed_task_and_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db")
    task = make_task()
    task.status = TaskStatus.PAUSED
    store.save_task_state(task)
    store.memory_save("preferred_app", "Linear")

    context = store.load_startup_context("TRI-3")

    assert context["task"]["task_id"] == "TRI-3"
    assert context["task"]["session_count"] == 2
    assert context["memory"] == {"preferred_app": "Linear"}
    store.close()


def test_load_startup_context_unknown_task(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.db")
    context = store.load_startup_context("nope")
    assert context["task"] is None
    store.close()
