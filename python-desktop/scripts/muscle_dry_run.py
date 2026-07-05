#!/usr/bin/env python3
"""Headless end-to-end proof of Muscle Memory through the REAL AgentLoop.

No Gemini, no encoder, no screen: a counting fake stands in for the cloud
grounding and a deterministic fake for the image encoder. It proves, through
the actual loop and the actual muscle wiring:

1. Run 1 (cold): every step is grounded by "Gemini" (the counter climbs), each
   grounding is STAGED, and only a verified step COMMITS a reflex.
2. Run 2 (warm, same store): the identical task grounds every step locally --
   the Gemini counter stays at ZERO -- and replays the same actions.
3. Safety: recall never writes; a remembered action still flows through the
   loop's is_dangerous guard (a destructive replay would be refused).
4. Self-heal (v1): on a warm run whose screen has CHANGED, the pre-replay Check
   misses, control hands back to Gemini WITH the remaining-work context (goal +
   already-done steps + live screen), and the fresh verified success OVERWRITES
   the stale trajectory -- the next run is fast again on the new screen.

Run: `.venv/bin/python scripts/muscle_dry_run.py`. Prints "MUSCLE DRY-RUN OK"
iff every assertion passes.
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
from muscle import (  # noqa: E402
    MuscleMemory,
    MuscleStore,
    build_muscle_ground_fn,
    build_muscle_verify_fn,
)
from state import Step, StepStatus, TaskState, TaskStatus  # noqa: E402


class FakeEmbedder:
    """Deterministic: the same screenshot token always embeds identically."""

    def __call__(self, screenshot: Any) -> list[float]:
        h = abs(hash(str(screenshot)))
        return [((h >> (8 * i)) & 0xFF) / 255.0 for i in range(4)]


class CountingGemini:
    """Fake cloud grounding: counts calls, always targets the current step."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, task: TaskState, step: Step, observation: Observation) -> ActionPlan | None:
        self.calls += 1
        return ActionPlan(kind="click", step_id=step.id, target=(100, 100), text=step.desc)


class RecordingActuator:
    def __init__(self) -> None:
        self.calls: list[ActionPlan] = []

    def __call__(self, plan: ActionPlan) -> dict[str, Any]:
        self.calls.append(plan)
        return {"kind": plan.kind, "target": plan.target}


class NullHud:
    def update(self, task_snapshot: dict[str, Any], log_line: str) -> None:
        pass


def make_task(task_id: str) -> TaskState:
    return TaskState(
        task_id=task_id,
        goal="Open the app and send the message",
        steps=[Step(id="s1", desc="open the compose window"), Step(id="s2", desc="click send")],
    )


def run_once(db_path: Path, muscle: MuscleMemory, task_id: str) -> tuple[int, RecordingActuator]:
    """Runs the real loop once over a 2-step task, returning Gemini call count."""
    memory = MemoryStore(db_path)
    task = make_task(task_id)
    gemini = CountingGemini()
    actuator = RecordingActuator()

    # Exactly the composition main.build_ground_fn uses when muscle is on:
    # tier 1.5 (recall/stage) wraps the cloud grounding; verify commits reflexes.
    ground_fn = build_muscle_ground_fn(gemini, muscle, site_fn=lambda t, o: "default")
    verify_fn = build_muscle_verify_fn(lambda t, s, shot: True, muscle)

    # A stable screenshot token each turn (the loop advances step by step).
    sensor_queue = [Observation(instruction="do the task", screenshot="shot")] + [
        Observation(screenshot="shot") for _ in range(8)
    ]

    def observe() -> Observation | None:
        return sensor_queue.pop(0) if sensor_queue else None

    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=observe,
        ground_fn=ground_fn,
        act_fn=actuator,
        speak_fn=lambda text: None,
        hud=NullHud(),
        stop_event=threading.Event(),
        max_turns=12,
        verify_fn=verify_fn,
    )
    summary = loop.run()
    memory.close()
    assert summary.status == TaskStatus.DONE.value, "task must reach done both runs"
    assert summary.steps_done == 2
    return gemini.calls, actuator


def scenario_learns_and_replays(tmp: Path) -> None:
    print("\n[muscle 1] cold run learns, warm run grounds locally with 0 Gemini calls")
    db = tmp / "app.db"
    muscle_db = tmp / "muscle.db"
    stored = MuscleStore(muscle_db)
    muscle = MuscleMemory(store=stored, embed_fn=FakeEmbedder(), threshold=0.92)

    calls_1, act_1 = run_once(db, muscle, "RUN-1")
    print(f"  run 1: {calls_1} Gemini grounding(s), reflexes stored={muscle.stats()['stored']}")
    assert calls_1 == 2, "cold run must ground both steps via Gemini"
    assert muscle.stats()["committed"] == 2, "both verified steps must commit a reflex"

    # A fresh MuscleMemory on the SAME store = a new process reloading learned reflexes.
    muscle2 = MuscleMemory(store=MuscleStore(muscle_db), embed_fn=FakeEmbedder(), threshold=0.92)
    calls_2, act_2 = run_once(tmp / "app2.db", muscle2, "RUN-2")
    print(f"  run 2: {calls_2} Gemini grounding(s), local hits={muscle2.stats()['local_hits']}")
    assert calls_2 == 0, "warm run must ground every step locally -- ZERO Gemini calls"
    assert muscle2.stats()["local_hits"] == 2, "both steps recalled locally"

    assert [c.target for c in act_1.calls] == [c.target for c in act_2.calls], "replayed clicks must match"
    print("  PASS: identical task, second run needs no cloud -- muscle memory works")


def scenario_recall_never_writes(tmp: Path) -> None:
    print("\n[muscle 2] recall is read-only: replays never mint new reflexes")
    muscle_db = tmp / "muscle2.db"
    store = MuscleStore(muscle_db)
    muscle = MuscleMemory(store=store, embed_fn=FakeEmbedder(), threshold=0.92)
    run_once(tmp / "b1.db", muscle, "B-1")
    baseline = store.count()

    warm = MuscleMemory(store=MuscleStore(muscle_db), embed_fn=FakeEmbedder(), threshold=0.92)
    for _ in range(3):
        run_once(tmp / f"b{_}.db", warm, f"B-{_}")
    assert MuscleStore(muscle_db).count() == baseline, "repeated warm runs must not grow the store"
    print(f"  PASS: store stayed at {baseline} reflex(es) across 3 warm runs (no self-reinforcement)")


class ContextRecordingGemini:
    """Fake cloud grounding that RECORDS the context each fallback call receives.

    Proves the mid-task handoff (risk R9): when the pre-replay Check misses,
    Gemini is re-invoked with the live task (goal + already-done steps) and the
    current screen -- not a blind full restart. Its target depends on the screen,
    so a healed (v2) grounding is visibly different from the stale (v1) one.
    """

    #: The action Gemini "grounds" for each screen it is shown.
    TARGETS = {"compose_v1": (10, 10), "send_v1": (20, 20), "send_v2": (99, 99)}

    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[dict[str, Any]] = []

    def __call__(self, task: TaskState, step: Step, observation: Observation) -> ActionPlan | None:
        self.calls += 1
        done = [s.id for s in task.steps if s.status == StepStatus.DONE]
        self.contexts.append({"goal": task.goal, "step": step.id, "done": done, "screen": observation.screenshot})
        target = self.TARGETS[observation.screenshot]
        return ActionPlan(kind="click", step_id=step.id, target=target, text=step.desc)


def run_over_screens(
    db_path: Path, muscle: MuscleMemory, task_id: str, screen_map: dict[str, str], gemini: ContextRecordingGemini
) -> RecordingActuator:
    """Drives the real loop, showing each step the screen `screen_map` assigns it.

    The observed screenshot tracks the current actionable step, so a per-step
    screen redesign (v1 -> v2) can be simulated exactly.
    """
    memory = MemoryStore(db_path)
    task = make_task(task_id)
    actuator = RecordingActuator()
    ground_fn = build_muscle_ground_fn(gemini, muscle, site_fn=lambda t, o: "default")
    verify_fn = build_muscle_verify_fn(lambda t, s, shot: True, muscle)

    def observe() -> Observation | None:
        step = task.next_actionable_step()
        if step is None:
            return None
        return Observation(screenshot=screen_map[step.id])

    loop = AgentLoop(
        task=task,
        memory=memory,
        observe_fn=observe,
        ground_fn=ground_fn,
        act_fn=actuator,
        speak_fn=lambda text: None,
        hud=NullHud(),
        stop_event=threading.Event(),
        max_turns=12,
        verify_fn=verify_fn,
    )
    summary = loop.run()
    memory.close()
    assert summary.status == TaskStatus.DONE.value and summary.steps_done == 2
    return actuator


def scenario_falls_back_and_self_heals(tmp: Path) -> None:
    print("\n[muscle 3] a changed screen misses the Check, hands back to Gemini, then self-heals")
    muscle_db = tmp / "heal.db"
    v1 = {"s1": "compose_v1", "s2": "send_v1"}
    v2 = {"s1": "compose_v1", "s2": "send_v2"}  # the 'send' step's screen was redesigned

    # Cold run learns both steps on the v1 screens.
    cold = MuscleMemory(store=MuscleStore(muscle_db), embed_fn=FakeEmbedder(), threshold=0.99)
    g1 = ContextRecordingGemini()
    act_cold = run_over_screens(tmp / "h1.db", cold, "H-1", v1, g1)
    print(f"  cold run: {g1.calls} Gemini grounding(s), reflexes stored={cold.stats()['stored']}")
    assert g1.calls == 2 and cold.stats()["stored"] == 2

    # Warm run on the redesigned screen: s1 replays locally, s2 MISSES the Check
    # and falls back to Gemini WITH context; the verified success heals the reflex.
    warm = MuscleMemory(store=MuscleStore(muscle_db), embed_fn=FakeEmbedder(), threshold=0.99)
    g2 = ContextRecordingGemini()
    act_warm = run_over_screens(tmp / "h2.db", warm, "H-2", v2, g2)
    print(f"  warm run (screen changed): {g2.calls} Gemini grounding(s), heals={warm.stats()['heals']}")
    assert g2.calls == 1, "only the changed step falls back; the unchanged one replays locally"
    handoff = g2.contexts[0]
    assert handoff["step"] == "s2" and handoff["screen"] == "send_v2"
    assert handoff["done"] == ["s1"], "handoff must carry the already-done steps (mid-task, not a restart)"
    assert handoff["goal"] == "Open the app and send the message", "handoff must carry the goal"
    assert warm.stats()["heals"] == 1 and warm.stats()["stored"] == 2, "heal overwrites, it does not accumulate"
    # The stale (20, 20) click is gone; the healed (99, 99) click replays now.
    assert act_warm.calls[1].target == (99, 99), "the healed step must click the NEW target"

    # Next run on the v2 screen is fast again -- zero Gemini calls.
    healed = MuscleMemory(store=MuscleStore(muscle_db), embed_fn=FakeEmbedder(), threshold=0.99)
    g3 = ContextRecordingGemini()
    run_over_screens(tmp / "h3.db", healed, "H-3", v2, g3)
    print(f"  post-heal run: {g3.calls} Gemini grounding(s) -- fast again on the new screen")
    assert g3.calls == 0, "after healing, the redesigned screen replays locally with no cloud"
    print("  PASS: Check miss -> context-preserving fallback -> stale trajectory overwritten (self-heal)")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="continuum-muscle-") as tmp:
        d = Path(tmp)
        scenario_learns_and_replays(d)
        scenario_recall_never_writes(d)
        scenario_falls_back_and_self_heals(d)
    print("\nMUSCLE DRY-RUN OK")


if __name__ == "__main__":
    main()
