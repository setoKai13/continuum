"""The hold-state core: a persistent situational model for a long-running task.

`TaskState` is the load-bearing primitive of Continuum. Every action the
agent takes must be selected through `next_actionable_step()`; nothing acts
straight off a fresh observation. The state survives process restarts
(persisted via memory.py), gets recalibrated by voice overrides, and its
`session_count` proves that a `--resume` reloads real history instead of
starting a blank snapshot.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def now_iso() -> str:
    """Returns the current UTC timestamp as an ISO-8601 string.

    Shared by every module that stamps rows/records (memory.py imports it)
    so the timestamp format cannot drift between tables.
    """
    return datetime.now(timezone.utc).isoformat()


class StepStatus(str, Enum):
    """Lifecycle of a single step inside a task's plan."""

    TODO = "todo"
    DOING = "doing"
    BLOCKED = "blocked"
    DONE = "done"


class TaskStatus(str, Enum):
    """Lifecycle of the overall task."""

    ACTIVE = "active"
    PAUSED = "paused"
    DONE = "done"


class Step(BaseModel):
    """A single unit of work inside a task plan.

    `history` records the concrete actions already executed for this step
    (last few only). It feeds two prompts: grounding (do not repeat what
    did not work) and verification (an outcome invisible on screen, like a
    clipboard copy, can be judged from the actions instead of the pixels).
    """

    id: str
    desc: str
    status: StepStatus = StepStatus.TODO
    note: str | None = None
    history: list[str] = Field(default_factory=list)


class Override(BaseModel):
    """A rule learned live from an operator correction.

    Example: "network bugs now go to INFRA, not OPS" recalibrates every
    remaining (non-done) step whose description matches `when`.
    """

    when: str
    rule: str
    applied: bool = False
    created_at: str = Field(default_factory=now_iso)


class TaskState(BaseModel):
    """Persistent situational model for one long-running task.

    Attributes:
        task_id: Stable identifier, used as the SQLite primary key and as
            the `--resume <task_id>` argument.
        goal: The operator's stated objective for this task.
        steps: Ordered plan; each step tracks its own status.
        facts: Free-form observations accumulated from OBSERVE/UPDATE_STATE.
        open_questions: Unresolved ambiguities the agent has flagged.
        overrides: Operator corrections that recalibrate remaining steps.
        session_count: Incremented every time this task is resumed/reloaded.
        status: Overall task lifecycle (active/paused/done).
    """

    task_id: str
    goal: str
    steps: list[Step] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    overrides: list[Override] = Field(default_factory=list)
    session_count: int = 1
    status: TaskStatus = TaskStatus.ACTIVE

    def mark_step(self, step_id: str, status: StepStatus, note: str | None = None) -> None:
        """Transitions one step to a new status, optionally attaching a note.

        Args:
            step_id: Identifier of the step to update.
            status: New status to assign.
            note: Optional free-text annotation (e.g. what the model observed).

        Raises:
            KeyError: If no step with `step_id` exists.
        """
        for step in self.steps:
            if step.id == step_id:
                step.status = status
                if note is not None:
                    step.note = note
                return
        raise KeyError(f"Unknown step id: {step_id!r}")

    def add_fact(self, fact: str) -> None:
        """Appends an observation to the running fact log.

        Args:
            fact: A short, human-readable observation string.
        """
        self.facts.append(fact)

    def add_open_question(self, question: str) -> None:
        """Records an unresolved ambiguity for later clarification.

        Args:
            question: The question text.
        """
        self.open_questions.append(question)

    def add_steps(self, descriptions: list[str]) -> list[Step]:
        """Appends new TODO steps to the plan from a list of descriptions.

        This is how a planner grows the hold-state live: the operator's first
        voice instruction is decomposed into concrete steps and appended here,
        so `next_actionable_step()` has something to act on. Ids continue after
        any existing steps (`s{n}`) so appending after a resume never clobbers
        existing step ids or their status.

        Args:
            descriptions: Ordered, human-readable step descriptions.

        Returns:
            The list of `Step` objects actually created (blank entries skipped).
        """
        created: list[Step] = []
        for description in descriptions:
            text = description.strip()
            if not text:
                continue
            step = Step(id=f"s{len(self.steps) + 1}", desc=text)
            self.steps.append(step)
            created.append(step)
        return created

    def apply_override(self, when: str, rule: str) -> list[Step]:
        """Records an operator correction and recalibrates matching steps.

        Only steps that are NOT yet `done` and whose description matches
        `when` (case-insensitive substring) get their note updated to
        reflect the new rule; already-completed steps are left untouched
        so history stays honest. A `blocked` step that gets recalibrated
        returns to `todo`: the correction is new information, so the step
        deserves a fresh attempt instead of staying dead.

        Args:
            when: A substring describing which steps are affected.
            rule: The new rule text to attach to matching steps.

        Returns:
            The steps recalibrated by this override (empty if none matched).
        """
        override = Override(when=when, rule=rule)
        self.overrides.append(override)

        pattern = re.compile(re.escape(when), re.IGNORECASE)
        recalibrated: list[Step] = []
        for step in self.steps:
            if step.status == StepStatus.DONE:
                continue
            if pattern.search(step.desc) or (step.note and pattern.search(step.note)):
                step.note = rule
                if step.status == StepStatus.BLOCKED:
                    step.status = StepStatus.TODO
                recalibrated.append(step)
        override.applied = bool(recalibrated)
        return recalibrated

    def next_actionable_step(self) -> Step | None:
        """Selects the next step the agent is allowed to act on.

        This is the sole gate through which the agent loop may choose an
        action: it prefers a step already `doing` (resume mid-step), then
        falls back to the first `todo` step in order. `blocked` and `done`
        steps are never returned.

        Returns:
            The step to act on next, or None if there is nothing actionable
            (task complete, or every remaining step is blocked).
        """
        for step in self.steps:
            if step.status == StepStatus.DOING:
                return step
        for step in self.steps:
            if step.status == StepStatus.TODO:
                return step
        return None

    def progress(self) -> tuple[int, int]:
        """Returns (steps_done, steps_total)."""
        done = sum(1 for s in self.steps if s.status == StepStatus.DONE)
        return done, len(self.steps)

    def is_complete(self) -> bool:
        """True once every step is done (and there is at least one step)."""
        if not self.steps:
            return False
        return all(s.status == StepStatus.DONE for s in self.steps)

    def resume(self) -> None:
        """Bumps the session counter and reactivates a paused task.

        Called every time a task is reloaded from storage (including
        `--resume`), so `session_count` is proof the process is reading real
        history rather than a fresh in-memory snapshot.
        """
        self.session_count += 1
        if self.status == TaskStatus.PAUSED:
            self.status = TaskStatus.ACTIVE

    def render(self) -> dict[str, Any]:
        """Produces a plain-dict snapshot suitable for the HUD panel.

        Returns:
            A dict with goal, status, session_count, progress, steps,
            facts, open_questions and overrides -- everything the left HUD
            panel needs to draw without importing pydantic internals.
        """
        done, total = self.progress()
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "status": self.status.value,
            "session_count": self.session_count,
            "progress": f"{done}/{total}",
            "steps": [
                {"id": s.id, "desc": s.desc, "status": s.status.value, "note": s.note}
                for s in self.steps
            ],
            "facts": list(self.facts),
            "open_questions": list(self.open_questions),
            "overrides": [
                {"when": o.when, "rule": o.rule, "applied": o.applied}
                for o in self.overrides
            ],
        }
