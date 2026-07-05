"""SQLite-backed persistence for the hold-state and its supporting logs.

This is what makes `TaskState` a real hold-state instead of an in-memory
toy: task snapshots survive process exit, a key/value memory table carries
free-form facts across tasks, `task_log` records every tool call for the
HUD/audit trail, and `trajectories` captures full run traces for replay.

Only pure standard-library `sqlite3` is used here (no lazy-import needed --
sqlite3 ships with CPython), so this module is always safe to import in
tests and the dry-run.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from state import TaskState, now_iso as _now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_state (
    task_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    payload TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    events TEXT NOT NULL DEFAULT '[]'
);
"""


class MemoryStore:
    """Owns the SQLite connection and all hold-state persistence operations.

    Attributes:
        db_path: Filesystem path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Opens (creating if needed) the SQLite database and its schema.

        Args:
            db_path: Path to the database file, e.g. "continuum.db".
        """
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        with closing(self._conn.cursor()) as cur:
            cur.executescript(_SCHEMA)
        self._conn.commit()
        self._open_trajectory_ids: dict[str, int] = {}

    def close(self) -> None:
        """Closes the underlying SQLite connection."""
        self._conn.close()

    # -- task_state ---------------------------------------------------

    def save_task_state(self, task: TaskState) -> None:
        """Upserts a `TaskState` snapshot keyed by `task_id`.

        Args:
            task: The task state to persist.
        """
        payload = task.model_dump_json()
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO task_state (task_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (task.task_id, payload, _now_iso()),
            )
        self._conn.commit()

    def load_task_state(self, task_id: str) -> TaskState | None:
        """Loads a raw (non-resumed) `TaskState` snapshot from disk.

        Args:
            task_id: Identifier of the task to load.

        Returns:
            The parsed `TaskState`, or None if no snapshot exists.
        """
        with closing(self._conn.cursor()) as cur:
            row = cur.execute(
                "SELECT payload FROM task_state WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return TaskState.model_validate_json(row["payload"])

    def resume_task_state(self, task_id: str) -> TaskState | None:
        """Loads a `TaskState` and bumps its session counter (proof of resume).

        This is the entry point `main.py --resume <task_id>` calls: it is
        distinct from `load_task_state` because resuming is a stateful act
        (it increments `session_count` and reactivates a paused task) while
        a plain load is a read-only peek.

        Args:
            task_id: Identifier of the task to resume.

        Returns:
            The resumed `TaskState` (already saved back with the bumped
            session count), or None if no snapshot exists for `task_id`.
        """
        task = self.load_task_state(task_id)
        if task is None:
            return None
        task.resume()
        self.save_task_state(task)
        return task

    # -- memory (key/value) --------------------------------------------

    def memory_save(self, key: str, value: Any) -> None:
        """Upserts a JSON-serializable value under `key`.

        Args:
            key: Memory key.
            value: Any JSON-serializable value.
        """
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO memory (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(value), _now_iso()),
            )
        self._conn.commit()

    def memory_get(self, key: str) -> Any | None:
        """Fetches a single memory value by key.

        Args:
            key: Memory key.

        Returns:
            The deserialized value, or None if absent.
        """
        with closing(self._conn.cursor()) as cur:
            row = cur.execute("SELECT value FROM memory WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def memory_all(self) -> dict[str, Any]:
        """Returns every memory key/value pair as a plain dict.

        Returns:
            A dict mapping memory keys to their deserialized values.
        """
        with closing(self._conn.cursor()) as cur:
            rows = cur.execute("SELECT key, value FROM memory").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    # -- task_log (tool call stream, feeds the HUD) ---------------------

    def log_tool(self, task_id: str, turn: int, tool_name: str, payload: dict[str, Any] | None = None) -> None:
        """Appends one tool-call record to the audit log.

        Args:
            task_id: Task this call belongs to.
            turn: Loop iteration number.
            tool_name: Name of the tool/action invoked.
            payload: Optional structured detail (args, result, etc).
        """
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO task_log (task_id, turn, tool_name, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task_id, turn, tool_name, json.dumps(payload or {}), _now_iso()),
            )
        self._conn.commit()

    def get_task_log(self, task_id: str) -> list[dict[str, Any]]:
        """Returns every logged tool call for a task, oldest first.

        Args:
            task_id: Task to fetch the log for.

        Returns:
            A list of dicts with turn/tool_name/payload/created_at.
        """
        with closing(self._conn.cursor()) as cur:
            rows = cur.execute(
                "SELECT turn, tool_name, payload, created_at FROM task_log "
                "WHERE task_id = ? ORDER BY id ASC",
                (task_id,),
            ).fetchall()
        return [
            {
                "turn": row["turn"],
                "tool_name": row["tool_name"],
                "payload": json.loads(row["payload"]) if row["payload"] else {},
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # -- trajectories (replayable run traces) ---------------------------

    def open_trajectory(self, task_id: str) -> int:
        """Starts a new trajectory record for a task.

        Args:
            task_id: Task this trajectory belongs to.

        Returns:
            The new trajectory's row id.
        """
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "INSERT INTO trajectories (task_id, started_at, events) VALUES (?, ?, '[]')",
                (task_id, _now_iso()),
            )
            trajectory_id = cur.lastrowid
        self._conn.commit()
        self._open_trajectory_ids[task_id] = trajectory_id
        return trajectory_id

    def append_trajectory(self, task_id: str, event: dict[str, Any]) -> None:
        """Appends one event to the currently open trajectory for a task.

        Args:
            task_id: Task whose open trajectory receives the event.
            event: Structured event payload (e.g. observation/decision/action).

        Raises:
            ValueError: If no trajectory is open for this task.
        """
        trajectory_id = self._open_trajectory_ids.get(task_id)
        if trajectory_id is None:
            raise ValueError(f"No open trajectory for task_id={task_id!r}")
        with closing(self._conn.cursor()) as cur:
            row = cur.execute(
                "SELECT events FROM trajectories WHERE id = ?", (trajectory_id,)
            ).fetchone()
            events = json.loads(row["events"]) if row else []
            events.append(event)
            cur.execute(
                "UPDATE trajectories SET events = ? WHERE id = ?",
                (json.dumps(events), trajectory_id),
            )
        self._conn.commit()

    def close_trajectory(self, task_id: str) -> None:
        """Marks the currently open trajectory for a task as ended.

        Args:
            task_id: Task whose open trajectory should be closed.
        """
        trajectory_id = self._open_trajectory_ids.pop(task_id, None)
        if trajectory_id is None:
            return
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "UPDATE trajectories SET ended_at = ? WHERE id = ?",
                (_now_iso(), trajectory_id),
            )
        self._conn.commit()

    # -- startup context reinjection -------------------------------------

    def load_startup_context(self, task_id: str) -> dict[str, Any]:
        """Builds the context dict reinjected into the system instruction at boot.

        This is the mechanism that makes a `--resume` more than a database
        read: it hands the agent its resumed (session-bumped) task state
        plus the global key/value memory, so the model can literally say
        "I'm resuming <task_id>, N steps remain."

        Args:
            task_id: Task to resume and build context for.

        Returns:
            A dict with keys "task" (rendered TaskState dict, or None if
            unknown) and "memory" (the full key/value memory dump).
        """
        task = self.resume_task_state(task_id)
        return {
            "task": task.render() if task else None,
            "memory": self.memory_all(),
        }
