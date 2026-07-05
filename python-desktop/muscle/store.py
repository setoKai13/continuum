"""SQLite persistence for Muscle Memory (its own table, its own file).

Kept entirely inside the `muscle/` package so the rest of the codebase stays
untouched: this owns a single `muscle_memory` table and never reaches into
`memory.py`'s schema. Pure stdlib `sqlite3`, so it is always importable in
tests and the dry-run (no lazy import needed).

Each row is ONE learned reflex, unique per `(site, step_key)`: on a given
`site`, for a given templated `step_key`, a screenshot `embedding` (the Check
capture) grounded to a concrete `action`. Writes go through `upsert`, so a fresh
verified grounding OVERWRITES the stale reflex for that key -- the self-heal a
site redesign needs (risk R9), with no accumulation. `site`/`created_at` are
stamped on every row (risk R7); `last_used_at`/`success_count` drive the cheap
per-site eviction policy.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from state import now_iso as _now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS muscle_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site TEXT NOT NULL,
    step_key TEXT NOT NULL,
    embedding TEXT NOT NULL,
    action TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL,
    success_count INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_muscle_site_step
    ON muscle_memory (site, step_key);
"""

# Columns added after v0 shipped; a store opened on a pre-v1 db is migrated in
# place (the table owns its own schema, so this needs no app-wide migration).
_ADDED_COLUMNS = {
    "last_used_at": "TEXT NOT NULL DEFAULT ''",
    "success_count": "INTEGER NOT NULL DEFAULT 1",
}


@dataclass
class MuscleRow:
    """One stored reflex: a capture embedding paired with the action it grounded."""

    embedding: list[float]
    action: dict


class MuscleStore:
    """Owns the SQLite connection for the muscle_memory table."""

    def __init__(self, db_path: str | Path) -> None:
        """Opens (creating if needed) the muscle store and its schema.

        Args:
            db_path: Path to the SQLite file. May be the same file the rest of
                the app uses -- this only ever touches its own table.
        """
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        with closing(self._conn.cursor()) as cur:
            cur.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Adds v1 columns to a table created by an earlier (v0) build."""
        with closing(self._conn.cursor()) as cur:
            cur.execute("PRAGMA table_info(muscle_memory)")
            present = {row["name"] for row in cur.fetchall()}
            for column, decl in _ADDED_COLUMNS.items():
                if column not in present:
                    cur.execute(f"ALTER TABLE muscle_memory ADD COLUMN {column} {decl}")

    def close(self) -> None:
        """Closes the underlying SQLite connection."""
        self._conn.close()

    def upsert(self, site: str, step_key: str, embedding: list[float], action: dict) -> bool:
        """Writes ONE reflex per (site, step_key), replacing any stale one.

        A verified re-grounding of an already-known step (which only happens when
        the pre-replay Check missed on a changed screen) overwrites the stale row
        instead of appending -- this is the self-heal (risk R9). `success_count`
        carries forward and increments so eviction can favor proven reflexes.

        Args:
            site: Stable scope key (app/host) the reflex was learned on.
            step_key: Templated, normalized step description this reflex answers.
            embedding: Check capture at the moment it was grounded.
            action: Serialized ActionPlan (kind/target/text) to replay.

        Returns:
            True if an existing reflex was overwritten (a heal), False if new.
        """
        now = _now_iso()
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "SELECT MAX(success_count) AS sc, MIN(created_at) AS ca FROM muscle_memory "
                "WHERE site = ? AND step_key = ?",
                (site, step_key),
            )
            prior = cur.fetchone()
            existed = prior["sc"] is not None
            success_count = (prior["sc"] or 0) + 1
            created_at = prior["ca"] if existed else now
            cur.execute(
                "DELETE FROM muscle_memory WHERE site = ? AND step_key = ?", (site, step_key)
            )
            cur.execute(
                """
                INSERT INTO muscle_memory
                    (site, step_key, embedding, action, created_at, last_used_at, success_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (site, step_key, json.dumps(embedding), json.dumps(action), created_at, now, success_count),
            )
        self._conn.commit()
        return existed

    def touch(self, site: str, step_key: str) -> None:
        """Bumps recency/usage for a reflex that was successfully REPLAYED.

        Called only from the verify-gated commit path (never from `recall`
        itself), so the read path stays write-free (risk R2) while eviction
        ranking can still reflect real usage. Without this, a reflex replayed
        100 times keeps its original `success_count`/`last_used_at` and looks
        "unused" -- so `enforce_cap` would evict the most-proven reflex first
        (the exact inversion the reviewer flagged). Incrementing here makes a
        heavily-replayed reflex the LAST evicted, not the first.
        """
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "UPDATE muscle_memory SET last_used_at = ?, success_count = success_count + 1 "
                "WHERE site = ? AND step_key = ?",
                (_now_iso(), site, step_key),
            )
        self._conn.commit()

    def enforce_cap(self, site: str, cap: int) -> int:
        """Evicts the weakest reflexes so `site` keeps at most `cap` of them.

        The cheap, maintenance-free policy from the design: keep the most-proven
        and most-recent, drop the rest. Ordering evicts lowest `success_count`
        first, then oldest `last_used_at` -- so a rarely-successful or long-idle
        reflex goes before a proven, recently-refreshed one.

        Args:
            site: Scope to bound.
            cap: Maximum reflexes to retain for this site (<= 0 disables).

        Returns:
            The number of reflexes evicted.
        """
        if cap <= 0:
            return 0
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT COUNT(*) AS n FROM muscle_memory WHERE site = ?", (site,))
            overflow = int(cur.fetchone()["n"]) - cap
            if overflow <= 0:
                return 0
            cur.execute(
                """
                DELETE FROM muscle_memory WHERE id IN (
                    SELECT id FROM muscle_memory WHERE site = ?
                    ORDER BY success_count ASC, last_used_at ASC
                    LIMIT ?
                )
                """,
                (site, overflow),
            )
        self._conn.commit()
        return overflow

    def lookup(self, site: str, step_key: str) -> list[MuscleRow]:
        """Returns every reflex stored for this (site, step_key) pair.

        Scoping the SQL lookup by site+step_key BEFORE any Check math is what
        makes per-site memory cheap and keeps cross-app matches from ever being
        considered. (Writes keep this to one row per key; the list return type is
        preserved so the recall path stays unchanged.)
        """
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "SELECT embedding, action FROM muscle_memory WHERE site = ? AND step_key = ?",
                (site, step_key),
            )
            rows = cur.fetchall()
        return [MuscleRow(embedding=json.loads(r["embedding"]), action=json.loads(r["action"])) for r in rows]

    def count(self) -> int:
        """Returns the total number of stored reflexes (for tests/HUD)."""
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT COUNT(*) AS n FROM muscle_memory")
            return int(cur.fetchone()["n"])

    def count_for_site(self, site: str) -> int:
        """Returns how many reflexes are stored for one site (for eviction tests)."""
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT COUNT(*) AS n FROM muscle_memory WHERE site = ?", (site,))
            return int(cur.fetchone()["n"])

    def clear(self) -> None:
        """Wipes every reflex -- the clean rollback path (risk R6)."""
        with closing(self._conn.cursor()) as cur:
            cur.execute("DELETE FROM muscle_memory")
        self._conn.commit()
