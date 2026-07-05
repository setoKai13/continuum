"""Muscle Memory: local, learned grounding that replays past successful clicks.

The one load-bearing rule (risks R1/R2): a reflex is written to the store ONLY
after a step was grounded by the cloud (Gemini) AND later verified done. The
recall path is strictly read-only. Concretely:

    stage(step_id, ...)   # a Gemini grounding, held PENDING (not yet trusted)
    commit(step_id)       # verify_step_done passed -> pending becomes a reflex
    recall(...)           # read-only lookup + pre-replay Check; never writes

So the loop is never closed on itself: every reflex traces back to an
independent Gemini decision that the verifier confirmed. Local reads,
Gemini-gated writes.

Two v1 abstractions borrowed from muscle-mem carry the design (see
memory-agent-design): a `Check = capture + compare` validates the live screen
BEFORE a step replays (risk R8), and goal `params` templating keeps the cache
key reusable across dynamic arguments. A verified re-grounding of a stale step
OVERWRITES it (self-heal, risk R9), and a per-site cap keeps the store bounded.

Only stdlib (math/json) is used here, so importing this never pulls in a model;
the real image encoder is injected as `embed_fn` and lives behind a lazy import
in `embedders.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from agent import ActionPlan

from .check import Check, CaptureFn, build_default_check, cosine
from .store import MuscleStore
from .templating import fill_template, templatize

logger = logging.getLogger(__name__)

# Screenshot -> embedding vector. Injected so tests use a deterministic fake and
# live wiring uses a real local encoder (see embedders.build_default_embed_fn).
EmbedFn = CaptureFn
# (task, observation) -> the dynamic goal params for this run (e.g. {"query":
# "shoes"}). Injected into the ground wrapper; the default is no params.
ParamsFn = Callable[[Any, Any], "dict[str, str]"]

# Re-exported for the existing test imports (`from muscle.core import cosine`).
__all__ = [
    "MuscleMemory",
    "build_muscle_ground_fn",
    "build_muscle_verify_fn",
    "normalize_step_key",
    "cosine",
]


def normalize_step_key(desc: str) -> str:
    """Normalizes a step description into a stable lookup key."""
    return " ".join(desc.lower().split())


def plan_to_action(plan: ActionPlan) -> dict:
    """Serializes the replayable payload of an ActionPlan (no step_id: reused)."""
    return {"kind": plan.kind, "target": plan.target, "text": plan.text}


def action_to_plan(action: dict, step_id: str) -> ActionPlan:
    """Rebuilds an ActionPlan for `step_id`, restoring tuples JSON flattened.

    JSON turns the click target (x, y) and hotkey key lists into plain lists;
    the actuator expects tuples for those, so restore them here.
    """
    kind = action["kind"]
    target = action.get("target")
    if kind in ("click", "hotkey") and isinstance(target, list):
        target = tuple(target)
    return ActionPlan(kind=kind, step_id=step_id, target=target, text=action.get("text"))


@dataclass
class _Pending:
    """A Gemini grounding awaiting verification before it becomes a reflex."""

    site: str
    step_key: str
    embedding: list[float]
    action: dict


class MuscleMemory:
    """Learned local grounding tier: recall reflexes, stage/commit new ones.

    Counters (`local_hits`, `gemini_groundings`, `committed`, `heals`) drive the
    HUD's "Gemini calls this run" line and the headless proofs (a repeat run
    grounds with zero cloud calls; a healed run overwrites a stale reflex).
    """

    def __init__(
        self,
        store: MuscleStore,
        embed_fn: EmbedFn,
        threshold: float = 0.92,
        check: Check | None = None,
        site_cap: int = 20,
    ) -> None:
        """Wires the store, the injected encoder/Check and the eviction cap.

        Args:
            store: Persistence for reflexes.
            embed_fn: Screenshot -> embedding vector (fake in tests, a local
                encoder live). Used to build the default Check when none is given.
            threshold: Minimum cosine similarity for the default Check to trust a
                stored reflex (conservative by default: below it, fall through to
                Gemini). Ignored when an explicit `check` is passed.
            check: The pre-replay precondition (capture + compare). Defaults to an
                embedding Check over `embed_fn` at `threshold`.
            site_cap: Per-site reflex ceiling; the weakest are evicted past it
                (<= 0 disables eviction).
        """
        self._store = store
        self._embed_fn = embed_fn
        self._check = check if check is not None else build_default_check(embed_fn, threshold)
        self._site_cap = site_cap
        self._pending: dict[str, _Pending] = {}
        # step_id -> (site, step_key) for steps served by a recall HIT this run,
        # so a later verified commit can bump the reflex's recency/usage without
        # the read path ever writing (keeps recall read-only, risk R2).
        self._recalled: dict[str, tuple[str, str]] = {}
        # One-frame capture memo (see _capture): avoids embedding the same
        # screenshot twice when recall misses and stage then re-captures it.
        self._cap_key: Any = None
        self._cap_val: list[float] | None = None
        self.local_hits = 0
        self.gemini_groundings = 0
        self.committed = 0
        self.heals = 0

    # -- read path (never writes) -------------------------------------

    def recall(
        self,
        site: str,
        screenshot: Any,
        step_desc: str,
        params: dict[str, str] | None = None,
        step_id: str | None = None,
    ) -> ActionPlan | None:
        """Returns a replayable ActionPlan if a reflex matches AND validates.

        Strictly read-only (risk R2): no INSERT, no re-stage. Two gates run in
        order: a site+step_key lookup, then the pre-replay Check (risk R8) --
        `capture` reads the live screen and `compare` decides whether it still
        matches the screen the reflex was learned on. A lookup miss OR a Check
        miss returns None so the caller falls through to Gemini.

        Args:
            site: Current scope (app/host).
            screenshot: The current frame (captured and compared).
            step_desc: The step being grounded.
            params: Dynamic goal params, used to template the lookup key and to
                fill the recalled action back in (e.g. "type {query}" -> "boots").
            step_id: The id of the step being served. Passed so a HIT can be
                noted (in memory only, not the store) for a later verified commit
                to bump the reflex's recency -- keeps the read path write-free.

        Returns:
            An ActionPlan to replay (step_id filled when provided), or None when
            no stored reflex both matches the key and clears the Check.
        """
        step_key = self._step_key(step_desc, params)
        rows = self._store.lookup(site, step_key)
        if not rows:
            return None
        try:
            live = self._capture(screenshot)
        except Exception as error:  # pragma: no cover - live encoder path
            logger.warning("muscle: capture failed, deferring to Gemini: %s", error)
            return None
        for row in rows:
            if self._check.compare(row.embedding, live):
                self.local_hits += 1
                logger.info("muscle: local hit on %r -> 0 Gemini calls", step_key)
                if step_id is not None:
                    self._recalled[step_id] = (site, step_key)
                action = self._fill_action(row.action, params)
                return action_to_plan(action, step_id or "")
        logger.info("muscle: reflex for %r failed pre-replay Check -> fallback to Gemini", step_key)
        return None

    # -- write path (Gemini-gated, verification-committed) ------------

    def stage(
        self,
        step_id: str,
        site: str,
        screenshot: Any,
        step_desc: str,
        plan: ActionPlan,
        params: dict[str, str] | None = None,
    ) -> None:
        """Holds a Gemini grounding as PENDING until the step is verified done.

        Nothing is persisted here: an unverified grounding must never become a
        reflex (risk R1). Called only from the Gemini branch of grounding, so
        router fast-paths and local replays are never staged. The captured
        features and the action are templated with `params` so the stored reflex
        is reusable across dynamic arguments.
        """
        # Count the cloud call first: a real Gemini grounding happened even if
        # embedding the frame then fails, so the HUD's count must not undercount.
        self.gemini_groundings += 1
        try:
            embedding = self._capture(screenshot)
        except Exception as error:  # pragma: no cover - live encoder path
            logger.warning("muscle: could not capture for staging, skipping: %s", error)
            return
        self._pending[step_id] = _Pending(
            site=site,
            step_key=self._step_key(step_desc, params),
            embedding=embedding,
            action=self._templatize_action(plan_to_action(plan), params),
        )

    def commit(self, step_id: str) -> None:
        """Promotes a pending grounding to a stored reflex after verification.

        This is the ONLY path that writes to the store, and it fires only when
        `verify_step_done` confirmed the step -- the independent oracle that
        keeps the learning loop from grading its own homework. The write is an
        upsert: a verified re-grounding overwrites the stale reflex for this key
        (self-heal, risk R9), then the per-site cap evicts any overflow.
        """
        pending = self._pending.pop(step_id, None)
        if pending is None:
            # No fresh grounding to store. If this step was served by a recall
            # HIT, record the successful REPLAY so eviction ranking reflects it
            # (this is the only write on the recall-hit path, and it is still
            # verify-gated -- recall itself never wrote).
            recalled = self._recalled.pop(step_id, None)
            if recalled is not None:
                self._store.touch(*recalled)
            return
        self._recalled.pop(step_id, None)  # a fresh grounding supersedes any stale hit note
        healed = self._store.upsert(pending.site, pending.step_key, pending.embedding, pending.action)
        self._store.enforce_cap(pending.site, self._site_cap)
        self.committed += 1
        if healed:
            self.heals += 1
            logger.info("muscle: self-healed reflex for %r on %s", pending.step_key, pending.site)
        else:
            logger.info("muscle: committed reflex for step %s on %r", step_id, pending.step_key)

    def discard(self, step_id: str) -> None:
        """Drops a pending grounding / recall note that never verified (blocked)."""
        self._pending.pop(step_id, None)
        self._recalled.pop(step_id, None)

    def stats(self) -> dict[str, int]:
        """Returns run counters for the HUD / proof assertions."""
        return {
            "local_hits": self.local_hits,
            "gemini_groundings": self.gemini_groundings,
            "committed": self.committed,
            "heals": self.heals,
            "stored": self._store.count(),
        }

    # -- helpers ------------------------------------------------------

    def _capture(self, screenshot: Any) -> list[float]:
        """Captures the Check features for `screenshot`, memoized per frame.

        Within one turn, recall() and stage() can be handed the SAME frame (a
        Check miss falls through to Gemini, which then stages that frame);
        identity-memoizing avoids a second live encode of the same image on the
        self-heal path, where latency already hurts because Gemini is being
        called too.
        """
        if self._cap_key is screenshot and self._cap_val is not None:
            return self._cap_val
        val = self._check.capture(screenshot)
        self._cap_key = screenshot
        self._cap_val = val
        return val

    @staticmethod
    def _step_key(step_desc: str, params: dict[str, str] | None) -> str:
        """Normalizes then templates the step description into the cache key."""
        return templatize(normalize_step_key(step_desc), params)

    @staticmethod
    def _templatize_action(action: dict, params: dict[str, str] | None) -> dict:
        """Genericizes an action's text so the stored reflex is param-invariant."""
        if not params or not action.get("text"):
            return action
        return {**action, "text": templatize(action["text"], params)}

    @staticmethod
    def _fill_action(action: dict, params: dict[str, str] | None) -> dict:
        """Substitutes live params back into a recalled action's text."""
        if not params or not action.get("text"):
            return action
        return {**action, "text": fill_template(action["text"], params)}


def build_muscle_ground_fn(
    inner_gemini_ground: Callable[[Any, Any, Any], ActionPlan | None],
    muscle: MuscleMemory,
    site_fn: Callable[[Any, Any], str],
    params_fn: ParamsFn | None = None,
) -> Callable[[Any, Any, Any], ActionPlan | None]:
    """Wraps a Gemini-only grounding callable with the local recall tier.

    The wrapped `inner_gemini_ground` must represent the CLOUD decision only
    (no router fast-path, no muscle) -- this wrapper adds tier 1.5:

        recall (read-only, pre-replay Check) -> hit? replay locally, 0 Gemini
                                             -> miss? Gemini grounds, STAGED pending

    The fallback on a Check miss is a MID-TASK handoff, not a blind restart
    (risk R9): `inner_gemini_ground(task, step, observation)` re-grounds only the
    CURRENT step, with the task carrying the goal and the already-done steps and
    the observation carrying the live screen. A subsequent verify then commits,
    which OVERWRITES the stale reflex (self-heal).

    The returned plan flows back through the loop's normal `is_dangerous` guard
    exactly like a fresh Gemini plan (risk R3): this wrapper never bypasses it.
    """

    def _ground(task: Any, step: Any, observation: Any) -> ActionPlan | None:
        screenshot = getattr(observation, "screenshot", None)
        site = site_fn(task, observation)
        params = params_fn(task, observation) if params_fn is not None else None
        if screenshot is not None:
            hit = muscle.recall(site, screenshot, step.desc, params, step_id=step.id)
            if hit is not None:
                return hit
        # Fallback: hand the current step back to Gemini with full context.
        plan = inner_gemini_ground(task, step, observation)
        if plan is not None and screenshot is not None:
            muscle.stage(step.id, site, screenshot, step.desc, plan, params)
        return plan

    return _ground


def build_muscle_verify_fn(
    inner_verify: Callable[[Any, Any, Any], bool],
    muscle: MuscleMemory,
) -> Callable[[Any, Any, Any], bool]:
    """Wraps the step verifier so a confirmed step commits its pending reflex.

    Verification is the oracle: only when the inner verifier says the step is
    done does the staged Gemini grounding become a stored reflex.
    """

    def _verify(task: Any, step: Any, screenshot: Any) -> bool:
        done = bool(inner_verify(task, step, screenshot))
        if done:
            muscle.commit(step.id)
        return done

    return _verify
