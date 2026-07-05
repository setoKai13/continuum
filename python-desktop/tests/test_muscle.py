"""Muscle Memory unit tests: the reflex store, and the closed-safe learning loop.

The headline proof (test_second_run_uses_zero_gemini_calls) is the "wow" made
mechanical: an identical run, after learning, grounds every step locally with
zero cloud calls. The rest lock in the safety invariants from the risk-check
skill (writes are Gemini+verify gated; recall never writes; below-threshold
falls through).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agent import ActionPlan
from muscle import (
    Check,
    MuscleMemory,
    MuscleStore,
    build_muscle_ground_fn,
    build_muscle_verify_fn,
    fill_template,
    templatize,
)
from muscle.core import cosine, normalize_step_key


class FakeEmbedder:
    """Deterministic screenshot -> vector: identical frames embed identically."""

    def __init__(self) -> None:
        self._table: dict[str, list[float]] = {
            "screen_login": [1.0, 0.0, 0.0],
            "screen_inbox": [0.0, 1.0, 0.0],
            "screen_other": [0.0, 0.0, 1.0],
        }

    def __call__(self, screenshot: str) -> list[float]:
        # Unknown frames get a stable per-token vector so tests stay explicit.
        return self._table.get(screenshot, [0.5, 0.5, 0.0])


class Obs:
    """Minimal Observation stand-in (only .screenshot is read by the wrapper)."""

    def __init__(self, screenshot: str) -> None:
        self.screenshot = screenshot


class Step:
    """Minimal Step stand-in (only .id and .desc are read)."""

    def __init__(self, step_id: str, desc: str) -> None:
        self.id = step_id
        self.desc = desc


def _muscle(tmp: Path, threshold: float = 0.92) -> MuscleMemory:
    store = MuscleStore(tmp / "muscle.db")
    return MuscleMemory(store=store, embed_fn=FakeEmbedder(), threshold=threshold)


def test_cosine_and_key_helpers() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([], [1.0]) == 0.0
    assert normalize_step_key("  Click   SEND ") == "click send"


def test_recall_miss_then_hit_after_verified_commit() -> None:
    with tempfile.TemporaryDirectory() as d:
        m = _muscle(Path(d))
        # Cold: nothing stored yet.
        assert m.recall("default", "screen_login", "click login") is None
        # A Gemini grounding is staged (pending), but must NOT be recallable yet.
        plan = ActionPlan(kind="click", step_id="s1", target=(10, 20), text="the login button")
        m.stage("s1", "default", "screen_login", "click login", plan)
        assert m.recall("default", "screen_login", "click login") is None, "pending must not be recallable"
        # Verification commits it -> now it is a reflex.
        m.commit("s1")
        hit = m.recall("default", "screen_login", "click login")
        assert hit is not None
        assert hit.kind == "click" and hit.target == (10, 20)


def test_discard_prevents_commit() -> None:
    with tempfile.TemporaryDirectory() as d:
        m = _muscle(Path(d))
        m.stage("s1", "default", "screen_login", "click login",
                ActionPlan(kind="click", step_id="s1", target=(1, 2)))
        m.discard("s1")
        m.commit("s1")  # no pending -> no-op
        assert m.recall("default", "screen_login", "click login") is None
        assert m.stats()["stored"] == 0


def test_below_threshold_falls_through() -> None:
    with tempfile.TemporaryDirectory() as d:
        m = _muscle(Path(d), threshold=0.99)
        m.stage("s1", "default", "screen_login", "click login",
                ActionPlan(kind="click", step_id="s1", target=(1, 2)))
        m.commit("s1")
        # A different screen for the same step: similarity below the floor -> miss.
        assert m.recall("default", "screen_other", "click login") is None


def test_site_scoping_isolates_reflexes() -> None:
    with tempfile.TemporaryDirectory() as d:
        m = _muscle(Path(d))
        m.stage("s1", "slack", "screen_inbox", "click send",
                ActionPlan(kind="click", step_id="s1", target=(3, 4)))
        m.commit("s1")
        assert m.recall("slack", "screen_inbox", "click send") is not None
        assert m.recall("gmail", "screen_inbox", "click send") is None, "reflexes must not leak across sites"


def test_recall_is_read_only() -> None:
    with tempfile.TemporaryDirectory() as d:
        m = _muscle(Path(d))
        m.stage("s1", "default", "screen_login", "click login",
                ActionPlan(kind="click", step_id="s1", target=(1, 2)))
        m.commit("s1")
        before = m.stats()["stored"]
        for _ in range(5):
            m.recall("default", "screen_login", "click login")
        assert m.stats()["stored"] == before, "recall must never write to the store"


def test_clear_is_a_clean_rollback() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = MuscleStore(Path(d) / "muscle.db")
        m = MuscleMemory(store=store, embed_fn=FakeEmbedder())
        m.stage("s1", "default", "screen_login", "click login",
                ActionPlan(kind="click", step_id="s1", target=(1, 2)))
        m.commit("s1")
        assert store.count() == 1
        store.clear()
        assert store.count() == 0
        assert m.recall("default", "screen_login", "click login") is None


def test_reflex_persists_across_processes() -> None:
    """A reflex committed in one 'process' is recalled by a fresh store on the same db."""
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "muscle.db"
        m1 = MuscleMemory(store=MuscleStore(db), embed_fn=FakeEmbedder())
        m1.stage("s1", "default", "screen_login", "click login",
                 ActionPlan(kind="click", step_id="s1", target=(7, 8)))
        m1.commit("s1")
        # Fresh MuscleMemory + fresh store on the same file = a new run.
        m2 = MuscleMemory(store=MuscleStore(db), embed_fn=FakeEmbedder())
        hit = m2.recall("default", "screen_login", "click login")
        assert hit is not None and hit.target == (7, 8)


def test_second_run_uses_zero_gemini_calls() -> None:
    """THE proof: a repeated run grounds every step locally, 0 Gemini calls.

    A counting fake stands in for the Gemini grounding. Run 1 misses on every
    step (calls Gemini, stages, then a verify commits the reflex). Run 2, on the
    identical screens/steps, recalls every step locally -- the Gemini counter
    stays at zero.
    """
    steps = [Step("s1", "click login"), Step("s2", "click inbox")]
    screen_for = {"s1": "screen_login", "s2": "screen_inbox"}

    class CountingGemini:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, task, step, observation):
            self.calls += 1
            return ActionPlan(kind="click", step_id=step.id, target=(step.id_num(), 0))

    # Give Step a stable numeric target so replayed clicks are checkable.
    Step.id_num = lambda self: int(self.id[1:])  # type: ignore[attr-defined]

    def run(muscle: MuscleMemory) -> tuple[int, list[ActionPlan]]:
        gemini = CountingGemini()
        ground = build_muscle_ground_fn(gemini, muscle, site_fn=lambda t, o: "default")
        verify = build_muscle_verify_fn(lambda t, s, shot: True, muscle)
        plans: list[ActionPlan] = []
        for step in steps:
            obs = Obs(screen_for[step.id])
            plan = ground(task=None, step=step, observation=obs)
            plans.append(plan)
            # The loop verifies the step on a later turn -> commits the reflex.
            verify(None, step, obs.screenshot)
        return gemini.calls, plans

    with tempfile.TemporaryDirectory() as d:
        m = _muscle(Path(d))
        calls_1, plans_1 = run(m)
        calls_2, plans_2 = run(m)

    assert calls_1 == 2, "run 1 must ground both steps via Gemini"
    assert calls_2 == 0, "run 2 must ground both steps locally -- zero Gemini calls"
    assert [p.target for p in plans_1] == [p.target for p in plans_2], "replayed actions must match"
    assert m.stats()["local_hits"] == 2


# -- v1: Check (pre-replay validation), templating, self-heal, eviction --------


def test_templating_helpers() -> None:
    params = {"query": "shoes"}
    assert templatize("search for shoes", params) == "search for {query}"
    assert templatize("Search For SHOES", params) == "Search For {query}"  # case-insensitive
    assert fill_template("type {query}", {"query": "boots"}) == "type boots"
    # No params -> identity both ways (v0 behavior preserved).
    assert templatize("click send", None) == "click send"
    assert fill_template("click send", None) == "click send"


def test_check_compare_gates_replay_before_execute() -> None:
    """The pre-replay Check (R8) must run and a compare-miss must fall through."""
    calls = {"compare": 0}

    def _capture(shot: str) -> list[float]:
        return {"A": [1.0, 0.0], "B": [0.0, 1.0]}.get(shot, [0.0, 0.0])

    def _compare(cached: list[float], live: list[float]) -> bool:
        calls["compare"] += 1
        return cached == live

    with tempfile.TemporaryDirectory() as d:
        m = MuscleMemory(
            store=MuscleStore(Path(d) / "m.db"),
            embed_fn=_capture,
            check=Check(capture=_capture, compare=_compare),
        )
        m.stage("s1", "default", "A", "click go", ActionPlan(kind="click", step_id="s1", target=(1, 2)))
        m.commit("s1")
        # Same screen: Check passes -> replay.
        assert m.recall("default", "A", "click go") is not None
        # Changed screen: Check must run and REFUSE (no blind click on a stale screen).
        assert m.recall("default", "B", "click go") is None
        assert calls["compare"] >= 2, "compare must gate every replay attempt"


def test_templated_reflex_recalls_across_params() -> None:
    """One reflex for 'search for {query}' serves any query value (goal templating)."""
    with tempfile.TemporaryDirectory() as d:
        m = _muscle(Path(d))
        # Learn on 'shoes'; the typed text IS the dynamic param.
        m.stage(
            "s1", "amazon", "screen_inbox", "search for shoes",
            ActionPlan(kind="type", step_id="s1", text="shoes"), params={"query": "shoes"},
        )
        m.commit("s1")
        assert m.stats()["stored"] == 1
        # A different query on the same screen hits the SAME template and fills 'boots'.
        hit = m.recall("amazon", "screen_inbox", "search for boots", params={"query": "boots"})
        assert hit is not None, "a new param value must reuse the templated reflex"
        assert hit.text == "boots", "the recalled action must be filled with the live param"


def test_fallback_miss_heals_and_overwrites_stale_trajectory() -> None:
    """A Check miss falls back to Gemini; the fresh success OVERWRITES the stale reflex (R9)."""
    with tempfile.TemporaryDirectory() as d:
        m = _muscle(Path(d), threshold=0.99)  # strict: a changed screen misses
        # Cold learn: 'click send' on the login screen -> target (1, 2).
        m.stage("s1", "app", "screen_login", "click send",
                ActionPlan(kind="click", step_id="s1", target=(1, 2)))
        m.commit("s1")
        assert m.stats() == {"local_hits": 0, "gemini_groundings": 1, "committed": 1, "heals": 0, "stored": 1}

        # Site redesign: the same step now lives on a different screen -> recall MISS.
        assert m.recall("app", "screen_other", "click send") is None, "stale screen must not replay"

        # Fallback re-grounds with a NEW target and verifies -> heal (overwrite, not append).
        m.stage("s2", "app", "screen_other", "click send",
                ActionPlan(kind="click", step_id="s2", target=(9, 9)))
        m.commit("s2")
        assert m.stats()["stored"] == 1, "heal must OVERWRITE the stale reflex, not accumulate"
        assert m.stats()["heals"] == 1

        # The reflex now matches the NEW screen; the OLD screen no longer replays.
        healed = m.recall("app", "screen_other", "click send")
        assert healed is not None and healed.target == (9, 9)
        assert m.recall("app", "screen_login", "click send") is None, "the stale trajectory is gone"


def test_per_site_eviction_caps_the_store() -> None:
    """Past the per-site cap, the oldest/weakest reflex is evicted (bounded store)."""
    embedder = FakeEmbedder()
    embedder._table.update({"scr_a": [1.0, 0.0, 0.0], "scr_b": [0.0, 1.0, 0.0], "scr_c": [0.0, 0.0, 1.0]})
    with tempfile.TemporaryDirectory() as d:
        store = MuscleStore(Path(d) / "m.db")
        m = MuscleMemory(store=store, embed_fn=embedder, threshold=0.9, site_cap=2)
        for sid, (scr, desc) in enumerate([("scr_a", "step a"), ("scr_b", "step b"), ("scr_c", "step c")]):
            m.stage(f"s{sid}", "site", scr, desc, ActionPlan(kind="click", step_id=f"s{sid}", target=(sid, sid)))
            m.commit(f"s{sid}")
        assert store.count_for_site("site") == 2, "per-site cap must bound the store"
        # The oldest reflex ('step a') was evicted; the newest ('step c') survives.
        assert m.recall("site", "scr_a", "step a") is None
        assert m.recall("site", "scr_c", "step c") is not None


def test_replayed_reflex_survives_eviction_over_unused_one() -> None:
    """Recency touch (review finding #1): a REPLAYED reflex outlives an unused one.

    Recall stays read-only, but each *verified* replay bumps the reflex's usage
    via commit -> store.touch, so eviction ranks it as proven rather than "unused".
    This is the exact case the plain eviction test cannot catch: here 'step a' is
    the OLDEST reflex, so without the recency touch it would be evicted first
    (like in the test above) -- the touch is what keeps it alive instead of the
    never-replayed 'step b'.
    """
    embedder = FakeEmbedder()
    embedder._table.update({"scr_a": [1.0, 0.0, 0.0], "scr_b": [0.0, 1.0, 0.0], "scr_c": [0.0, 0.0, 1.0]})
    with tempfile.TemporaryDirectory() as d:
        store = MuscleStore(Path(d) / "m.db")
        m = MuscleMemory(store=store, embed_fn=embedder, threshold=0.9, site_cap=2)

        # Learn A (oldest) and B, filling the cap.
        m.stage("a", "site", "scr_a", "step a", ActionPlan(kind="click", step_id="a", target=(1, 1)))
        m.commit("a")
        m.stage("b", "site", "scr_b", "step b", ActionPlan(kind="click", step_id="b", target=(2, 2)))
        m.commit("b")

        # Replay A many times: read-only recall HIT -> verified commit -> recency touch.
        for i in range(5):
            hit = m.recall("site", "scr_a", "step a", step_id=f"replay{i}")
            assert hit is not None, "A must replay locally"
            m.commit(f"replay{i}")  # verify passed: bumps usage, adds no row
        assert store.count_for_site("site") == 2, "replays stay read-only -> no store growth"

        # A new reflex C overflows the cap and triggers eviction.
        m.stage("c", "site", "scr_c", "step c", ActionPlan(kind="click", step_id="c", target=(3, 3)))
        m.commit("c")

        assert store.count_for_site("site") == 2, "cap still bounds the store"
        # B (never replayed) is evicted; A (heavily replayed, though oldest) survives.
        assert m.recall("site", "scr_b", "step b") is None, "the UNUSED reflex is evicted"
        assert m.recall("site", "scr_a", "step a") is not None, "the replayed reflex must survive eviction"
        assert m.recall("site", "scr_c", "step c") is not None, "the fresh reflex survives"
