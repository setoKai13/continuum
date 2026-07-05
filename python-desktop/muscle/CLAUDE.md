# Muscle Memory — agent guide (read this before touching `muscle/`)

This folder is Continuum's **local grounding cache**. Continuum records clicks it
grounded via Gemini and *verified done*, then replays them locally when the same
step recurs on the same site — so a repeated task skips the Gemini call: faster,
free, offline. **It is a CACHE (faster on solved tasks), not learning (it does
not get smarter on novel screens).** Never describe it as "self-improving/RL".

Continuum is a **pixel/screenshot vision agent** (Gemini returns a box, we click
pixels) — there are **no DOM selectors**. Any "does this still match?" check here
is a screenshot/embedding check, not a selector check. Don't port DOM code in.

## The one load-bearing rule (safety)

**Local reads, Gemini-gated writes.** A reflex is stored ONLY from a step that was
grounded by Gemini AND confirmed by `verify_step_done`. The recall path is
strictly read-only. Recalled actions still flow back through the loop's
`is_dangerous()` gate before ACT — muscle code never calls the actuator.

```
recall(site, screenshot, step)  -> read-only; Check passes -> replay, 0 Gemini
stage(step_id, ...)             -> a Gemini grounding held PENDING (not trusted)
commit(step_id)                 -> verify passed -> pending becomes a stored reflex
                                   (and self-heals/overwrites a stale one)
```

The only nuance: a *verified replay* bumps the reflex's recency via
`commit -> store.touch` (so eviction ranks proven reflexes last). That write is
still verify-gated; `recall` itself never writes.

## How it wires into the loop (DECIDE, cheapest tier first)

`main.build_ground_fn`: **router (tier 1, zero-LLM)** → **muscle (tier 1.5, local
recall)** → **Gemini (tier 2)**. On a recall miss, the same turn falls back to
Gemini *with full context* (mid-task handoff, not a restart); a later verify
commits and OVERWRITES the stale reflex (self-heal). `main.build_verify_fn` wraps
the verifier so a confirmed step commits. Both are optional/flag-guarded
(`MUSCLE_ENABLED`, `MUSCLE_THRESHOLD`, `MUSCLE_SITE_CAP` in `config.py`).

## Key concepts (borrowed from muscle-mem, hand-rolled here — see decision below)

- **`Check = capture + compare`** (`check.py`): the pre-replay validation. `capture`
  reads live-screen features; `compare` decides if the screen still matches the one
  the reflex was learned on. A miss → fallback (never a blind click on a stale
  screen).
- **Goal templating / `params`** (`templating.py`): cache key is `(site,
  goal_template)` with `{param}` slots, so "search for shoes"/"boots" share one
  reflex. Currently the wired path passes no params (`main._params_key -> {}`).
- **Per-site eviction** (`store.enforce_cap`): keep ~`site_cap` most-proven/recent
  reflexes per site, evict lowest `success_count` then oldest `last_used_at`.

## File map

| File | Role |
|---|---|
| `core.py` | `MuscleMemory` (recall/stage/commit/touch), the ground/verify wrappers, capture memo |
| `check.py` | `Check(capture, compare)` + default embedding/cosine Check |
| `store.py` | SQLite `muscle_memory` table (own file/table; upsert=self-heal, touch, enforce_cap) |
| `templating.py` | `templatize` / `fill_template` for goal params |
| `embedders.py` | live CLIP encoder, **lazily** imported (tests inject a fake) |
| `README.md` | short human/GitHub overview |

## Repo laws (this codebase is strict — a change breaking one is wrong)

1. **Dependency injection**: collaborators are injected callables (see `main.build_*_fn`).
2. **Lazy heavy imports**: models/GUI/net libs import *inside functions*. Invariant:
   `python -c "import main"` works with only pydantic/pydantic-settings/pytest/rich.
3. **Config via pydantic `Settings`** only (no `os.getenv`), safe defaults.
4. **Persistence is SQLite**, wipeable (`MuscleStore.clear()`).
5. **Safety is the loop's job** — never call the actuator or bypass `is_dangerous()`.

## Prove it (headless, no key/screen/mic)

```bash
cd python-desktop
python3.12 -m venv .venv && .venv/bin/pip install pydantic pydantic-settings pytest rich   # first time
.venv/bin/python -c "import main"             # lazy-import invariant
.venv/bin/pytest -q                           # full suite (incl. tests/test_muscle.py)
.venv/bin/python scripts/dry_run.py           # -> DRY-RUN OK (no regression)
.venv/bin/python scripts/muscle_dry_run.py    # -> MUSCLE DRY-RUN OK (learn / recall / self-heal)
```

## Safety risks this code must keep upholding (verify before merging changes)

R1 no closed loop (only verified Gemini groundings write) · R2 recall is read-only ·
R3 recalled actions still pass `is_dangerous()` · R4 conservative, configurable
threshold · R5 no eval on non-held-out data · R6 deterministic + wipeable · R7
`site`/`created_at` stamped, site-scoped · R8 validate (Check) BEFORE replay · R9
fallback preserves context + self-heals by overwrite · R10 it's a cache, not learning.

## Settled decisions & scope

- **Rebuild, don't depend on muscle-mem (pig.dev).** Its `Check` is validation-
  agnostic (not DOM-only), but it sits in the decision hot path and would break the
  headless zero-deps proof, and Continuum's `AgentLoop` already *is* the engine. We
  borrowed its vocabulary (`Check`/capture/compare, `params`), not the package.
- **v1 (shipped):** record, recall+Check validation, fallback+handoff, self-heal,
  per-site eviction, goal templates, Gemini-gated writes.
- **v2 (later):** pitfall memory (learn from failures), fuzzy goal match, promoting
  recurring plans into reusable skills (Metis), per-app/URL `site` keys (today
  `main._site_key` returns `"default"`), deeper non-stationarity than staleness.
- **Known dormant issue:** `templatize` does case-insensitive *substring* replace —
  harmless now (no params in the wired path) but needs word-boundary matching before
  params go live (see the `TODO(params)` in `templating.py`).

## Prior art / vocabulary (for the v2 direction)

Self-evolving / experience memory: **muscle-mem** (trajectory caching w/ fallback),
**Evo-Memory** (test-time self-evolving memory benchmark), **Metis** (text+code
memory, distills reusable tools + pitfalls). Search: episodic/procedural memory,
experience replay, test-time learning, reflection, continual learning.
