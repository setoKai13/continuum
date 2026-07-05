# Muscle Memory

Continuum's self-improving local grounding tier. The agent learns to ground
clicks locally from its own **verified** successes, so a repeated step skips the
Gemini call: **faster, free, offline, and better the more it is used.**

Resume brings back *where* the task was; Muscle Memory brings back *how to move*.

## The one rule (why the loop stays honest)

**Local reads, Gemini-gated writes.** A reflex is written to the store ONLY after
a step was grounded by Gemini **and** later confirmed by `verify_step_done`. The
recall path is strictly read-only. So every reflex traces back to an independent
cloud decision the verifier approved — the loop never grades its own homework.

```
recall(site, screenshot, step)   # read-only; hit above threshold -> replay, 0 Gemini
stage(step_id, ...)              # a Gemini grounding, held PENDING (not trusted yet)
commit(step_id)                  # verify passed -> pending becomes a stored reflex
```

## Files

| File | Role |
|---|---|
| `core.py` | `MuscleMemory` (recall/stage/commit), cosine sim, plan↔row serialization, and the `build_muscle_ground_fn` / `build_muscle_verify_fn` wrappers |
| `store.py` | `MuscleStore` — own SQLite table `muscle_memory` (never touches `memory.py`) |
| `embedders.py` | live local image encoder (CLIP), imported lazily; tests inject a fake |
| `__init__.py` | public API |

## How it wires in (`main.py`, all optional / flag-guarded)

Tier 1 `router` → **tier 1.5 `muscle`** → tier 2 `vision` (Gemini). A recalled
action returns through the loop's normal path, so it still passes the
`is_dangerous()` safety gate exactly like a fresh Gemini action. Controlled by
`MUSCLE_ENABLED` / `MUSCLE_THRESHOLD` in `.env`; if no local encoder is
installed, the agent logs it and runs Gemini-only (never crashes).

## Prove it (headless, no key / screen / mic)

```bash
.venv/bin/python scripts/muscle_dry_run.py   # -> "MUSCLE DRY-RUN OK"
.venv/bin/pytest tests/test_muscle.py -q
```

`scripts/muscle_dry_run.py` runs the **real** `AgentLoop` twice on the same task:
run 1 grounds via (fake) Gemini and learns; run 2 grounds every step locally with
**zero** Gemini calls and replays identical actions.

## PoC scope

**In:** verification-gated writes, site-scoped local recall, conservative
similarity threshold, clean wipe (`MuscleStore.clear()`).
**Out (deliberately, for now):** staleness decay / eviction (non-stationarity),
online threshold learning, per-app/URL `site` keys (PoC uses `"default"` — see
`_site_key` in `main.py`), any cloud/distributed infra.

Risks and how to verify them: `.claude/muscle-memory-risk-check`.
