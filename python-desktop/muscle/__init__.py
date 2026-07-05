"""Muscle Memory -- Continuum's local grounding cache.

Continuum records clicks it grounded via Gemini and verified done, then replays
them locally when the same step recurs on the same site -- so a repeated task
skips the Gemini call: faster, free, offline. It is a CACHE (faster on tasks it
has already solved), not learning (it does not get smarter on novel screens).
This package is fully self-contained -- wiring it in is a handful of optional
lines in `main.py` -- so the rest of the codebase stays readable.

Scope (see .claude/muscle-memory-risk-check): local reads, Gemini-gated writes, a
conservative pre-replay Check, per-`site` scoping, self-heal on stale screens, a
per-site eviction cap, and a clean wipe. Online threshold learning, pitfall
memory, fuzzy goal match and any cloud/distributed infra are out of scope for now.

Public API:
    MuscleMemory                -- recall / stage / commit the reflex store
    MuscleStore                 -- SQLite persistence (own table)
    Check                       -- capture + compare pre-replay validation (R8)
    build_default_check         -- default embedding Check (cosine threshold)
    templatize / fill_template  -- goal param templating for reusable cache keys
    build_muscle_ground_fn      -- adds the local recall tier to a Gemini ground fn
    build_muscle_verify_fn      -- commits a reflex when a step verifies done
    build_default_embed_fn      -- live local image encoder (lazy)
"""

from .check import Check, build_default_check, cosine
from .core import (
    MuscleMemory,
    build_muscle_ground_fn,
    build_muscle_verify_fn,
    normalize_step_key,
)
from .store import MuscleStore
from .templating import fill_template, templatize

__all__ = [
    "MuscleMemory",
    "MuscleStore",
    "Check",
    "build_default_check",
    "cosine",
    "templatize",
    "fill_template",
    "build_muscle_ground_fn",
    "build_muscle_verify_fn",
    "normalize_step_key",
    "build_default_embed_fn",
]


def build_default_embed_fn(*args, **kwargs):
    """Lazy re-export of the live encoder builder (keeps model imports lazy)."""
    from .embedders import build_default_embed_fn as _impl

    return _impl(*args, **kwargs)
