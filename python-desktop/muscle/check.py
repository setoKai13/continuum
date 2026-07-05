"""The `Check = capture + compare` abstraction (borrowed from muscle-mem).

A `Check` is Continuum's pre-replay precondition validation (risk R8): before a
cached step is fired, `capture(screenshot) -> features` reads a cheap feature of
the LIVE screen and `compare(cached_features, live_features) -> bool` decides
whether that screen still matches the one the reflex was learned on. Only on a
pass does the recalled action replay; a fail falls through to Gemini (the loop
never clicks blindly on a stale screen).

Continuum is a PIXEL agent with no DOM selectors, so `capture` is deliberately a
screenshot feature (an embedding here; a region hash or expected-text probe would
slot in the same way) -- never a selector-resolves check that cannot run in this
repo. The encoder stays injected (`embed_fn`): a deterministic fake in tests, a
lazy local model live (`embedders.build_default_embed_fn`).

Only stdlib math is used, so importing this never pulls in a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

# Screenshot -> feature vector (the injected encoder). Kept generic so a future
# region-hash / OCR capture drops in without touching the recall path.
CaptureFn = Callable[[Any], list[float]]
# (cached_features, live_features) -> is the live screen still valid to replay on?
CompareFn = Callable[[list[float], list[float]], bool]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 on a zero vector)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Check:
    """A precondition: how to read the live screen and how to validate it.

    Attributes:
        capture: Screenshot -> features. Recorded per step at grounding time and
            re-run on the live screen before replay.
        compare: (cached_features, live_features) -> bool. The pre-replay gate;
            below it the caller falls through to Gemini.
    """

    capture: CaptureFn
    compare: CompareFn


def build_default_check(embed_fn: CaptureFn, threshold: float = 0.92) -> Check:
    """Builds the default embedding Check: capture = encoder, compare = cosine floor.

    Args:
        embed_fn: The injected screenshot encoder (fake in tests, lazy CLIP live).
        threshold: Conservative cosine floor (risk R4); below it the live screen
            is judged too different and replay is refused.

    Returns:
        A `Check` whose `compare` passes only at/above `threshold`.
    """

    def _compare(cached: list[float], live: list[float]) -> bool:
        return cosine(cached, live) >= threshold

    return Check(capture=embed_fn, compare=_compare)
