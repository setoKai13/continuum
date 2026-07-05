"""Live screenshot encoders, imported lazily so the package stays headless-safe.

`core.py` takes the encoder as an injected `embed_fn`; tests inject a
deterministic fake and never touch this file. Only a real live run calls
`build_default_embed_fn()`, which pulls in a local CLIP-style model behind a
lazy import (never at package import time, matching the project rule that no
GUI/model dependency loads on `import main`).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

EmbedFn = Callable[[Any], list[float]]


def build_default_embed_fn() -> EmbedFn:
    """Builds a local image embedder (CLIP via sentence-transformers).

    Deterministic and offline once the model weights are cached: the same
    screenshot always yields the same vector, so a repeat run recalls its own
    past reflexes. Raises a clear error if the optional dependency is absent,
    so `main` can log it and simply run Gemini-only instead of crashing.

    Returns:
        A callable mapping a PIL image to a list-of-floats embedding.

    Raises:
        RuntimeError: If no local embedding backend is installed.
    """
    try:
        from sentence_transformers import SentenceTransformer  # lazy: heavy model dep
    except Exception as error:  # noqa: BLE001 - optional dependency
        raise RuntimeError(
            "Muscle Memory needs a local image encoder. Install one, e.g. "
            "`pip install sentence-transformers`, or set MUSCLE_ENABLED=false."
        ) from error

    model = SentenceTransformer("clip-ViT-B-32")

    def _embed(screenshot: Any) -> list[float]:
        vector = model.encode(screenshot, normalize_embeddings=True)
        return [float(x) for x in vector]

    return _embed
