"""Goal templating: fold dynamic `params` into `{slot}`s so the cache key is reusable.

Without this, "search for shoes" and "search for boots" would be two distinct
cache keys and the hit-rate collapses. With a `params` dict `{"query": "shoes"}`,
`templatize` rewrites the step into the param-invariant template
"search for {query}", which is what gets stored and looked up; `fill_template`
does the reverse on replay, substituting the LIVE param value back into the
recalled action ("type {query}" -> "type boots").

This is the exact/templated-string stage from the design playbook -- fuzzy
embedding match on goals is deliberately v2. Pure stdlib `re`, so it stays
headless-safe.
"""

from __future__ import annotations

import re


def templatize(text: str, params: dict[str, str] | None) -> str:
    """Replaces each param VALUE in `text` with its `{name}` slot.

    Longest values are substituted first so an overlapping shorter value cannot
    eat part of a longer one; matching is case-insensitive because step keys are
    lowercased upstream while param values may not be.

    Args:
        text: The step/action text to genericize.
        params: Dynamic argument values for this run (e.g. {"query": "shoes"}).

    Returns:
        `text` with every param value swapped for its `{name}` placeholder.
    """
    if not params:
        return text
    # TODO(params): this replaces param values as case-insensitive SUBSTRINGS, so
    # a short value like "in" would corrupt "login" -> "log{query}". Harmless
    # today because the wired path passes no params (main._params_key -> {}), but
    # before params go live switch to word-boundary matching and skip very short
    # values. (Flagged in the v1 code review, finding #4.)
    out = text
    for name, value in sorted(params.items(), key=lambda kv: -len(str(kv[1]))):
        value = str(value)
        if not value:
            continue
        out = re.sub(re.escape(value), "{" + name + "}", out, flags=re.IGNORECASE)
    return out


def fill_template(text: str, params: dict[str, str] | None) -> str:
    """Substitutes live param values back into a `{name}`-slotted template.

    The inverse of `templatize`: turns the stored generic action text back into
    a concrete one for this run's params.

    Args:
        text: Template text containing `{name}` slots.
        params: This run's param values.

    Returns:
        `text` with every `{name}` replaced by its param value.
    """
    if not params:
        return text
    out = text
    for name, value in params.items():
        out = out.replace("{" + name + "}", str(value))
    return out
