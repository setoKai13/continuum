"""IntentRouter: security gate + zero-LLM fast paths for cheap intents.

Reimplements (not forked) the "blocklist first, then regex fast-paths"
pattern: every incoming instruction is checked against a destructive-action
blocklist BEFORE anything else runs (agent.py refuses the turn outright if
`is_dangerous()` fires). Only after that gate passes do we try fast regex
matches for trivially-cheap intents (open an app, open a URL) so we don't
burn a Gemini call on "open Slack".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Patterns that must never be auto-executed, regardless of phrasing. This is
# deliberately conservative: prefer a false positive (asks for confirmation)
# over a false negative (silently wipes something).
_DANGEROUS_PATTERNS: tuple[str, ...] = (
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bsupprime(r)?\s+tout\b",
    r"\bdelete\s+everything\b",
    r"\bformat(er)?\s+(le\s+)?disque\b",
    r"\bformat\s+disk\b",
    r"\bwipe\b",
    r"\bfactory\s+reset\b",
    r"\bvide(r)?\s+la\s+corbeille\b",
    r"\bempty\s+(the\s+)?trash\b",
    r"\bdesinstalle(r)?\b",
    r"\buninstall\b",
    r"\bkill\s+-9\b",
    r"\bshutdown\b",
    r"\bteindre\s+le\s+mac\b",
)

_DANGEROUS_RE = re.compile("|".join(_DANGEROUS_PATTERNS), re.IGNORECASE)

# Markers that flag a spoken instruction as a live CORRECTION of the current
# plan rather than a brand-new task ("non, ...", "en fait ...", "actually ...").
# This is only the cheap zero-LLM gate: it decides WHETHER to spend a Gemini
# call extracting the override (vision.extract_override), never WHAT the
# override is. Conservative on purpose: a missed marker just means the phrase
# is treated as a plain instruction, which is the safe default.
_CORRECTION_MARKERS: tuple[str, ...] = (
    r"^\s*(?:non|no|nope)\b",
    r"\ben fait\b",
    r"\bactually\b",
    r"\bplut[oô]t\b",
    r"\binstead\b",
    r"\bcorrection\b",
    r"\bcorrige\b",
    r"\bje me suis tromp",
    r"\bmy mistake\b",
    r"\bd[ée]sormais\b",
    r"\bmaintenant\s+c['’]est\b",
    r"\b[aà] partir de maintenant\b",
    r"\bfrom now on\b",
    r"\bpas\s+(?:[aà]|dans|sur|vers|en)\b.+\bmais\b",
    r"\bnot\b.+\bbut\b",
)

_CORRECTION_RE = re.compile("|".join(_CORRECTION_MARKERS), re.IGNORECASE)

_OPEN_URL_RE = re.compile(r"\bouvre[rz]?\s+(?P<url>https?://\S+)|open\s+(?P<url2>https?://\S+)", re.IGNORECASE)
_OPEN_APP_RE = re.compile(
    r"(?:ouvre[rz]?|lance[rz]?|open|launch)\s+(?:l['’]application\s+|the\s+app\s+|l['’]app\s+)?(?P<app>[A-Za-z][\w \-]{1,40})$",
    re.IGNORECASE,
)
# The fast path must only fire on REAL app names. Planner steps like
# "open the queue" or "ouvre le navigateur" would otherwise become
# `open -a "the queue"` (silent failure, 3 burned attempts, step blocked).
# An article-led or multi-clause "name" is not an app: fall through to
# vision grounding instead -- the safe default.
_APP_NOT_A_NAME_RE = re.compile(
    r"^(?:the|le|la|les|un|une|a|an|ce|cette|mon|ma|mes|my)\b|[,;]|\b(?:and|et|puis|then)\b",
    re.IGNORECASE,
)


def is_dangerous(instruction: str) -> bool:
    """Detects a destructive instruction that must never auto-execute.

    Args:
        instruction: Raw operator instruction (voice transcript or typed text).

    Returns:
        True if the instruction matches a destructive-action pattern.
    """
    return bool(_DANGEROUS_RE.search(instruction))


def is_correction(instruction: str) -> bool:
    """Detects whether a spoken phrase sounds like a live plan correction.

    Zero-LLM tier of the override path: when this fires mid-task, the caller
    (main.build_override_fn) spends one Gemini call to extract the actual
    (when, rule) pair; when it does not, the phrase is treated as a plain
    instruction and no extra model call is made.

    Args:
        instruction: Raw operator instruction (voice transcript or typed text).

    Returns:
        True if the phrasing carries a correction marker.
    """
    return bool(_CORRECTION_RE.search(instruction))


@dataclass(frozen=True)
class RoutedIntent:
    """Result of a successful zero-LLM fast-path match.

    Attributes:
        kind: "open_url" or "open_app".
        target: The URL or application name to open.
    """

    kind: str
    target: str


class IntentRouter:
    """Routes an instruction to a fast path, or signals a full-model decision.

    The contract is: `route()` NEVER returns a fast-path result for a
    dangerous instruction (it returns None so the caller falls through to
    the safety refusal in agent.py) and only recognizes narrow, explicit
    "open X" phrasing -- anything else also returns None so the slower
    Gemini-grounded loop handles it.
    """

    def route(self, instruction: str) -> RoutedIntent | None:
        """Attempts to resolve an instruction without calling the model.

        Args:
            instruction: Raw operator instruction.

        Returns:
            A `RoutedIntent` for a recognized "open app/URL" phrasing, or
            None if the instruction is dangerous or does not match a fast
            path (the caller must fall back to full Gemini-grounded reasoning).
        """
        if is_dangerous(instruction):
            return None

        stripped = instruction.strip()

        url_match = _OPEN_URL_RE.search(stripped)
        if url_match:
            url = url_match.group("url") or url_match.group("url2")
            return RoutedIntent(kind="open_url", target=url)

        app_match = _OPEN_APP_RE.match(stripped)
        if app_match:
            app = app_match.group("app").strip()
            if _APP_NOT_A_NAME_RE.search(app):
                return None
            return RoutedIntent(kind="open_app", target=app)

        return None
