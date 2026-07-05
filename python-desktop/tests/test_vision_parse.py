"""Pure parsing of Gemini responses -- no SDK, no network, fully headless.

Covers the action/verification parsers that turn a raw model reply into a
`GroundedAction` or a done/not-done verdict. The network wrapper around these
(`_generate`, timeout/retry/breaker) is live-only and cannot be exercised
without the SDK; the parsing is where the logic lives and is tested here.
"""

from __future__ import annotations

from types import SimpleNamespace

from vision import (
    GeminiVision,
    GroundedAction,
    _extract_json_object,
    _grounded_from_action,
    _is_transient_error,
    _parse_override,
    _parse_yes_no,
)


def _vision() -> GeminiVision:
    """Builds a client-less GeminiVision: parsing never touches the network."""
    return GeminiVision(SimpleNamespace())


class _CodedError(Exception):
    """Fake SDK error carrying an HTTP-like status code."""

    def __init__(self, code: int) -> None:
        super().__init__(f"http {code}")
        self.code = code


def test_extract_json_object() -> None:
    assert _extract_json_object('blah {"action":"noop"} tail') == {"action": "noop"}
    assert _extract_json_object("no json here") is None
    assert _extract_json_object("{not valid json}") is None
    assert _extract_json_object("[1,2,3]") is None  # array is not an object


def test_extract_json_object_two_objects_returns_first() -> None:
    # A chatty model reply with TWO objects must still parse (a greedy
    # `{.*}` span would be invalid JSON and lose both).
    text = '{"action":"type","text":"hi"} and also {"note":"ignore me"}'
    assert _extract_json_object(text) == {"action": "type", "text": "hi"}


def test_extract_json_object_recovers_after_invalid_brace() -> None:
    text = "{oops} then the real one {\"action\":\"noop\"}"
    assert _extract_json_object(text) == {"action": "noop"}


def test_grounded_from_action_click() -> None:
    g = _grounded_from_action({"action": "click", "box": [10, 20, 30, 40]}, "reason")
    assert g is not None and g.kind == "click"
    assert g.box == (10.0, 20.0, 30.0, 40.0)
    # malformed box -> None (caller falls back to the bare-box regex)
    assert _grounded_from_action({"action": "click", "box": [1, 2]}, "r") is None


def test_grounded_from_action_non_numeric_box_returns_none_not_crash() -> None:
    # Model hallucinations must degrade, never raise into the agent loop.
    assert _grounded_from_action({"action": "click", "box": ["a", "b", "c", "d"]}, "r") is None
    assert _grounded_from_action({"action": "click", "box": [None, 10, 20, 30]}, "r") is None


def test_grounded_from_action_scroll() -> None:
    g = _grounded_from_action({"action": "scroll", "amount": -5}, "r")
    assert g is not None and g.kind == "scroll" and g.amount == -5
    assert _grounded_from_action({"action": "scroll", "amount": "down"}, "r") is None
    assert _grounded_from_action({"action": "scroll"}, "r") is None


def test_grounded_from_action_type_and_hotkey() -> None:
    t = _grounded_from_action({"action": "type", "text": "hello"}, "r")
    assert t is not None and t.kind == "type" and t.text == "hello"
    assert _grounded_from_action({"action": "type", "text": ""}, "r") is None

    h = _grounded_from_action({"action": "hotkey", "keys": ["command", "v"]}, "r")
    assert h is not None and h.kind == "hotkey" and h.keys == ["command", "v"]
    assert _grounded_from_action({"action": "hotkey", "keys": []}, "r") is None


def test_grounded_from_action_noop_and_unknown() -> None:
    n = _grounded_from_action({"action": "noop"}, "r")
    assert isinstance(n, GroundedAction) and n.kind == "noop"
    assert _grounded_from_action({"action": "teleport"}, "r") is None


def test_parse_override_confirmed_correction() -> None:
    text = 'noise {"correction": true, "when": "network bug", "rule": "assign to INFRA"} tail'
    assert _parse_override(text) == ("network bug", "assign to INFRA")


def test_parse_override_not_a_correction() -> None:
    assert _parse_override('{"correction": false}') is None


def test_parse_override_malformed_payloads() -> None:
    assert _parse_override(None) is None
    assert _parse_override("") is None
    assert _parse_override("no json at all") is None
    assert _parse_override('{"correction": true, "when": "", "rule": "x"}') is None
    assert _parse_override('{"correction": true, "when": "y"}') is None


def test_is_transient_error_classification() -> None:
    assert _is_transient_error(_CodedError(500)) is True, "5xx retries"
    assert _is_transient_error(_CodedError(503)) is True
    assert _is_transient_error(_CodedError(429)) is True, "rate limit retries"
    assert _is_transient_error(_CodedError(403)) is False, "bad key must fail fast"
    assert _is_transient_error(_CodedError(400)) is False
    assert _is_transient_error(TimeoutError("t")) is True, "no code -> assume transient"
    assert _is_transient_error(ConnectionError("c")) is True


def test_parse_yes_no() -> None:
    assert _parse_yes_no("YES") is True
    assert _parse_yes_no("yes, it is done") is True
    assert _parse_yes_no("Oui") is True
    assert _parse_yes_no("NO") is False
    assert _parse_yes_no("not yet") is False
    assert _parse_yes_no("") is False
    assert _parse_yes_no(None) is False


def test_parse_steps_happy_path_and_cap() -> None:
    vision = _vision()
    reply = SimpleNamespace(text='Here you go:\n["open Slack", "  ", "click #general", "type hi"]')
    assert vision._parse_steps(reply) == ["open Slack", "click #general", "type hi"]

    too_many = SimpleNamespace(text=str([f"step {i}" for i in range(12)]).replace("'", '"'))
    assert len(vision._parse_steps(too_many)) == 6, "the planner cap bounds the parsed list"


def test_parse_steps_malformed_replies_yield_empty_plan() -> None:
    # The very first Gemini call of a live run parses here: a malformed
    # reply must yield [] (loop stalls, operator repeats), never crash.
    vision = _vision()
    assert vision._parse_steps(SimpleNamespace(text=None)) == []
    assert vision._parse_steps(SimpleNamespace(text="no array here")) == []
    assert vision._parse_steps(SimpleNamespace(text="[not json]")) == []
    assert vision._parse_steps(SimpleNamespace(text='{"a": 1}')) == []


def test_parse_response_fallbacks() -> None:
    vision = _vision()
    # Bare-box fallback: no JSON object, but a [ymin,xmin,ymax,xmax] in prose.
    bare = vision._parse_response(SimpleNamespace(text="I would click [100, 300, 200, 500] here"))
    assert bare.kind == "click" and bare.box == (100.0, 300.0, 200.0, 500.0)
    # Nothing parseable at all -> explicit noop, never an exception.
    noop = vision._parse_response(SimpleNamespace(text="I am not sure what to do"))
    assert noop.kind == "noop"


def test_vote_grounded_majority_kind_wins() -> None:
    from vision import GroundedAction, _vote_grounded

    actions = [
        GroundedAction(kind="click", box=(100, 100, 120, 120)),
        GroundedAction(kind="click", box=(102, 98, 122, 118)),
        GroundedAction(kind="noop"),
    ]
    voted = _vote_grounded(actions)
    assert voted.kind == "click", "2 clicks outvote 1 noop"


def test_vote_grounded_clusters_clicks_never_averages_distinct_targets() -> None:
    """Two samples on button A, one on faraway button B: the consensus click
    must land ON A, not at the meaningless midpoint between A and B."""
    from vision import GroundedAction, _vote_grounded

    actions = [
        GroundedAction(kind="click", box=(100, 100, 120, 120)),   # A
        GroundedAction(kind="click", box=(104, 96, 124, 116)),    # A (jitter)
        GroundedAction(kind="click", box=(800, 800, 820, 820)),   # B (outlier)
    ]
    voted = _vote_grounded(actions)
    center_y = (voted.box[0] + voted.box[2]) / 2
    center_x = (voted.box[1] + voted.box[3]) / 2
    assert center_y < 200 and center_x < 200, "consensus stays on target A"


def test_vote_grounded_type_picks_most_common_text() -> None:
    from vision import GroundedAction, _vote_grounded

    actions = [
        GroundedAction(kind="type", text="bonjour"),
        GroundedAction(kind="type", text="bonjour"),
        GroundedAction(kind="type", text="bonjour!"),
    ]
    assert _vote_grounded(actions).text == "bonjour"


def test_vote_grounded_single_sample_passthrough() -> None:
    from vision import GroundedAction, _vote_grounded

    only = GroundedAction(kind="hotkey", keys=["command", "v"])
    assert _vote_grounded([only]) is only
