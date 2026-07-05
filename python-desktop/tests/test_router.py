"""Tests for the IntentRouter (router.py)."""

from __future__ import annotations

from router import IntentRouter, is_correction, is_dangerous


def test_is_dangerous_detects_common_destructive_phrasings() -> None:
    assert is_dangerous("supprime tout") is True
    assert is_dangerous("Please rm -rf / now") is True
    assert is_dangerous("delete everything on the desktop") is True
    assert is_dangerous("format disk C") is True
    assert is_dangerous("vide la corbeille") is True
    assert is_dangerous("sudo reboot") is True


def test_is_dangerous_false_for_benign_instructions() -> None:
    assert is_dangerous("open Linear") is False
    assert is_dangerous("triage the network bugs") is False
    assert is_dangerous("supprime le fichier test.txt") is False  # narrow, not "tout"


def test_route_returns_none_for_dangerous_instruction() -> None:
    router = IntentRouter()
    assert router.route("supprime tout") is None
    assert router.route("sudo rm -rf /") is None


def test_route_open_url_fast_path() -> None:
    router = IntentRouter()
    result = router.route("ouvre https://linear.app/team/queue")
    assert result is not None
    assert result.kind == "open_url"
    assert result.target == "https://linear.app/team/queue"


def test_route_open_app_fast_path() -> None:
    router = IntentRouter()
    result = router.route("ouvre Linear")
    assert result is not None
    assert result.kind == "open_app"
    assert result.target == "Linear"

    result_en = router.route("open Reminders")
    assert result_en is not None
    assert result_en.kind == "open_app"
    assert result_en.target == "Reminders"

    multi_word = router.route("open Visual Studio Code")
    assert multi_word is not None and multi_word.target == "Visual Studio Code"


def test_route_rejects_generic_open_phrases() -> None:
    """Planner steps like "open the queue" must fall through to vision, not
    become a doomed `open -a "the queue"` call."""
    router = IntentRouter()
    assert router.route("open the queue") is None
    assert router.route("open the settings menu") is None
    assert router.route("ouvre le navigateur") is None
    assert router.route("open System Settings and disable notifications") is None


def test_route_returns_none_for_unmatched_instruction() -> None:
    router = IntentRouter()
    result = router.route("assign this ticket to OPS and add a comment")
    assert result is None


def test_is_correction_detects_french_and_english_markers() -> None:
    assert is_correction("non, les bugs reseau vont a INFRA") is True
    assert is_correction("en fait mets-les dans INFRA") is True
    assert is_correction("actually, assign them to INFRA") is True
    assert is_correction("mets-les plutot dans INFRA") is True
    assert is_correction("from now on network bugs go to INFRA") is True
    assert is_correction("pas a OPS mais a INFRA") is True
    assert is_correction("not OPS but INFRA") is True
    assert is_correction("je me suis trompe, c'est INFRA") is True


def test_is_correction_false_for_plain_instructions() -> None:
    assert is_correction("ouvre Linear") is False
    assert is_correction("triage the network bugs") is False
    assert is_correction("process the queue") is False
    assert is_correction("note the ticket number") is False


def test_route_type_step_with_quoted_text_is_deterministic() -> None:
    """The exact bug from the live run: "Type 'simple recipes'." must become
    a type action, never a vision call that re-clicks the field."""
    router = IntentRouter()
    routed = router.route("Type 'simple recipes'.")
    assert routed is not None
    assert routed.kind == "type" and routed.target == "simple recipes"


def test_route_type_step_without_quotes_falls_through_to_vision() -> None:
    router = IntentRouter()
    assert router.route("type the message to the team") is None


def test_route_press_key_combo() -> None:
    router = IntentRouter()
    routed = router.route("Press cmd+l to focus the address bar.")
    assert routed is not None
    assert routed.kind == "hotkey" and routed.keys == ("command", "l")


def test_route_press_enter_alone() -> None:
    router = IntentRouter()
    routed = router.route("Press Enter.")
    assert routed is not None
    assert routed.kind == "hotkey" and routed.keys == ("enter",)


def test_route_press_ui_button_is_not_a_hotkey() -> None:
    """'Press the New Note button' describes a click target, not a chord."""
    router = IntentRouter()
    assert router.route("Press the New Note button in the toolbar.") is None


def test_route_scroll_direction() -> None:
    router = IntentRouter()
    down = router.route("Scroll down to see more results")
    assert down is not None and down.kind == "scroll" and down.target == "-5"
    up = router.route("scroll up")
    assert up is not None and up.target == "5"
