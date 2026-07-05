"""Headless tests for the Interactions Computer Use path (Route 1).

No SDK and no network: `InteractionsComputerUse` takes an injected fake
`client` exposing `interactions.create`, so the real session lifecycle,
function-call mapping, safety refusal and circuit breaker all run against
scripted responses. The `build_ground_fn` dispatch is exercised with stub
collaborators to prove CU_MODE routes to the right brain in both modes while
the router fast-paths stay upstream.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import main
from agent import Observation
from interactions_cu import InteractionsComputerUse
from state import Step, TaskState


# --- fakes -----------------------------------------------------------------

class _FakeInteractions:
    """Stands in for `client.interactions`, replaying scripted responses."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self._responses.pop(0) if self._responses else self._responses_default()
        if isinstance(item, Exception):
            raise item
        return item

    @staticmethod
    def _responses_default() -> Any:
        return {"id": "ix_default", "steps": [], "output_text": "done"}


class _FakeClient:
    """SDK-shaped fake exposing only `.interactions.create`."""

    def __init__(self, responses: list[Any]) -> None:
        self.interactions = _FakeInteractions(responses)


class _CodedError(Exception):
    """Fake SDK error carrying an HTTP-like status code."""

    def __init__(self, code: int, message: str = "") -> None:
        super().__init__(message or f"http {code}")
        self.code = code


def _settings(**overrides: Any) -> SimpleNamespace:
    # The fake client is injected, so gemini_api_key (read only by
    # _ensure_client) is irrelevant. Breaker knobs are shared with vision.
    base = dict(
        gemini_api_key="",
        model_name="cu-model",
        cu_model_name="",
        cu_environment="desktop",
        cu_norm_max=999,
        cu_prompt_injection_detection=True,
        cu_scroll_pixels_per_click=100,
        cu_timeout_ms=60_000,
        cu_mode="interactions",
        gemini_max_attempts=3,
        gemini_backoff_s=0.0,
        gemini_breaker_failures=4,
        gemini_breaker_cooldown_s=60.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _resp(id_: str, steps: list[Any] | None = None, output_text: str | None = None) -> dict[str, Any]:
    return {"id": id_, "steps": steps or [], "output_text": output_text}


def _fc(name: str, arguments: dict[str, Any], id_: str = "c1") -> dict[str, Any]:
    return {"type": "function_call", "name": name, "id": id_, "arguments": arguments}


def _task_step() -> tuple[TaskState, Step]:
    step = Step(id="s1", desc="click the blue button")
    return TaskState(task_id="T", goal="do the thing", steps=[step]), step


def _cu(responses: list[Any], **overrides: Any) -> InteractionsComputerUse:
    return InteractionsComputerUse(_settings(**overrides), client=_FakeClient(responses))


# --- function-call mapping (payload parsing) --------------------------------

def test_click_maps_to_point() -> None:
    cu = _cu([_resp("ix1", [_fc("click", {"x": 500, "y": 400, "intent": "hit login"})])])
    task, step = _task_step()

    action = cu.ground(task, step, b"pngbytes")

    assert action is not None and action.kind == "click"
    assert action.point == (500.0, 400.0)
    assert action.reasoning == "hit login"


def test_type_maps_to_text() -> None:
    cu = _cu([_resp("ix1", [_fc("type", {"text": "bonjour", "press_enter": True})])])
    task, step = _task_step()

    action = cu.ground(task, step, b"png")

    assert action is not None and action.kind == "type" and action.text == "bonjour"


def test_hotkey_normalizes_keys() -> None:
    cu = _cu([_resp("ix1", [_fc("hotkey", {"keys": ["Cmd", "Return"]})])])
    task, step = _task_step()

    action = cu.ground(task, step, b"png")

    assert action is not None and action.kind == "hotkey"
    assert action.keys == ["command", "enter"], "cmd->command, return->enter aliasing"


def test_press_key_maps_to_single_hotkey() -> None:
    cu = _cu([_resp("ix1", [_fc("press_key", {"key": "Escape"})])])
    task, step = _task_step()

    action = cu.ground(task, step, b"png")

    assert action is not None and action.kind == "hotkey" and action.keys == ["esc"]


def test_scroll_down_is_negative_up_is_positive() -> None:
    task, step = _task_step()
    down = _cu([_resp("ix1", [_fc("scroll", {"direction": "down", "magnitude_in_pixels": 300})])])
    up = _cu([_resp("ix1", [_fc("scroll", {"direction": "up", "magnitude_in_pixels": 300})])])

    down_action = down.ground(task, step, b"png")
    up_action = up.ground(task, step, b"png")

    assert down_action.kind == "scroll" and down_action.amount == -3  # 300/100
    assert up_action.amount == 3


def test_horizontal_scroll_is_unsupported() -> None:
    cu = _cu([_resp("ix1", [_fc("scroll", {"direction": "left", "magnitude_in_pixels": 200})])])
    task, step = _task_step()

    assert cu.ground(task, step, b"png") is None, "no vertical pyautogui equivalent -> stall"


def test_unsupported_function_stalls() -> None:
    cu = _cu([_resp("ix1", [_fc("double_click", {"x": 1, "y": 1})])])
    task, step = _task_step()

    assert cu.ground(task, step, b"png") is None


def test_no_function_call_means_step_done() -> None:
    cu = _cu([_resp("ix1", steps=[], output_text="The button is already highlighted.")])
    task, step = _task_step()

    action = cu.ground(task, step, b"png")

    assert action is not None and action.kind == "done"
    assert action.reasoning == "The button is already highlighted."


def test_click_with_non_numeric_coords_stalls() -> None:
    cu = _cu([_resp("ix1", [_fc("click", {"x": "left", "y": None})])])
    task, step = _task_step()

    assert cu.ground(task, step, b"png") is None


# --- safety refusal ---------------------------------------------------------

def test_safety_confirmation_is_refused() -> None:
    args = {"x": 60, "y": 100, "safety_decision": {"decision": "require_confirmation", "explanation": "risky"}}
    cu = _cu([_resp("ix1", [_fc("click", args)])])
    task, step = _task_step()

    assert cu.ground(task, step, b"png") is None, "a confirmation-required action must be refused"
    assert cu._pending_call is None, "a refused action leaves nothing pending"


# --- session lifecycle ------------------------------------------------------

def test_first_call_opens_session_with_a_user_turn() -> None:
    cu = _cu([_resp("ix_first", [_fc("click", {"x": 10, "y": 10})])])
    task, step = _task_step()

    cu.ground(task, step, b"png")

    kwargs = cu._client.interactions.calls[0]
    assert "previous_interaction_id" not in kwargs, "the first call opens a fresh session"
    assert kwargs["input"][0]["role"] == "user", "initial input is a user Turn (not a bare content list)"
    assert kwargs["store"] is True, "store=True so previous_interaction_id resolves later"
    assert kwargs["tools"][0] == {
        "type": "computer_use",
        "environment": "desktop",
        "enable_prompt_injection_detection": True,
    }
    assert cu.session_id == "ix_first"


def test_second_call_continues_with_a_function_result() -> None:
    cu = _cu(
        [
            _resp("ix1", [_fc("click", {"x": 10, "y": 10}, id_="call_A")]),
            _resp("ix2", [_fc("type", {"text": "x"})]),
        ]
    )
    task, step = _task_step()

    cu.ground(task, step, b"png1")  # acts on call_A
    cu.ground(task, step, b"png2")  # should report call_A's result

    second = cu._client.interactions.calls[1]
    assert second["previous_interaction_id"] == "ix1", "the session id continues"
    result_step = second["input"][0]
    assert result_step["type"] == "function_result"
    assert result_step["call_id"] == "call_A", "the executed call's id is echoed back"
    assert any(part["type"] == "image" for part in result_step["result"]), "the new screen is attached"


def test_restore_session_reanchors_with_previous_id() -> None:
    cu = _cu([_resp("ix_new", [_fc("click", {"x": 1, "y": 1})])])
    cu.restore_session("ix_persisted")
    task, step = _task_step()

    assert cu.session_id == "ix_persisted"
    cu.ground(task, step, b"png")

    kwargs = cu._client.interactions.calls[0]
    assert kwargs["previous_interaction_id"] == "ix_persisted", "resume reattaches to the live session"
    assert kwargs["input"][0]["role"] == "user", "no pending call -> re-anchor with a fresh turn"


def test_step_change_resets_to_a_fresh_turn() -> None:
    cu = _cu(
        [
            _resp("ix1", [_fc("click", {"x": 1, "y": 1}, id_="cA")]),
            _resp("ix2", [_fc("click", {"x": 2, "y": 2})]),
        ]
    )
    task = TaskState(task_id="T", goal="g", steps=[Step(id="s1", desc="a"), Step(id="s2", desc="b")])

    cu.ground(task, task.steps[0], b"png1")  # step s1, sets pending
    cu.ground(task, task.steps[1], b"png2")  # step s2 -> must NOT reuse s1's pending call

    second = cu._client.interactions.calls[1]
    assert second["input"][0].get("role") == "user", "a new step re-anchors, not a stale function_result"


def test_encode_failure_stalls_without_calling_the_api() -> None:
    class _Unencodable:
        def save(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("cannot encode")

    cu = _cu([_resp("ix1", [_fc("click", {"x": 1, "y": 1})])])
    task, step = _task_step()

    assert cu.ground(task, step, _Unencodable()) is None
    assert cu._client.interactions.calls == [], "a broken screenshot never reaches the API"


# --- circuit breaker degradation -------------------------------------------

def test_transient_failures_degrade_to_none_and_open_the_breaker() -> None:
    boom = RuntimeError("transient boom")
    cu = _cu([boom, boom, boom], gemini_max_attempts=1, gemini_breaker_failures=2)
    task, step = _task_step()

    assert cu.ground(task, step, b"png") is None, "a failed CU call degrades to a stall, never raises"
    assert cu.ground(task, step, b"png") is None
    # Breaker is now open: a third ground must not touch the client.
    assert cu.ground(task, step, b"png") is None
    assert len(cu._client.interactions.calls) == 2, "an open breaker fails fast without calling the API"


def test_non_transient_error_is_not_retried() -> None:
    cu = _cu([_CodedError(403)], gemini_max_attempts=3)
    task, step = _task_step()

    assert cu.ground(task, step, b"png") is None
    assert len(cu._client.interactions.calls) == 1, "a 403 (bad key) is not retried"


def test_expired_session_renews_cleanly() -> None:
    cu = _cu(
        [
            _resp("ix1", [_fc("click", {"x": 1, "y": 1}, id_="cA")]),  # opens session
            _CodedError(404, "interaction not found"),                  # continuation -> expired
            _resp("ix2", [_fc("type", {"text": "renewed"})]),           # fresh session
        ],
        gemini_max_attempts=1,
    )
    task, step = _task_step()

    cu.ground(task, step, b"png1")           # session ix1, pending cA
    action = cu.ground(task, step, b"png2")  # 404 -> renew -> fresh turn

    assert action is not None and action.kind == "type" and action.text == "renewed"
    assert cu.session_id == "ix2", "the session was renewed to the new interaction id"
    renewed_call = cu._client.interactions.calls[-1]
    assert "previous_interaction_id" not in renewed_call, "renewal opens a brand-new session"


# --- build_ground_fn dispatch ----------------------------------------------

class _StubCu:
    """Minimal InteractionsComputerUse stand-in for the dispatch tests."""

    def __init__(self, action: Any, session_id: str = "ix_stub") -> None:
        self._action = action
        self._session_id = session_id
        self.calls = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    def ground(self, task: TaskState, step: Step, screenshot: Any) -> Any:
        self.calls += 1
        return self._action


def _fake_mac() -> Any:
    return SimpleNamespace(screen_size_logical=lambda: (1000, 1000))


def _obs() -> Observation:
    return Observation(instruction=None, screenshot=b"png")


def test_dispatch_interactions_denormalizes_point_and_persists_session() -> None:
    from interactions_cu import CuAction
    from router import IntentRouter

    stub = _StubCu(CuAction(kind="click", point=(999.0, 0.0), reasoning="hit it"))
    task = TaskState(task_id="T", goal="g", steps=[Step(id="s1", desc="click the blue button")])
    ground = main.build_ground_fn(
        vision=None, mac=_fake_mac(), router=IntentRouter(),
        settings=_settings(cu_mode="interactions", cu_norm_max=999), cu=stub,
    )

    plan = ground(task, task.steps[0], _obs())

    assert plan is not None and plan.kind == "click"
    assert plan.target == (998, 1), "x=999/999*1000 clamped to 998; y=0 clamped to 1"
    assert task.interactions_session_id == "ix_stub", "the live session id is copied onto the task"


def test_dispatch_interactions_done_maps_to_done_plan() -> None:
    from interactions_cu import CuAction
    from router import IntentRouter

    stub = _StubCu(CuAction(kind="done"))
    task = TaskState(task_id="T", goal="g", steps=[Step(id="s1", desc="verify the page loaded")])
    ground = main.build_ground_fn(
        vision=None, mac=_fake_mac(), router=IntentRouter(), settings=_settings(), cu=stub,
    )

    plan = ground(task, task.steps[0], _obs())

    assert plan is not None and plan.kind == "done"


def test_router_fast_path_fires_in_interactions_mode_without_touching_cu() -> None:
    from router import IntentRouter

    stub = _StubCu(action=None)
    task = TaskState(task_id="T", goal="g", steps=[Step(id="s1", desc="open Safari")])
    ground = main.build_ground_fn(
        vision=None, mac=_fake_mac(), router=IntentRouter(), settings=_settings(), cu=stub,
    )

    plan = ground(task, task.steps[0], _obs())

    assert plan is not None and plan.kind == "open_app" and plan.text == "Safari"
    assert plan.completes_step is True
    assert stub.calls == 0, "a fast-path step never reaches the CU brain"


def test_grounding_mode_uses_vision_not_cu() -> None:
    from router import IntentRouter
    from vision import GroundedAction

    fake_vision = SimpleNamespace(
        ground=lambda task, step, shot: GroundedAction(kind="click", box=(0, 0, 1000, 1000), reasoning="r")
    )
    stub = _StubCu(action=None)
    task = TaskState(task_id="T", goal="g", steps=[Step(id="s1", desc="click the blue button")])
    ground = main.build_ground_fn(
        vision=fake_vision, mac=_fake_mac(), router=IntentRouter(),
        settings=_settings(cu_mode="grounding"), cu=stub,
    )

    plan = ground(task, task.steps[0], _obs())

    assert plan is not None and plan.kind == "click" and plan.target == (500, 500)
    assert stub.calls == 0, "grounding mode must not call the CU brain"


# --- state / memory persistence --------------------------------------------

def test_interactions_session_id_persists_and_reloads(tmp_path: Any) -> None:
    from memory import MemoryStore

    memory = MemoryStore(tmp_path / "t.db")
    task = TaskState(task_id="X", goal="g", interactions_session_id="ix_persist")
    memory.save_task_state(task)

    reloaded = memory.resume_task_state("X")

    assert reloaded is not None
    assert reloaded.interactions_session_id == "ix_persist", "the session id survives a reload/resume"
    assert reloaded.render()["interactions_session_id"] == "ix_persist", "render() exposes it for the HUD"
    memory.close()


def test_new_task_has_no_session_id() -> None:
    task = TaskState(task_id="X", goal="g")
    assert task.interactions_session_id is None
    assert task.render()["interactions_session_id"] is None
