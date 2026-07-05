"""Retry / circuit-breaker behavior of vision._generate, fully headless.

No SDK needed: `GeminiVision._ensure_client` returns any pre-injected
`_client` untouched, so a fake client exercises the real retry loop, the
transient/permanent split, and the breaker -- the only protections standing
between the demo and a dead network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vision import GeminiVision, VisionError


class _FakeModels:
    """Stands in for `client.models`, scripted to fail N times."""

    def __init__(self, outer: "_FakeClient") -> None:
        self._outer = outer

    def generate_content(self, model: str, contents: Any, **kwargs: Any) -> Any:
        self._outer.calls += 1
        self._outer.models_seen.append(model)
        if self._outer.calls <= self._outer.failures_before_success:
            raise self._outer.error
        return SimpleNamespace(text="ok")


class _FakeClient:
    """SDK-shaped fake: counts calls, raises a scripted error first."""

    def __init__(self, failures_before_success: int = 0, error: Exception | None = None) -> None:
        self.calls = 0
        self.models_seen: list[str] = []
        self.failures_before_success = failures_before_success
        self.error = error or RuntimeError("transient boom")
        self.models = _FakeModels(self)


class _CodedError(Exception):
    """Fake SDK error carrying an HTTP-like status code."""

    def __init__(self, code: int) -> None:
        super().__init__(f"http {code}")
        self.code = code


def _settings(**overrides: Any) -> SimpleNamespace:
    # gemini_api_key stays empty: these tests inject `_client` directly, so
    # `_ensure_client` (the only reader of the key) is never exercised.
    base = dict(
        gemini_api_key="",
        model_name="test-model",
        planner_model_name="",
        planner_thinking_level="",
        ground_samples=1,
        ground_confidence=0.75,
        max_context_screenshots=3,
        gemini_timeout_ms=1000,
        planner_timeout_ms=2000,
        gemini_max_attempts=3,
        gemini_backoff_s=0.0,
        gemini_breaker_failures=4,
        gemini_breaker_cooldown_s=60.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _vision_with(client: _FakeClient, **overrides: Any) -> GeminiVision:
    vision = GeminiVision(_settings(**overrides))
    vision._client = client
    return vision


def test_generate_retries_transient_then_raises_vision_error() -> None:
    client = _FakeClient(failures_before_success=99)
    vision = _vision_with(client)

    with pytest.raises(VisionError):
        vision._generate(["prompt"])

    assert client.calls == 3, "exactly max_attempts tries on a transient error"


def test_generate_succeeds_after_one_transient_failure() -> None:
    client = _FakeClient(failures_before_success=1)
    vision = _vision_with(client)

    response = vision._generate(["prompt"])

    assert response.text == "ok"
    assert client.calls == 2
    assert vision._consecutive_failures == 0, "a success resets the breaker counter"


def test_generate_fails_fast_on_non_transient_error() -> None:
    client = _FakeClient(failures_before_success=99, error=_CodedError(403))
    vision = _vision_with(client)

    with pytest.raises(VisionError):
        vision._generate(["prompt"])

    assert client.calls == 1, "a 403 (bad key) must not be retried"


def test_breaker_opens_after_consecutive_failed_calls_and_rejects_without_calling() -> None:
    client = _FakeClient(failures_before_success=99)
    vision = _vision_with(client, gemini_max_attempts=1, gemini_breaker_failures=2)

    for _ in range(2):
        with pytest.raises(VisionError):
            vision._generate(["prompt"])
    assert client.calls == 2

    with pytest.raises(VisionError, match="circuit breaker is open"):
        vision._generate(["prompt"])
    assert client.calls == 2, "an open breaker fails fast without touching the network"


def test_breaker_closes_after_cooldown_and_success_resets() -> None:
    client = _FakeClient(failures_before_success=2)
    vision = _vision_with(client, gemini_max_attempts=1, gemini_breaker_failures=2, gemini_breaker_cooldown_s=0.0)

    for _ in range(2):
        with pytest.raises(VisionError):
            vision._generate(["prompt"])
    assert vision._circuit_open_until > 0, "the breaker opened"

    # Cooldown of 0 means the breaker is immediately half-open again: the
    # next call goes through, succeeds, and fully resets the failure count.
    response = vision._generate(["prompt"])
    assert response.text == "ok"
    assert vision._consecutive_failures == 0


def test_two_model_split_routes_planner_calls_to_mastermind() -> None:
    """plan_steps/extract_override use PLANNER_MODEL; ground/verify keep MODEL_NAME."""
    from state import Step, TaskState

    client = _FakeClient()
    vision = _vision_with(client, planner_model_name="mastermind-model")
    task = TaskState(task_id="T", goal="g", steps=[Step(id="s1", desc="click A")])

    vision.plan_steps(task, "do something", screenshot=None)
    vision.extract_override(task, "non, en fait fais B")
    vision.ground(task, task.steps[0], screenshot="shot")
    vision.verify_step_done(task, task.steps[0], screenshot="shot")

    assert client.models_seen == [
        "mastermind-model",  # plan_steps -> reasoning tier
        "mastermind-model",  # extract_override -> reasoning tier
        "test-model",        # ground -> computer-use tier
        "test-model",        # verify_step_done -> computer-use tier
    ]


def test_planner_model_falls_back_to_grounding_model_when_unset() -> None:
    from state import TaskState

    client = _FakeClient()
    vision = _vision_with(client)  # planner_model_name=""
    vision.plan_steps(TaskState(task_id="T", goal="g"), "do something", screenshot=None)

    assert client.models_seen == ["test-model"]


class _ScriptedModels:
    """client.models fake replying with a scripted text per call."""

    def __init__(self, outer: "_ScriptedClient") -> None:
        self._outer = outer

    def generate_content(self, model: str, contents: Any, **kwargs: Any) -> Any:
        self._outer.calls += 1
        text = self._outer.texts[min(self._outer.calls - 1, len(self._outer.texts) - 1)]
        return SimpleNamespace(text=text)


class _ScriptedClient:
    def __init__(self, texts: list[str]) -> None:
        self.calls = 0
        self.texts = texts
        self.models = _ScriptedModels(self)


def _ground_task():
    from state import Step, TaskState

    return TaskState(task_id="T", goal="g", steps=[Step(id="s1", desc="click A")])


def test_confident_grounding_acts_on_a_single_call() -> None:
    client = _ScriptedClient(['{"action":"click","box":[10,10,20,20],"confidence":0.95}'])
    vision = _vision_with(client, ground_samples=3)

    action = vision.ground(_ground_task(), _ground_task().steps[0], screenshot="shot")

    assert client.calls == 1, "a confident answer never pays for extra samples"
    assert action.kind == "click" and action.confidence == 0.95


def test_uncertain_grounding_escalates_to_vote() -> None:
    client = _ScriptedClient(
        [
            '{"action":"click","box":[10,10,20,20],"confidence":0.3}',
            '{"action":"click","box":[12,8,22,18],"confidence":0.6}',
            '{"action":"click","box":[500,500,520,520],"confidence":0.5}',
        ]
    )
    vision = _vision_with(client, ground_samples=3)

    action = vision.ground(_ground_task(), _ground_task().steps[0], screenshot="shot")

    assert client.calls == 3, "low confidence buys the full sample budget"
    assert action.kind == "click"
    assert action.box[0] < 100, "the two near-agreeing clicks outvote the outlier"


def test_missing_confidence_does_not_escalate() -> None:
    client = _ScriptedClient(['{"action":"type","text":"bonjour"}'])
    vision = _vision_with(client, ground_samples=3)

    action = vision.ground(_ground_task(), _ground_task().steps[0], screenshot="shot")

    assert client.calls == 1, "a non-compliant reply must not silently triple latency"
    assert action.kind == "type" and action.confidence is None
