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
        if self._outer.calls <= self._outer.failures_before_success:
            raise self._outer.error
        return SimpleNamespace(text="ok")


class _FakeClient:
    """SDK-shaped fake: counts calls, raises a scripted error first."""

    def __init__(self, failures_before_success: int = 0, error: Exception | None = None) -> None:
        self.calls = 0
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
        max_context_screenshots=3,
        gemini_timeout_ms=1000,
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
