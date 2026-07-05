#!/usr/bin/env python3
"""Ground ONE step against the LIVE screen in both routes, side by side.

This is the demo-day decision tool for CU_MODE. It captures the current
screen once, then grounds the same step twice:

  * Route 2 (grounding): `GeminiVision.ground` -> a normalized box.
  * Route 1 (interactions): `InteractionsComputerUse.ground` -> a 0-999 point
    via the official Computer Use `interactions.create` API.

For each it prints the wall-clock latency, the raw decision, and the pixel
target it would click, so you can pick the route that grounds the real target
reliably. Needs a real GEMINI_API_KEY and the macOS Screen Recording
permission (it takes a real screenshot); it performs NO clicks -- it only
grounds and reports.

Run: `.venv/bin/python scripts/probe_cu.py "click the Safari address bar"`
(the step description is optional; a default is used when omitted).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import is_placeholder_key, load_settings  # noqa: E402
from interactions_cu import InteractionsComputerUse  # noqa: E402
from mac_control import MacController, denormalize_box, denormalize_point  # noqa: E402
from state import Step, TaskState  # noqa: E402
from vision import GeminiVision, VisionError  # noqa: E402

_DEFAULT_STEP = "click the search field"


def _make_task(step_desc: str) -> tuple[TaskState, Step]:
    """Builds a one-step task to ground against the live screen."""
    step = Step(id="s1", desc=step_desc)
    task = TaskState(task_id="PROBE", goal=f"probe: {step_desc}", steps=[step])
    return task, step


def probe_grounding(task: TaskState, step: Step, screenshot: Any, mac: MacController) -> None:
    """Grounds via Route 2 (vision box) and prints latency + pixel target."""
    settings = load_settings()
    vision = GeminiVision(settings)
    start = time.monotonic()
    try:
        action = vision.ground(task, step, screenshot)
    except VisionError as error:
        print(f"[grounding] FAILED after {time.monotonic() - start:.2f}s: {error}")
        return
    elapsed = time.monotonic() - start
    target = None
    if action.kind == "click" and action.box is not None:
        width, height = mac.screen_size_logical()
        target = denormalize_box(action.box, width, height)
    print(f"[grounding ] {elapsed:5.2f}s  kind={action.kind}  box={action.box}  -> click {target}")


def probe_interactions(task: TaskState, step: Step, screenshot: Any, mac: MacController) -> None:
    """Grounds via Route 1 (Computer Use point) and prints latency + pixel target."""
    settings = load_settings()
    cu = InteractionsComputerUse(settings)
    start = time.monotonic()
    action = cu.ground(task, step, screenshot)  # never raises; None on any failure
    elapsed = time.monotonic() - start
    if action is None:
        print(f"[interact.  ] {elapsed:5.2f}s  -> None (stall: unsupported/safety/failure)")
        return
    target = None
    if action.kind == "click" and action.point is not None:
        width, height = mac.screen_size_logical()
        target = denormalize_point(action.point[0], action.point[1], width, height, settings.cu_norm_max)
    print(
        f"[interact.  ] {elapsed:5.2f}s  kind={action.kind}  point={action.point}  "
        f"keys={action.keys}  amount={action.amount}  -> click {target}  (session {cu.session_id})"
    )


def main() -> int:
    """Captures one live screen and grounds the step in both routes."""
    settings = load_settings()
    if is_placeholder_key(settings.gemini_api_key):
        print("GEMINI_API_KEY is missing or a placeholder -- set a real key in .env first.")
        return 1

    step_desc = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_STEP
    task, step = _make_task(step_desc)

    mac = MacController(pyautogui_pause_s=settings.pyautogui_pause_s)
    try:
        screenshot = mac.capture_screenshot_logical()
    except Exception as error:  # noqa: BLE001 - a probe on a headless box should say so, not crash
        print(f"could not capture the screen (grant Screen Recording?): {error}")
        return 1

    print(f"Grounding step: {step_desc!r}\n")
    probe_grounding(task, step, screenshot, mac)
    probe_interactions(task, step, screenshot, mac)
    print("\nPick the route that lands the click on the real target reliably (set CU_MODE).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
