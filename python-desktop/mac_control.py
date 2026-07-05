"""ACTIONNEUR: turns a normalized Gemini target into a real Mac action.

Pure coordinate math (denormalize/scale) lives at module level with zero
dependencies so it is testable without any GUI library installed. Every
GUI/native library (`pyautogui`, `mss`, `PIL`) is imported lazily inside the
methods that need it, so `import mac_control` always succeeds even on a
machine (or CI box) where those packages are absent.

Handles the two documented traps: the Retina physical/logical pixel
mismatch (mss captures physical pixels, pyautogui clicks logical points),
and long-text typing reliability (paste via clipboard, save/restore the
previous clipboard contents, fallback to `typewrite`).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from agent import ActionPlan

DEFAULT_NORM_MAX = 1000

# Local subprocess guards (pbcopy/pbpaste/open are non-network, but a hung
# child would still freeze the loop). Named here once, reused everywhere.
_CLIPBOARD_TIMEOUT_S = 2.0
_OPEN_TIMEOUT_S = 10.0
_TYPEWRITE_INTERVAL_S = 0.04


class ActuatorError(Exception):
    """Raised when a real Mac action cannot be executed."""


def denormalize_point(
    x_norm: float, y_norm: float, width: int, height: int, norm_max: int = DEFAULT_NORM_MAX
) -> tuple[int, int]:
    """Converts a Gemini-normalized (0..norm_max) point into pixel coordinates.

    Args:
        x_norm: Normalized x in [0, norm_max].
        y_norm: Normalized y in [0, norm_max].
        width: Target image/screen width in pixels.
        height: Target image/screen height in pixels.
        norm_max: The normalization ceiling Gemini used (1000, or 999).

    Returns:
        (pixel_x, pixel_y) rounded to the nearest integer, clamped inside
        [1, width-2] x [1, height-2]: a normalized 1000 would land one pixel
        OFF screen, and the four exact screen corners trigger pyautogui's
        FAILSAFE (reserved as the operator's emergency abort gesture).
    """
    pixel_x = int(round(x_norm / norm_max * width))
    pixel_y = int(round(y_norm / norm_max * height))
    pixel_x = min(max(pixel_x, 1), max(width - 2, 1))
    pixel_y = min(max(pixel_y, 1), max(height - 2, 1))
    return pixel_x, pixel_y


def denormalize_box(
    box: tuple[float, float, float, float], width: int, height: int, norm_max: int = DEFAULT_NORM_MAX
) -> tuple[int, int]:
    """Converts a Gemini [ymin, xmin, ymax, xmax] normalized box to its pixel center.

    Args:
        box: (ymin, xmin, ymax, xmax), each normalized in [0, norm_max].
        width: Target image/screen width in pixels.
        height: Target image/screen height in pixels.
        norm_max: The normalization ceiling Gemini used.

    Returns:
        (pixel_x, pixel_y) of the box's center.
    """
    ymin, xmin, ymax, xmax = box
    center_x_norm = (xmin + xmax) / 2
    center_y_norm = (ymin + ymax) / 2
    return denormalize_point(center_x_norm, center_y_norm, width, height, norm_max)


def compute_retina_scale(physical_width: int, logical_width: int) -> float:
    """Computes the mss-physical to pyautogui-logical scale factor.

    On a Retina display, mss captures at physical resolution (e.g. 2880px)
    while pyautogui clicks in logical points (e.g. 1440pt): SCALE = 2. If
    the screenshot was already resized to the logical resolution before
    grounding, this factor is 1.0 and no further division is needed.

    Args:
        physical_width: Width in pixels of the raw mss capture.
        logical_width: Width in points reported by `pyautogui.size()`.

    Returns:
        The scale factor to divide raw-capture pixel coordinates by.

    Raises:
        ValueError: If `logical_width` is not positive.
    """
    if logical_width <= 0:
        raise ValueError("logical_width must be positive")
    return physical_width / logical_width


def apply_scale(pixel: tuple[int, int], scale: float) -> tuple[int, int]:
    """Divides a physical-capture pixel coordinate by the Retina scale factor.

    Args:
        pixel: (x, y) computed against the physical-resolution screenshot.
        scale: Value from `compute_retina_scale`.

    Returns:
        (x, y) adjusted to logical (pyautogui-clickable) coordinates.
    """
    x, y = pixel
    return int(round(x / scale)), int(round(y / scale))


@dataclass
class ScreenGeometry:
    """Snapshot of logical vs physical screen dimensions for one capture."""

    logical_width: int
    logical_height: int
    physical_width: int
    physical_height: int

    @property
    def scale(self) -> float:
        """Retina scale factor (physical / logical), 1.0 on non-Retina."""
        return compute_retina_scale(self.physical_width, self.logical_width)


class MacController:
    """Executes ActionPlans on the real macOS desktop via pyautogui/mss.

    All GUI/native imports are deferred to first use so this class can be
    instantiated (and the module imported) without pyautogui/mss/Pillow
    installed -- only calling its methods requires them.
    """

    def __init__(self, pyautogui_pause_s: float = 0.1) -> None:
        """Prepares the controller without touching any GUI library yet.

        Args:
            pyautogui_pause_s: Delay pyautogui inserts between actions.
        """
        self._pause_s = pyautogui_pause_s
        self._configured = False

    def _pyautogui(self) -> Any:
        """Lazily imports and one-time-configures pyautogui (FAILSAFE on)."""
        import pyautogui  # lazy: GUI dependency

        if not self._configured:
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = self._pause_s
            self._configured = True
        return pyautogui

    def screen_size_logical(self) -> tuple[int, int]:
        """Returns (width, height) in logical points, per pyautogui."""
        pyautogui = self._pyautogui()
        size = pyautogui.size()
        return int(size.width), int(size.height)

    def capture_screenshot(self) -> tuple[Any, ScreenGeometry]:
        """Grabs the primary display via mss and reports its geometry.

        The raw mss capture is in physical pixels; callers that want to
        avoid a manual SCALE division should resize this image to
        `screen_size_logical()` before sending it to Gemini (then
        `denormalize_box`/`denormalize_point` map directly to pyautogui
        coordinates with `norm_max`/logical dims, scale == 1.0).

        Returns:
            A tuple (pil_image, geometry).
        """
        import mss  # lazy: native screen-capture dependency
        from PIL import Image  # lazy: only needed alongside mss

        logical_w, logical_h = self.screen_size_logical()
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            raw = sct.grab(monitor)
        image = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
        geometry = ScreenGeometry(
            logical_width=logical_w,
            logical_height=logical_h,
            physical_width=raw.width,
            physical_height=raw.height,
        )
        return image, geometry

    def capture_screenshot_logical(self) -> Any:
        """Grabs the screen and resizes it to logical resolution (Retina-safe)."""
        from PIL import Image  # lazy

        image, geometry = self.capture_screenshot()
        if (geometry.physical_width, geometry.physical_height) != (
            geometry.logical_width,
            geometry.logical_height,
        ):
            image = image.resize((geometry.logical_width, geometry.logical_height), Image.LANCZOS)
        return image

    def click(self, x: int, y: int) -> None:
        """Clicks at logical pixel coordinates (x, y)."""
        self._pyautogui().click(x, y)

    def hotkey(self, *keys: str) -> None:
        """Sends a chorded key combination, e.g. hotkey('command', 'v')."""
        self._pyautogui().hotkey(*keys)

    def scroll(self, clicks: int) -> None:
        """Scrolls the given number of wheel clicks (positive = up)."""
        self._pyautogui().scroll(clicks)

    def type_text(self, text: str) -> None:
        """Types text reliably via the clipboard, with a typewrite fallback.

        Saves the current clipboard, pastes `text` via Cmd+V, then restores
        the previous clipboard contents. Falls back to `pyautogui.typewrite`
        if `pbcopy`/`pbpaste` are unavailable.

        Args:
            text: The text to type into the currently focused field.
        """
        pyautogui = self._pyautogui()
        previous_clipboard: str | None = None
        try:
            previous_clipboard = subprocess.run(
                ["pbpaste"], capture_output=True, text=True, check=True, timeout=_CLIPBOARD_TIMEOUT_S
            ).stdout
        except (OSError, subprocess.SubprocessError):
            previous_clipboard = None

        try:
            subprocess.run(["pbcopy"], input=text, text=True, check=True, timeout=_CLIPBOARD_TIMEOUT_S)
            pyautogui.hotkey("command", "v")
        except (OSError, subprocess.SubprocessError):
            pyautogui.typewrite(text, interval=_TYPEWRITE_INTERVAL_S)
        finally:
            if previous_clipboard is not None:
                try:
                    subprocess.run(
                        ["pbcopy"], input=previous_clipboard, text=True, check=True, timeout=_CLIPBOARD_TIMEOUT_S
                    )
                except (OSError, subprocess.SubprocessError):
                    pass

    def open_app(self, name: str) -> None:
        """Launches a macOS application by name via `open -a`.

        Args:
            name: Application name as `open -a` expects it.

        Raises:
            ActuatorError: If macOS reports the app cannot be opened -- a
                silent failure here would burn the step's whole attempt
                budget without anyone knowing why.
        """
        completed = subprocess.run(
            ["open", "-a", name], capture_output=True, text=True, timeout=_OPEN_TIMEOUT_S
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown application"
            raise ActuatorError(f"open -a {name!r} failed: {detail}")

    def open_url(self, url: str) -> None:
        """Opens a URL in the default browser via `open`.

        Args:
            url: The URL to open.

        Raises:
            ActuatorError: If macOS reports the URL cannot be opened.
        """
        completed = subprocess.run(
            ["open", url], capture_output=True, text=True, timeout=_OPEN_TIMEOUT_S
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown scheme"
            raise ActuatorError(f"open {url!r} failed: {detail}")

    def wait(self, seconds: float) -> None:
        """Blocks for `seconds` (used for UI settle time between actions)."""
        time.sleep(seconds)

    def execute(self, plan: "ActionPlan") -> dict[str, Any]:
        """Dispatches an `agent.ActionPlan` to the matching real Mac action.

        Args:
            plan: The action plan chosen by the agent loop this turn.

        Returns:
            A small dict describing what was executed (for logging).

        Raises:
            ActuatorError: If `plan.kind` is not a supported action.
        """
        if plan.kind == "click":
            x, y = plan.target
            self.click(x, y)
        elif plan.kind == "type":
            self.type_text(plan.text or "")
        elif plan.kind == "hotkey":
            self.hotkey(*(plan.target or ()))
        elif plan.kind == "scroll":
            self.scroll(int(plan.target or 0))
        elif plan.kind == "open_app":
            self.open_app(plan.text or "")
        elif plan.kind == "open_url":
            self.open_url(plan.text or "")
        elif plan.kind == "noop":
            pass
        else:
            raise ActuatorError(f"Unsupported action kind: {plan.kind!r}")
        return {"kind": plan.kind, "target": plan.target, "text": plan.text}


def check_macos_permissions() -> dict[str, bool]:
    """Probes the macOS permissions Continuum needs live.

    Accessibility and Screen Recording use the real CoreGraphics preflight
    calls (`CGPreflightPostEventAccess` / `CGPreflightScreenCaptureAccess`,
    via the pyobjc Quartz bindings pyautogui already depends on): reading
    the mouse position or grabbing a frame would SUCCEED without the
    permission (macOS silently ignores the clicks / returns a wallpaper-only
    frame), so only the preflights tell the truth. The microphone probe
    stays best-effort (listing devices does not require the permission; the
    OS prompt fires on first real recording). Input Monitoring (needed by
    the pynput key listener) has no cheap preflight -- covered in RUNBOOK.

    Returns:
        A dict with keys "accessibility", "screen_recording", "microphone",
        each True if the corresponding probe succeeded.
    """
    results = {"accessibility": False, "screen_recording": False, "microphone": False}

    try:
        import Quartz  # lazy: pyobjc, ships with pyautogui on macOS

        results["accessibility"] = bool(Quartz.CGPreflightPostEventAccess())
        results["screen_recording"] = bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        # No Quartz bindings (non-mac dev box): fall back to weak probes so
        # the boot check stays usable, while the docstring documents the gap.
        try:
            import pyautogui  # lazy

            pyautogui.position()
            results["accessibility"] = True
        except Exception:
            results["accessibility"] = False
        try:
            import mss  # lazy

            with mss.mss() as sct:
                sct.grab(sct.monitors[1])
            results["screen_recording"] = True
        except Exception:
            results["screen_recording"] = False

    try:
        import sounddevice as sd  # lazy

        sd.query_devices()
        results["microphone"] = True
    except Exception:
        results["microphone"] = False

    return results
