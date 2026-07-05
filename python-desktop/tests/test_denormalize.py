"""Tests for pure coordinate math in mac_control.py (no GUI libs required)."""

from __future__ import annotations

import mac_control as mc


def test_denormalize_point_center_of_screen() -> None:
    # 500/1000 of a 1000x2000 screen is dead center.
    assert mc.denormalize_point(500, 500, width=1000, height=2000) == (500, 1000)


def test_denormalize_point_corners_are_clamped_on_screen() -> None:
    # (0,0) and (1000,1000) map to the exact screen corners, which are BOTH
    # off-limits: 1000 lands one pixel off screen, and the corners trigger
    # pyautogui's FAILSAFE (reserved as the operator's emergency abort).
    assert mc.denormalize_point(0, 0, width=1440, height=900) == (1, 1)
    assert mc.denormalize_point(1000, 1000, width=1440, height=900) == (1438, 898)


def test_denormalize_point_cu_range_999_center_and_corners() -> None:
    # The Interactions Computer Use path returns 0-999 points (norm_max=999):
    # ~500/999 of the screen is the center, and the ceiling 999 maps to the
    # far edge, clamped one pixel inside like every other path.
    assert mc.denormalize_point(500, 500, width=1000, height=2000, norm_max=999) == (501, 1001)
    assert mc.denormalize_point(999, 999, width=1440, height=900, norm_max=999) == (1438, 898)
    assert mc.denormalize_point(0, 0, width=1440, height=900, norm_max=999) == (1, 1)


def test_denormalize_point_cu_range_asymmetric_pins_x_y_order() -> None:
    # x-first, y-second (a {x, y} POINT, not a [y, x, ...] box): a swap would
    # pass every symmetric fixture but transpose real clicks. width != height,
    # x=999 lands on the wide axis, y=0 on the top edge.
    assert mc.denormalize_point(999, 0, width=2000, height=1000, norm_max=999) == (1998, 1)


def test_denormalize_box_center() -> None:
    # Box covering the whole normalized space -> center of the real screen.
    box = (0, 0, 1000, 1000)
    assert mc.denormalize_box(box, width=1440, height=900) == (720, 450)


def test_denormalize_box_quadrant() -> None:
    # ymin, xmin, ymax, xmax = top-left quadrant.
    box = (0, 0, 500, 500)
    assert mc.denormalize_box(box, width=1000, height=1000) == (250, 250)


def test_denormalize_box_asymmetric_pins_gemini_axis_order() -> None:
    # THE documented trap: Gemini boxes are [ymin, xmin, ymax, xmax]. An
    # implementation that read them as [xmin, ymin, xmax, ymax] would pass
    # every symmetric fixture above but transpose every real click. This
    # asymmetric box (y center 150, x center 400) pins the axis order.
    box = (100, 300, 200, 500)
    assert mc.denormalize_box(box, width=1000, height=1000) == (400, 150)
    # And with width != height, so a scale mix-up cannot hide either.
    assert mc.denormalize_box(box, width=2000, height=1000) == (800, 150)


def test_compute_retina_scale_doubles_on_retina() -> None:
    assert mc.compute_retina_scale(physical_width=2880, logical_width=1440) == 2.0


def test_compute_retina_scale_one_on_non_retina() -> None:
    assert mc.compute_retina_scale(physical_width=1440, logical_width=1440) == 1.0


def test_compute_retina_scale_rejects_zero_logical_width() -> None:
    try:
        mc.compute_retina_scale(physical_width=2880, logical_width=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_apply_scale_divides_physical_pixel_to_logical() -> None:
    # A click computed against a 2880-wide physical capture, scale=2,
    # should land at the correct logical (pyautogui) coordinate.
    assert mc.apply_scale((720, 450), scale=2.0) == (360, 225)


def test_apply_scale_noop_at_scale_one() -> None:
    assert mc.apply_scale((100, 200), scale=1.0) == (100, 200)


def test_full_pipeline_gemini_box_to_retina_logical_click() -> None:
    # Simulates: Gemini graded a box against a physical 2880x1800 screenshot
    # (i.e. we did NOT resize to logical first), so we must apply SCALE.
    box = (0, 0, 1000, 1000)  # whole screen -> its center
    physical_w, physical_h = 2880, 1800
    logical_w, logical_h = 1440, 900

    pixel = mc.denormalize_box(box, width=physical_w, height=physical_h)
    scale = mc.compute_retina_scale(physical_w, logical_w)
    logical_pixel = mc.apply_scale(pixel, scale)

    assert logical_pixel == (720, 450)  # center of the 1440x900 logical screen
