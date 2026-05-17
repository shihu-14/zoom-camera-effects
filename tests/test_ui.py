import cv2
import numpy as np

from finger_quad_effect.effects import EffectConfig
from finger_quad_effect.ui import OverlayControlUI


def test_overlay_ui_renders_on_frame():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    output = ui.render(frame, EffectConfig(mode="blur"))

    assert output.shape == frame.shape
    assert output.max() > frame.max()


def test_overlay_ui_shows_only_current_effect_options():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    ui.render(frame, EffectConfig(mode="blur"))
    payloads = [region.payload for region in ui._regions if region.kind == "slider"]

    assert "kernel_size" in payloads
    assert "mosaic_block_size" not in payloads


def test_overlay_ui_gear_collapses_panel():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ui.render(frame, EffectConfig())
    gear = next(region for region in ui._regions if region.kind == "gear")
    x = gear.rect[0] + gear.rect[2] // 2
    y = gear.rect[1] + gear.rect[3] // 2

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    ui.render(frame, EffectConfig())

    assert ui.expanded is False
    assert {region.kind for region in ui._regions} == {"gear"}


def test_overlay_ui_click_outside_panel_collapses_panel():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ui.render(frame, EffectConfig())

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, 639, 479, 0, None)

    assert ui.expanded is False


def test_overlay_ui_click_inside_panel_keeps_panel_open():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ui.render(frame, EffectConfig())
    assert ui._panel_rect is not None
    x = ui._panel_rect[0] + 8
    y = ui._panel_rect[1] + 8

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

    assert ui.expanded is True


def test_overlay_ui_can_switch_effect():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current = EffectConfig(mode="blur")
    ui.render(frame, current)
    edge = next(
        region
        for region in ui._regions
        if region.kind == "effect" and region.payload == "edge"
    )
    x = edge.rect[0] + edge.rect[2] // 2
    y = edge.rect[1] + edge.rect[3] // 2

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

    assert ui.consume_pending_config(current).mode == "edge"


def test_overlay_ui_can_cycle_current_option():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current = EffectConfig(mode="thermal", thermal_colormap="jet")
    ui.render(frame, current)
    next_colormap = next(
        region
        for region in ui._regions
        if region.kind == "cycle" and region.payload == ("thermal_colormap", 1)
    )
    x = next_colormap.rect[0] + next_colormap.rect[2] // 2
    y = next_colormap.rect[1] + next_colormap.rect[3] // 2

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

    assert ui.consume_pending_config(current).thermal_colormap == "turbo"


def test_overlay_ui_can_adjust_current_option_with_slider():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current = EffectConfig(mode="blur", kernel_size=35)
    ui.render(frame, current)
    slider = next(
        region
        for region in ui._regions
        if region.kind == "slider" and region.payload == "kernel_size"
    )
    x = slider.rect[0] + slider.rect[2]
    y = slider.rect[1] + slider.rect[3] // 2

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

    assert ui.consume_pending_config(current).kernel_size == 101


def test_overlay_ui_slider_drag_updates_value():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current = EffectConfig(mode="noise", noise_strength=0.0)
    ui.render(frame, current)
    slider = next(
        region
        for region in ui._regions
        if region.kind == "slider" and region.payload == "noise_strength"
    )
    y = slider.rect[1] + slider.rect[3] // 2

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, slider.rect[0], y, 0, None)
    ui.handle_mouse(cv2.EVENT_MOUSEMOVE, slider.rect[0] + slider.rect[2], y, 0, None)
    ui.handle_mouse(cv2.EVENT_LBUTTONUP, slider.rect[0] + slider.rect[2], y, 0, None)

    assert ui.consume_pending_config(current).noise_strength == 1.0
