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
    payloads = [region.payload for region in ui._regions if region.kind == "numeric"]

    assert ("kernel_size", 1) in payloads
    assert ("sigma", 1) not in payloads
    assert ("mosaic_block_size", 1) not in payloads


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


def test_overlay_ui_can_adjust_current_option():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current = EffectConfig(mode="blur", kernel_size=35)
    ui.render(frame, current)
    increase = next(
        region
        for region in ui._regions
        if region.kind == "numeric" and region.payload == ("kernel_size", 1)
    )
    x = increase.rect[0] + increase.rect[2] // 2
    y = increase.rect[1] + increase.rect[3] // 2

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

    assert ui.consume_pending_config(current).kernel_size == 37
