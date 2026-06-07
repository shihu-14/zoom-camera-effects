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


def test_overlay_ui_can_switch_scope():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current = EffectConfig()
    ui.render(frame, current)
    full_scope = next(
        region
        for region in ui._regions
        if region.kind == "scope" and region.payload == "full"
    )
    x = full_scope.rect[0] + full_scope.rect[2] // 2
    y = full_scope.rect[1] + full_scope.rect[3] // 2

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

    assert ui.consume_pending_config(current).scope == "full"


def test_overlay_ui_can_toggle_outline_fill_option():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current = EffectConfig(mode="outline", outline_fill=False)
    ui.render(frame, current)
    fill_toggle = next(
        region
        for region in ui._regions
        if region.kind == "cycle" and region.payload == ("outline_fill", 1)
    )
    x = fill_toggle.rect[0] + fill_toggle.rect[2] // 2
    y = fill_toggle.rect[1] + fill_toggle.rect[3] // 2

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

    assert ui.consume_pending_config(current).outline_fill is True


def test_overlay_ui_can_cycle_outline_fill_color():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current = EffectConfig(mode="outline", outline_fill_color_bgr=(255, 255, 255))
    ui.render(frame, current)
    next_color = next(
        region
        for region in ui._regions
        if region.kind == "cycle" and region.payload == ("outline_fill_color_bgr", 1)
    )
    x = next_color.rect[0] + next_color.rect[2] // 2
    y = next_color.rect[1] + next_color.rect[3] // 2

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

    assert ui.consume_pending_config(current).outline_fill_color_bgr == (255, 255, 0)


def test_overlay_ui_can_drag_partial_area_vertex():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current = EffectConfig(scope="partial")
    ui.render(frame, current)
    vertex = next(
        region
        for region in ui._regions
        if region.kind == "area_vertex" and region.payload == 2
    )
    x = vertex.rect[0] + vertex.rect[2] // 2
    y = vertex.rect[1] + vertex.rect[3] // 2

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    ui.handle_mouse(cv2.EVENT_MOUSEMOVE, 320, 240, 0, None)
    ui.handle_mouse(cv2.EVENT_LBUTTONUP, 320, 240, 0, None)

    updated = ui.consume_pending_config(current)
    assert updated.area_points[2] == (320 / 639, 240 / 479)


def test_overlay_ui_can_add_and_delete_partial_area_vertex():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current = EffectConfig(scope="partial")
    ui.render(frame, current)
    add = next(
        region
        for region in ui._regions
        if region.kind == "area_add" and region.payload == 1
    )
    add_x = add.rect[0] + add.rect[2] // 2
    add_y = add.rect[1] + add.rect[3] // 2

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, add_x, add_y, 0, None)
    added = ui.consume_pending_config(current)

    assert len(added.area_points) == 5

    ui.render(frame, added)
    delete = next(
        region
        for region in ui._regions
        if region.kind == "area_delete" and region.payload == 2
    )
    delete_x = delete.rect[0] + delete.rect[2] // 2
    delete_y = delete.rect[1] + delete.rect[3] // 2

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, delete_x, delete_y, 0, None)
    deleted = ui.consume_pending_config(added)

    assert len(deleted.area_points) == 4


def test_overlay_ui_limits_partial_area_vertex_count():
    ui = OverlayControlUI()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    triangle = EffectConfig(
        scope="partial",
        area_points=((0.1, 0.1), (0.9, 0.1), (0.1, 0.9)),
    )
    ui.render(frame, triangle)

    assert not any(region.kind == "area_delete" for region in ui._regions)

    ten_points = (
        (0.1, 0.1),
        (0.3, 0.1),
        (0.5, 0.1),
        (0.7, 0.1),
        (0.9, 0.1),
        (0.9, 0.5),
        (0.9, 0.9),
        (0.5, 0.9),
        (0.1, 0.9),
        (0.1, 0.5),
    )
    ui.render(frame, EffectConfig(scope="partial", area_points=ten_points))

    assert not any(region.kind == "area_add" for region in ui._regions)


def test_overlay_ui_hides_partial_area_editor_after_inactivity():
    now = 0.0

    def fake_now():
        return now

    ui = OverlayControlUI(now=fake_now)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current = EffectConfig(scope="partial")

    ui.render(frame, current)
    assert any(region.kind == "area_vertex" for region in ui._regions)

    now = 3.1
    ui.render(frame, current)
    assert not any(region.kind == "area_vertex" for region in ui._regions)
    assert not any(region.kind == "area_edge" for region in ui._regions)

    ui.handle_mouse(cv2.EVENT_LBUTTONDOWN, 320, 240, 0, None)
    ui.render(frame, current)
    assert any(region.kind == "area_vertex" for region in ui._regions)


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
