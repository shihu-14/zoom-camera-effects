"""On-frame OpenCV runtime controls for Zoom Camera Effects."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np

from ._ui_drawing import (
    _clip_rect,
    _draw_arrow_button,
    _draw_button,
    _draw_centered_text,
    _draw_gear_icon,
    _draw_translucent_rect,
    _draw_value_pill,
    _put_text,
    _rect_end,
    _rect_start,
)
from ._ui_options import (
    COLOR_CHOICES,
    COLORMAP_CHOICES,
    EFFECT_OPTIONS,
    NUMERIC_OPTIONS,
    NumericOption,
    _cycle_option,
    _cycle_option_label,
    _cycle_option_value,
    _format_numeric_value,
    _set_numeric_from_slider,
)
from .effects import (
    COLOR_NAMES_BGR,
    COLORMAPS,
    EFFECT_MODES,
    EFFECT_SCOPES,
    EffectConfig,
    EffectMode,
    MAX_AREA_POINTS,
    MIN_AREA_POINTS,
    format_color_hex,
)

PANEL_MARGIN = 12
GEAR_SIZE = 44
AREA_EDITOR_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True)
class HitRegion:
    kind: str
    payload: Any
    rect: tuple[int, int, int, int]


class OverlayControlUI:
    """Small clickable control panel rendered on top of the video frame."""

    def __init__(
        self,
        *,
        expanded: bool = True,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.expanded = expanded
        self._now = now or perf_counter
        self._regions: list[HitRegion] = []
        self._panel_rect: tuple[int, int, int, int] | None = None
        self._config = EffectConfig()
        self._pending_config: EffectConfig | None = None
        self._dragging_slider: str | None = None
        self._dragging_area_vertex: int | None = None
        self._area_editor_visible = True
        self._last_area_click_at = self._now()
        self._frame_size = (1, 1)

    def render(self, frame_bgr: np.ndarray, config: EffectConfig) -> np.ndarray:
        previous_scope = self._config.scope
        self._config = config
        self._regions = []
        self._panel_rect = None
        self._frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        if config.scope == "partial" and previous_scope != "partial":
            self._show_area_editor()
        output = frame_bgr.copy()
        if config.scope == "partial" and self._area_editor_should_show():
            self._draw_area_editor(output, config)
        self._draw_gear_button(output)
        if self.expanded:
            self._draw_panel(output, config)
        return output

    def handle_mouse(
        self,
        event: int,
        x: int,
        y: int,
        _flags: int,
        _param: object | None,
    ) -> None:
        if (
            self._config.scope == "partial"
            and event
            in {
                cv2.EVENT_LBUTTONDOWN,
                cv2.EVENT_RBUTTONDOWN,
                cv2.EVENT_LBUTTONDBLCLK,
            }
        ):
            self._show_area_editor()
        if event == cv2.EVENT_MOUSEMOVE and self._dragging_slider is not None:
            self._set_slider_value(self._dragging_slider, x)
            return
        if event == cv2.EVENT_MOUSEMOVE and self._dragging_area_vertex is not None:
            self._set_area_vertex(self._dragging_area_vertex, x, y)
            return
        if event == cv2.EVENT_LBUTTONUP:
            self._dragging_slider = None
            self._dragging_area_vertex = None
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            for region in reversed(self._regions):
                if _point_in_rect(x, y, region.rect) and region.kind in {
                    "area_vertex",
                    "area_delete",
                }:
                    self._delete_area_vertex(region.payload)
                    return
            return
        if event == cv2.EVENT_LBUTTONDBLCLK:
            for region in reversed(self._regions):
                if _point_in_rect(x, y, region.rect) and region.kind == "area_edge":
                    self._add_area_vertex(region.payload, x, y)
                    return
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for region in reversed(self._regions):
            if _point_in_rect(x, y, region.rect):
                self._activate(region, x, y)
                return
        if (
            self.expanded
            and self._panel_rect is not None
            and not _point_in_rect(x, y, self._panel_rect)
        ):
            self.expanded = False

    def consume_pending_config(
        self,
        base_config: EffectConfig,
    ) -> EffectConfig | None:
        if self._pending_config is None:
            self._config = base_config
            return None
        pending = self._pending_config
        self._pending_config = None
        self._config = pending
        return pending

    def _activate(self, region: HitRegion, x: int, y: int) -> None:
        if region.kind == "gear":
            self.expanded = not self.expanded
            return
        if region.kind == "effect":
            self._set_pending(replace(self._config, mode=region.payload))
            return
        if region.kind == "scope":
            if region.payload == "partial":
                self._show_area_editor()
            self._set_pending(replace(self._config, scope=region.payload))
            return
        if region.kind == "slider":
            self._dragging_slider = region.payload
            self._set_slider_value(region.payload, x)
            return
        if region.kind == "cycle":
            key, direction = region.payload
            self._set_pending(_cycle_option(self._config, key, direction))
            return
        if region.kind == "area_vertex":
            self._dragging_area_vertex = region.payload
            self._set_area_vertex(region.payload, x, y)
            return
        if region.kind in {"area_edge", "area_add"}:
            self._add_area_vertex(region.payload, x, y)
            return
        if region.kind == "area_delete":
            self._delete_area_vertex(region.payload)

    def _set_pending(self, config: EffectConfig) -> None:
        self._config = config
        self._pending_config = config

    def _show_area_editor(self) -> None:
        self._area_editor_visible = True
        self._last_area_click_at = self._now()

    def _area_editor_should_show(self) -> bool:
        if self._dragging_area_vertex is not None:
            return True
        if not self._area_editor_visible:
            return False
        if self._now() - self._last_area_click_at > AREA_EDITOR_TIMEOUT_SECONDS:
            self._area_editor_visible = False
            return False
        return True

    def _set_slider_value(self, key: str, x: int) -> None:
        region = next(
            (
                region
                for region in self._regions
                if region.kind == "slider" and region.payload == key
            ),
            None,
        )
        if region is None:
            return
        self._set_pending(_set_numeric_from_slider(self._config, key, x, region.rect))

    def _set_area_vertex(self, index: int, x: int, y: int) -> None:
        points = list(self._config.area_points)
        if not 0 <= index < len(points):
            return
        points[index] = _pixel_to_normalized_point(x, y, self._frame_size)
        self._set_area_points(tuple(points))

    def _add_area_vertex(self, edge_index: int, x: int, y: int) -> None:
        points = list(self._config.area_points)
        if len(points) >= MAX_AREA_POINTS or not 0 <= edge_index < len(points):
            return
        points.insert(
            edge_index + 1,
            _pixel_to_normalized_point(x, y, self._frame_size),
        )
        self._set_area_points(tuple(points))

    def _delete_area_vertex(self, index: int) -> None:
        points = list(self._config.area_points)
        if len(points) <= MIN_AREA_POINTS or not 0 <= index < len(points):
            return
        points.pop(index)
        self._set_area_points(tuple(points))

    def _set_area_points(self, points: tuple[tuple[float, float], ...]) -> None:
        try:
            config = replace(self._config, area_points=points)
        except ValueError:
            return
        self._set_pending(config)

    def _draw_area_editor(self, image: np.ndarray, config: EffectConfig) -> None:
        points = _area_points_to_pixels(config.area_points, self._frame_size)
        if len(points) < MIN_AREA_POINTS:
            return

        polygon = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon], (35, 178, 232), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.16, image, 0.84, 0, dst=image)
        cv2.polylines(image, [polygon], True, (30, 214, 255), 3, cv2.LINE_AA)
        cv2.polylines(image, [polygon], True, (8, 63, 83), 1, cv2.LINE_AA)

        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            self._regions.append(
                HitRegion(
                    "area_edge",
                    index,
                    _line_hit_rect(start, end, self._frame_size),
                )
            )
            if len(points) < MAX_AREA_POINTS:
                midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
                add_rect = _center_rect(midpoint, 18)
                self._regions.append(HitRegion("area_add", index, add_rect))
                cv2.circle(image, midpoint, 9, (12, 55, 68), -1, cv2.LINE_AA)
                cv2.circle(image, midpoint, 9, (50, 224, 255), 2, cv2.LINE_AA)
                _draw_centered_text(image, "+", add_rect, 0.48, (230, 252, 255), 1)

        for index, point in enumerate(points):
            vertex_rect = _center_rect(point, 24)
            self._regions.append(HitRegion("area_vertex", index, vertex_rect))
            cv2.circle(image, point, 10, (245, 251, 255), -1, cv2.LINE_AA)
            cv2.circle(image, point, 10, (0, 178, 255), 2, cv2.LINE_AA)
            cv2.circle(image, point, 4, (8, 63, 83), -1, cv2.LINE_AA)
            if len(points) > MIN_AREA_POINTS:
                delete_center = (
                    min(max(point[0] + 16, 8), self._frame_size[0] - 8),
                    min(max(point[1] - 16, 8), self._frame_size[1] - 8),
                )
                delete_rect = _center_rect(delete_center, 16)
                self._regions.append(HitRegion("area_delete", index, delete_rect))
                cv2.circle(image, delete_center, 8, (31, 37, 44), -1, cv2.LINE_AA)
                cv2.circle(image, delete_center, 8, (96, 128, 145), 1, cv2.LINE_AA)
                _draw_centered_text(image, "x", delete_rect, 0.36, (255, 215, 215), 1)

    def _draw_gear_button(self, image: np.ndarray) -> None:
        rect = (PANEL_MARGIN, PANEL_MARGIN, GEAR_SIZE, GEAR_SIZE)
        self._regions.append(HitRegion("gear", None, rect))
        _draw_translucent_rect(image, rect, (22, 28, 34), alpha=0.88)
        cv2.rectangle(image, _rect_start(rect), _rect_end(rect), (105, 122, 139), 1)
        center = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
        _draw_gear_icon(image, center, 13, (235, 242, 248))

    def _draw_panel(self, image: np.ndarray, config: EffectConfig) -> None:
        height, width = image.shape[:2]
        x = PANEL_MARGIN
        y = PANEL_MARGIN + GEAR_SIZE + 8
        panel_width = max(280, min(420, width - PANEL_MARGIN * 2))
        effect_rows = (len(EFFECT_MODES) + 1) // 2
        option_count = max(1, len(EFFECT_OPTIONS[config.mode]))
        panel_height = 48 + effect_rows * 32 + 50 + 34 + option_count * 38 + 14
        panel_height = min(panel_height, max(96, height - y - PANEL_MARGIN))
        panel_rect = (x, y, panel_width, panel_height)
        self._panel_rect = panel_rect
        _draw_translucent_rect(image, panel_rect, (18, 24, 30), alpha=0.86)
        cv2.rectangle(
            image,
            _rect_start(panel_rect),
            _rect_end(panel_rect),
            (82, 97, 112),
            1,
        )

        cursor_y = y + 28
        _put_text(image, "Effect", (x + 16, cursor_y), 0.48, (236, 243, 248), 1)
        cursor_y += 16
        cursor_y = self._draw_effect_buttons(
            image, config, x + 14, cursor_y, panel_width - 28
        )
        cursor_y += 24
        _put_text(image, "Scope", (x + 16, cursor_y), 0.48, (236, 243, 248), 1)
        cursor_y += 12
        cursor_y = self._draw_scope_buttons(
            image, config, x + 14, cursor_y, panel_width - 28
        )
        cursor_y += 24
        _put_text(image, "Options", (x + 16, cursor_y), 0.48, (236, 243, 248), 1)
        cursor_y += 12
        self._draw_options(image, config, x + 14, cursor_y, panel_width - 28)

    def _draw_effect_buttons(
        self,
        image: np.ndarray,
        config: EffectConfig,
        x: int,
        y: int,
        width: int,
    ) -> int:
        column_gap = 8
        button_height = 25
        button_width = (width - column_gap) // 2
        for index, mode in enumerate(EFFECT_MODES):
            column = index % 2
            row = index // 2
            rect = (
                x + column * (button_width + column_gap),
                y + row * 32,
                button_width,
                button_height,
            )
            self._regions.append(HitRegion("effect", mode, rect))
            _draw_button(image, rect, mode, active=mode == config.mode)
        return y + ((len(EFFECT_MODES) + 1) // 2) * 32

    def _draw_scope_buttons(
        self,
        image: np.ndarray,
        config: EffectConfig,
        x: int,
        y: int,
        width: int,
    ) -> int:
        column_gap = 8
        button_height = 28
        button_count = len(EFFECT_SCOPES)
        button_width = (width - column_gap * (button_count - 1)) // button_count
        for index, scope in enumerate(EFFECT_SCOPES):
            rect = (
                x + index * (button_width + column_gap),
                y,
                button_width,
                button_height,
            )
            self._regions.append(HitRegion("scope", scope, rect))
            _draw_button(image, rect, scope, active=scope == config.scope)
        return y + 38

    def _draw_options(
        self,
        image: np.ndarray,
        config: EffectConfig,
        x: int,
        y: int,
        width: int,
    ) -> None:
        option_keys = EFFECT_OPTIONS[config.mode]
        if not option_keys:
            _put_text(image, "none", (x + 2, y + 24), 0.45, (184, 195, 206), 1)
            return

        cursor_y = y
        for key in option_keys:
            row_rect = (x, cursor_y, width, 30)
            if key in NUMERIC_OPTIONS:
                self._draw_numeric_option(image, config, key, row_rect)
            else:
                self._draw_cycle_option(image, config, key, row_rect)
            cursor_y += 38

    def _draw_numeric_option(
        self,
        image: np.ndarray,
        config: EffectConfig,
        key: str,
        rect: tuple[int, int, int, int],
    ) -> None:
        option = NUMERIC_OPTIONS[key]
        value = _format_numeric_value(option, getattr(config, key))
        _put_text(
            image,
            option.label,
            (rect[0] + 2, rect[1] + 21),
            0.43,
            (215, 224, 232),
            1,
        )
        _put_text(
            image,
            value,
            (rect[0] + rect[2] - 48, rect[1] + 21),
            0.43,
            (235, 242, 248),
            1,
        )
        self._draw_slider(image, config, key, rect)

    def _draw_slider(
        self,
        image: np.ndarray,
        config: EffectConfig,
        key: str,
        rect: tuple[int, int, int, int],
    ) -> None:
        option = NUMERIC_OPTIONS[key]
        slider_rect = (rect[0] + 112, rect[1] + 8, max(40, rect[2] - 174), 14)
        self._regions.append(HitRegion("slider", key, slider_rect))
        value = float(getattr(config, key))
        ratio = (value - option.minimum) / max(option.maximum - option.minimum, 1e-6)
        ratio = min(max(ratio, 0.0), 1.0)
        track_y = slider_rect[1] + slider_rect[3] // 2
        start = (slider_rect[0], track_y)
        end = (slider_rect[0] + slider_rect[2], track_y)
        knob_x = int(round(slider_rect[0] + slider_rect[2] * ratio))
        cv2.line(image, start, end, (83, 101, 118), 4, cv2.LINE_AA)
        cv2.line(image, start, (knob_x, track_y), (118, 180, 226), 4, cv2.LINE_AA)
        cv2.circle(image, (knob_x, track_y), 6, (238, 246, 252), -1, cv2.LINE_AA)
        cv2.circle(image, (knob_x, track_y), 6, (78, 99, 118), 1, cv2.LINE_AA)

    def _draw_cycle_option(
        self,
        image: np.ndarray,
        config: EffectConfig,
        key: str,
        rect: tuple[int, int, int, int],
    ) -> None:
        label = _cycle_option_label(key)
        value = _cycle_option_value(config, key)
        _put_text(
            image,
            label,
            (rect[0] + 2, rect[1] + 21),
            0.43,
            (215, 224, 232),
            1,
        )
        selector_rect = (rect[0] + 110, rect[1], rect[2] - 110, rect[3])
        self._draw_choice_selector(
            image,
            value=value,
            rect=selector_rect,
            kind="cycle",
            key=key,
        )

    def _draw_choice_selector(
        self,
        image: np.ndarray,
        value: str,
        rect: tuple[int, int, int, int],
        *,
        kind: str,
        key: str,
    ) -> None:
        arrow_size = 28
        left_rect = (rect[0], rect[1], arrow_size, arrow_size)
        right_rect = (rect[0] + rect[2] - arrow_size, rect[1], arrow_size, arrow_size)
        value_rect = (
            rect[0] + arrow_size + 6,
            rect[1],
            max(32, rect[2] - arrow_size * 2 - 12),
            arrow_size,
        )
        self._regions.append(HitRegion(kind, (key, -1), left_rect))
        self._regions.append(HitRegion(kind, (key, 1), right_rect))
        _draw_arrow_button(image, left_rect, -1)
        _draw_arrow_button(image, right_rect, 1)
        _draw_value_pill(image, value_rect, value)


def _pixel_to_normalized_point(
    x: int,
    y: int,
    frame_size: tuple[int, int],
) -> tuple[float, float]:
    width, height = frame_size
    normalized_x = x / max(width - 1, 1)
    normalized_y = y / max(height - 1, 1)
    return (
        min(max(normalized_x, 0.0), 1.0),
        min(max(normalized_y, 0.0), 1.0),
    )


def _area_points_to_pixels(
    points: tuple[tuple[float, float], ...],
    frame_size: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    width, height = frame_size
    return tuple(
        (
            int(round(x * max(width - 1, 1))),
            int(round(y * max(height - 1, 1))),
        )
        for x, y in points
    )


def _line_hit_rect(
    start: tuple[int, int],
    end: tuple[int, int],
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    padding = 10
    min_x = min(start[0], end[0]) - padding
    min_y = min(start[1], end[1]) - padding
    max_x = max(start[0], end[0]) + padding
    max_y = max(start[1], end[1]) + padding
    return _clip_rect(
        (min_x, min_y, max_x - min_x, max_y - min_y),
        frame_size[0],
        frame_size[1],
    )


def _center_rect(center: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    return (center[0] - size // 2, center[1] - size // 2, size, size)


def _point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zoom Camera Effects on-frame UI helpers."
    )
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
