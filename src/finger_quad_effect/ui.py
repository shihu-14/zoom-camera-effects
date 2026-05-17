"""On-frame OpenCV runtime controls for Finger Quad Effect."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from .effects import (
    COLOR_NAMES_BGR,
    COLORMAPS,
    EFFECT_MODES,
    EffectConfig,
    EffectMode,
    format_color_hex,
)

COLOR_CHOICES = (
    "black",
    "white",
    "cyan",
    "magenta",
    "yellow",
    "red",
    "green",
    "blue",
)
COLORMAP_CHOICES = tuple(COLORMAPS)
PANEL_MARGIN = 12
GEAR_SIZE = 44


@dataclass(frozen=True)
class HitRegion:
    kind: str
    payload: Any
    rect: tuple[int, int, int, int]


@dataclass(frozen=True)
class NumericOption:
    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    integer: bool = False
    odd: bool = False
    decimals: int = 0


NUMERIC_OPTIONS = {
    "kernel_size": NumericOption(
        "kernel_size", "kernel", 3, 101, 2, integer=True, odd=True
    ),
    "edge_feather_px": NumericOption(
        "edge_feather_px", "edge feather", 0, 40, 1, integer=True
    ),
    "mosaic_block_size": NumericOption(
        "mosaic_block_size", "block", 1, 64, 1, integer=True
    ),
    "edge_low_threshold": NumericOption(
        "edge_low_threshold", "low", 0, 255, 5, decimals=0
    ),
    "edge_high_threshold": NumericOption(
        "edge_high_threshold", "high", 0, 255, 5, decimals=0
    ),
    "noise_strength": NumericOption(
        "noise_strength", "strength", 0.0, 1.0, 0.05, decimals=2
    ),
    "outline_thickness": NumericOption(
        "outline_thickness", "thickness", 0, 30, 1, integer=True
    ),
}

EFFECT_OPTIONS: dict[EffectMode, tuple[str, ...]] = {
    "blur": ("kernel_size", "edge_feather_px"),
    "mosaic": ("mosaic_block_size", "edge_feather_px"),
    "invert": ("edge_feather_px",),
    "grayscale": ("edge_feather_px",),
    "edge": ("edge_low_threshold", "edge_high_threshold", "edge_feather_px"),
    "thermal": ("thermal_colormap", "edge_feather_px"),
    "noise": ("noise_strength", "edge_feather_px"),
    "outline": ("outline_thickness", "outline_color_bgr"),
    "particles": (),
    "neon": (
        "edge_low_threshold",
        "edge_high_threshold",
        "noise_strength",
        "outline_color_bgr",
        "edge_feather_px",
    ),
    "glitch": ("noise_strength", "edge_feather_px"),
    "cartoon": ("edge_feather_px",),
    "sketch": ("edge_feather_px",),
}


class OverlayControlUI:
    """Small clickable control panel rendered on top of the video frame."""

    def __init__(self, *, expanded: bool = True) -> None:
        self.expanded = expanded
        self._regions: list[HitRegion] = []
        self._config = EffectConfig()
        self._pending_config: EffectConfig | None = None

    def render(self, frame_bgr: np.ndarray, config: EffectConfig) -> np.ndarray:
        self._config = config
        self._regions = []
        output = frame_bgr.copy()
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
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for region in reversed(self._regions):
            if _point_in_rect(x, y, region.rect):
                self._activate(region)
                return

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

    def _activate(self, region: HitRegion) -> None:
        if region.kind == "gear":
            self.expanded = not self.expanded
            return
        if region.kind == "effect":
            self._set_pending(replace(self._config, mode=region.payload))
            return
        if region.kind == "numeric":
            key, direction = region.payload
            self._set_pending(_adjust_numeric(self._config, key, direction))
            return
        if region.kind == "cycle":
            key, direction = region.payload
            self._set_pending(_cycle_option(self._config, key, direction))

    def _set_pending(self, config: EffectConfig) -> None:
        self._config = config
        self._pending_config = config

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
        panel_height = 48 + effect_rows * 32 + 34 + option_count * 38 + 14
        panel_height = min(panel_height, max(96, height - y - PANEL_MARGIN))
        panel_rect = (x, y, panel_width, panel_height)
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
            (rect[0] + 112, rect[1] + 21),
            0.43,
            (235, 242, 248),
            1,
        )
        self._draw_step_buttons(image, key, rect)

    def _draw_cycle_option(
        self,
        image: np.ndarray,
        config: EffectConfig,
        key: str,
        rect: tuple[int, int, int, int],
    ) -> None:
        label = "colormap" if key == "thermal_colormap" else "color"
        value = (
            config.thermal_colormap
            if key == "thermal_colormap"
            else _color_label(config.outline_color_bgr)
        )
        _put_text(
            image,
            label,
            (rect[0] + 2, rect[1] + 21),
            0.43,
            (215, 224, 232),
            1,
        )
        _put_text(
            image,
            value,
            (rect[0] + 112, rect[1] + 21),
            0.43,
            (235, 242, 248),
            1,
        )
        self._draw_step_buttons(image, key, rect, kind="cycle")

    def _draw_step_buttons(
        self,
        image: np.ndarray,
        key: str,
        rect: tuple[int, int, int, int],
        *,
        kind: str = "numeric",
    ) -> None:
        button_size = 28
        decrease = (
            rect[0] + rect[2] - button_size * 2 - 8,
            rect[1],
            button_size,
            button_size,
        )
        increase = (rect[0] + rect[2] - button_size, rect[1], button_size, button_size)
        self._regions.append(HitRegion(kind, (key, -1), decrease))
        self._regions.append(HitRegion(kind, (key, 1), increase))
        _draw_button(image, decrease, "-", active=False)
        _draw_button(image, increase, "+", active=False)


def _adjust_numeric(config: EffectConfig, key: str, direction: int) -> EffectConfig:
    option = NUMERIC_OPTIONS[key]
    value = float(getattr(config, key)) + option.step * direction
    value = min(max(value, option.minimum), option.maximum)
    if option.integer:
        value = int(round(value))
    if option.odd:
        value = int(value)
        if value % 2 == 0:
            value += 1 if direction >= 0 else -1
        value = min(max(value, int(option.minimum)), int(option.maximum))
    return replace(config, **{key: value})


def _cycle_option(config: EffectConfig, key: str, direction: int) -> EffectConfig:
    if key == "thermal_colormap":
        value = _cycle_value(config.thermal_colormap, COLORMAP_CHOICES, direction)
        return replace(config, thermal_colormap=value)
    if key == "outline_color_bgr":
        current = _color_label(config.outline_color_bgr)
        value = _cycle_value(current, COLOR_CHOICES, direction)
        return replace(config, outline_color_bgr=COLOR_NAMES_BGR[value])
    return config


def _cycle_value(value: str, choices: tuple[str, ...], direction: int) -> str:
    try:
        index = choices.index(value)
    except ValueError:
        index = 0
    return choices[(index + direction) % len(choices)]


def _format_numeric_value(option: NumericOption, value: float | int) -> str:
    if option.key == "outline_thickness" and int(value) == 0:
        return "auto"
    if option.integer:
        return str(int(value))
    return f"{float(value):.{option.decimals}f}"


def _color_label(color_bgr: tuple[int, int, int]) -> str:
    for name in COLOR_CHOICES:
        if COLOR_NAMES_BGR[name] == color_bgr:
            return name
    return format_color_hex(color_bgr)


def _draw_button(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    text: str,
    *,
    active: bool,
) -> None:
    fill = (50, 86, 118) if active else (36, 44, 52)
    border = (114, 176, 224) if active else (82, 96, 110)
    text_color = (248, 252, 255) if active else (214, 224, 232)
    cv2.rectangle(image, _rect_start(rect), _rect_end(rect), fill, -1)
    cv2.rectangle(image, _rect_start(rect), _rect_end(rect), border, 1)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    text_x = rect[0] + max(6, (rect[2] - text_size[0]) // 2)
    text_y = rect[1] + (rect[3] + text_size[1]) // 2
    _put_text(image, text, (text_x, text_y), 0.42, text_color, 1)


def _draw_translucent_rect(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    *,
    alpha: float,
) -> None:
    x, y, width, height = _clip_rect(rect, image.shape[1], image.shape[0])
    if width <= 0 or height <= 0:
        return
    roi = image[y : y + height, x : x + width]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _draw_gear_icon(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    cx, cy = center
    for angle in np.linspace(0, np.pi * 2, 8, endpoint=False):
        inner = (
            int(cx + np.cos(angle) * radius * 0.72),
            int(cy + np.sin(angle) * radius * 0.72),
        )
        outer = (
            int(cx + np.cos(angle) * radius * 1.12),
            int(cy + np.sin(angle) * radius * 1.12),
        )
        cv2.line(image, inner, outer, color, 2, cv2.LINE_AA)
    cv2.circle(image, center, radius, color, 2, cv2.LINE_AA)
    cv2.circle(image, center, max(3, radius // 3), color, 2, cv2.LINE_AA)


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _rect_start(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    return (rect[0], rect[1])


def _rect_end(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    return (rect[0] + rect[2], rect[1] + rect[3])


def _clip_rect(
    rect: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = rect
    x = min(max(x, 0), image_width)
    y = min(max(y, 0), image_height)
    width = min(width, image_width - x)
    height = min(height, image_height - y)
    return x, y, width, height


def _point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finger Quad Effect on-frame UI helpers."
    )
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
