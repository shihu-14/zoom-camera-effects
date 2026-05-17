"""Video effects for polygon-localized processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, cast

import cv2
import numpy as np

from .geometry import Point, normalized_to_pixels

EffectMode = Literal[
    "blur",
    "mosaic",
    "invert",
    "grayscale",
    "edge",
    "thermal",
    "noise",
    "outline",
    "neon",
    "glitch",
    "cartoon",
    "sketch",
]

EFFECT_MODES: tuple[EffectMode, ...] = (
    "blur",
    "mosaic",
    "invert",
    "grayscale",
    "edge",
    "thermal",
    "noise",
    "outline",
    "neon",
    "glitch",
    "cartoon",
    "sketch",
)
EFFECT_DESCRIPTIONS: dict[EffectMode, str] = {
    "blur": "Gaussian blur inside the fingertip quadrilateral.",
    "mosaic": "Pixelated mosaic blocks inside the fingertip quadrilateral.",
    "invert": "Inverted colors inside the fingertip quadrilateral.",
    "grayscale": "Monochrome grayscale inside the fingertip quadrilateral.",
    "edge": "Canny edge detection inside the fingertip quadrilateral.",
    "thermal": "False-color thermal palette inside the fingertip quadrilateral.",
    "noise": "Deterministic color noise inside the fingertip quadrilateral.",
    "outline": "Black outline around the fingertip quadrilateral.",
    "neon": "Glowing neon edges inside the fingertip quadrilateral.",
    "glitch": "RGB channel shift, sliced offsets, and scanlines.",
    "cartoon": "OpenCV stylization for a softened cartoon look.",
    "sketch": "OpenCV pencil sketch rendering.",
}
EFFECT_ALIASES = {"monochrome": "grayscale"}
COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "hot": cv2.COLORMAP_HOT,
    "cool": cv2.COLORMAP_COOL,
    "hsv": cv2.COLORMAP_HSV,
    "ocean": cv2.COLORMAP_OCEAN,
    "winter": cv2.COLORMAP_WINTER,
}
COLOR_NAMES_BGR = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "yellow": (0, 255, 255),
}


@dataclass(frozen=True)
class EffectConfig:
    mode: EffectMode = "blur"
    kernel_size: int = 35
    mosaic_block_size: int = 18
    edge_low_threshold: float = 60.0
    edge_high_threshold: float = 140.0
    thermal_colormap: str = "jet"
    noise_strength: float = 0.8
    outline_thickness: int = 0
    outline_color_bgr: tuple[int, int, int] = (0, 0, 0)


def normalize_effect_mode(value: str) -> EffectMode:
    mode = EFFECT_ALIASES.get(value, value)
    if mode not in EFFECT_MODES:
        raise ValueError(f"unsupported effect mode: {value}")
    return cast(EffectMode, mode)


def parse_color_bgr(value: str) -> tuple[int, int, int]:
    normalized = value.strip().lower()
    if normalized in COLOR_NAMES_BGR:
        return COLOR_NAMES_BGR[normalized]

    hex_value = normalized.removeprefix("#")
    if len(hex_value) != 6:
        raise ValueError("color must be #RRGGBB or a basic color name")
    try:
        red = int(hex_value[0:2], 16)
        green = int(hex_value[2:4], 16)
        blue = int(hex_value[4:6], 16)
    except ValueError as exc:
        raise ValueError("color must be #RRGGBB or a basic color name") from exc
    return (blue, green, red)


def format_color_hex(color_bgr: tuple[int, int, int]) -> str:
    blue, green, red = (min(max(int(channel), 0), 255) for channel in color_bgr)
    return f"#{red:02x}{green:02x}{blue:02x}"


def apply_polygon_effect(
    frame_bgr: np.ndarray,
    normalized_points: Sequence[Point] | None,
    config: EffectConfig,
    *,
    animation_phase: float = 0.0,
) -> np.ndarray:
    """Apply the selected effect only inside the polygon."""
    if normalized_points is None:
        return frame_bgr.copy()

    height, width = frame_bgr.shape[:2]
    polygon = normalized_to_pixels(normalized_points, width, height)
    if config.mode == "outline":
        output = frame_bgr.copy()
        thickness = config.outline_thickness
        if thickness <= 0:
            thickness = max(2, min(width, height) // 160)
        cv2.polylines(
            output,
            [polygon],
            True,
            config.outline_color_bgr,
            thickness,
            cv2.LINE_8,
        )
        return output

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    effected = _apply_effect(frame_bgr, config, animation_phase)

    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    output = (
        frame_bgr.astype(np.float32) * (1.0 - alpha)
        + effected.astype(np.float32) * alpha
    )
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def _apply_effect(
    frame_bgr: np.ndarray,
    config: EffectConfig,
    animation_phase: float = 0.0,
) -> np.ndarray:
    if config.mode == "blur":
        kernel_size = _odd_at_least_three(config.kernel_size)
        return cv2.GaussianBlur(
            frame_bgr,
            (kernel_size, kernel_size),
            0,
        )

    if config.mode == "mosaic":
        height, width = frame_bgr.shape[:2]
        block_size = max(1, int(config.mosaic_block_size))
        small_width = max(1, width // block_size)
        small_height = max(1, height // block_size)
        small = cv2.resize(
            frame_bgr,
            (small_width, small_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

    if config.mode == "invert":
        return 255 - frame_bgr

    if config.mode == "grayscale":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if config.mode == "edge":
        edges = _canny_edges(frame_bgr, config)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if config.mode == "thermal":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        colormap = COLORMAPS.get(config.thermal_colormap, cv2.COLORMAP_JET)
        return cv2.applyColorMap(gray, colormap)

    if config.mode == "noise":
        height, width = frame_bgr.shape[:2]
        y_indices, x_indices = np.indices((height, width), dtype=np.uint16)
        noise = np.stack(
            [
                (x_indices * 37 + y_indices * 17) % 256,
                (x_indices * 11 + y_indices * 53 + 97) % 256,
                (x_indices * 71 + y_indices * 29 + 193) % 256,
            ],
            axis=2,
        ).astype(np.uint8)
        strength = min(max(float(config.noise_strength), 0.0), 1.0)
        return cv2.addWeighted(frame_bgr, 1.0 - strength, noise, strength, 0)

    if config.mode == "neon":
        return _apply_neon_effect(frame_bgr, config)

    if config.mode == "glitch":
        return _apply_glitch_effect(frame_bgr, config, animation_phase)

    if config.mode == "cartoon":
        return _apply_cartoon_effect(frame_bgr)

    if config.mode == "sketch":
        return _apply_sketch_effect(frame_bgr)

    raise ValueError(f"unsupported effect mode: {config.mode}")


def _canny_edges(frame_bgr: np.ndarray, config: EffectConfig) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    low, high = sorted(
        (
            max(0.0, float(config.edge_low_threshold)),
            max(0.0, float(config.edge_high_threshold)),
        )
    )
    return cv2.Canny(smoothed, low, high)


def _apply_neon_effect(frame_bgr: np.ndarray, config: EffectConfig) -> np.ndarray:
    edges = _canny_edges(frame_bgr, config)
    glow = cv2.GaussianBlur(edges, (0, 0), 5.0)
    color = config.outline_color_bgr
    if color == (0, 0, 0):
        color = (255, 255, 0)
    strength = min(max(float(config.noise_strength), 0.0), 1.0)
    color_layer = np.zeros_like(frame_bgr, dtype=np.float32)
    for channel, value in enumerate(color):
        color_layer[:, :, channel] = glow.astype(np.float32) * (value / 255.0)
    dimmed = frame_bgr.astype(np.float32) * (1.0 - 0.45 * strength)
    output = dimmed + color_layer * (1.8 * strength)
    edge_color = np.zeros_like(frame_bgr, dtype=np.uint8)
    edge_color[edges > 0] = color
    output = cv2.addWeighted(
        np.clip(output, 0, 255).astype(np.uint8),
        1.0,
        edge_color,
        strength,
        0,
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def _apply_glitch_effect(
    frame_bgr: np.ndarray,
    config: EffectConfig,
    animation_phase: float,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    strength = min(max(float(config.noise_strength), 0.0), 1.0)
    shift = max(1, int(width * (0.01 + 0.03 * strength)))
    phase = int(animation_phase)
    blue, green, red = cv2.split(frame_bgr)
    red = np.roll(red, shift + phase % max(shift, 1), axis=1)
    blue = np.roll(blue, -(shift + (phase * 2) % max(shift, 1)), axis=1)
    output = cv2.merge((blue, green, red))

    block_height = max(2, height // 18)
    for index, y in enumerate(range(0, height, block_height * 2)):
        offset = int(np.sin(index * 1.7 + phase * 0.13) * shift * 2)
        end_y = min(height, y + block_height)
        output[y:end_y] = np.roll(output[y:end_y], offset, axis=1)

    output[::4] = (output[::4].astype(np.float32) * (0.35 + 0.35 * strength)).astype(np.uint8)
    return output


def _apply_cartoon_effect(frame_bgr: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "stylization"):
        return cv2.stylization(frame_bgr, sigma_s=60, sigma_r=0.45)

    color = cv2.bilateralFilter(frame_bgr, 9, 90, 90)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 7),
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        2,
    )
    return cv2.bitwise_and(color, color, mask=edges)


def _apply_sketch_effect(frame_bgr: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "pencilSketch"):
        sketch_gray, _ = cv2.pencilSketch(
            frame_bgr,
            sigma_s=60,
            sigma_r=0.07,
            shade_factor=0.045,
        )
        return cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def _odd_at_least_three(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1
