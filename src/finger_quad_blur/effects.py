"""Video effects for polygon-localized processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import cv2
import numpy as np

from .geometry import PlaneEquation, Point, normalized_to_pixels

EffectMode = Literal[
    "blur",
    "mosaic",
    "invert",
    "grayscale",
    "edge",
    "thermal",
    "noise",
    "outline",
    "particles",
]


@dataclass(frozen=True)
class BlurConfig:
    mode: EffectMode = "blur"
    kernel_size: int = 35
    edge_feather_px: int = 3
    mosaic_block_size: int = 18


def apply_polygon_blur(
    frame_bgr: np.ndarray,
    normalized_points: Sequence[Point] | None,
    config: BlurConfig,
    *,
    plane: PlaneEquation | None = None,
    animation_phase: float = 0.0,
) -> np.ndarray:
    """Apply the selected effect only inside the polygon."""
    if normalized_points is None:
        return frame_bgr.copy()

    height, width = frame_bgr.shape[:2]
    polygon = normalized_to_pixels(normalized_points, width, height)
    if config.mode == "outline":
        output = frame_bgr.copy()
        thickness = max(2, min(width, height) // 160)
        cv2.polylines(output, [polygon], True, (0, 0, 0), thickness, cv2.LINE_8)
        return output

    if config.mode == "particles":
        return _apply_particle_effect(
            frame_bgr,
            polygon.reshape((-1, 2)).astype(np.float32),
            plane,
            animation_phase,
        )

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    effected = _apply_effect(frame_bgr, config)

    if config.edge_feather_px > 0:
        feather_size = _odd_at_least_three(config.edge_feather_px * 2 + 1)
        mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0)

    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    output = (
        frame_bgr.astype(np.float32) * (1.0 - alpha)
        + effected.astype(np.float32) * alpha
    )
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def _apply_effect(frame_bgr: np.ndarray, config: BlurConfig) -> np.ndarray:
    if config.mode == "blur":
        kernel_size = _odd_at_least_three(config.kernel_size)
        return cv2.GaussianBlur(frame_bgr, (kernel_size, kernel_size), 0)

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
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(smoothed, 60, 140)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if config.mode == "thermal":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

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
        return cv2.addWeighted(frame_bgr, 0.2, noise, 0.8, 0)

    raise ValueError(f"unsupported effect mode: {config.mode}")


def _apply_particle_effect(
    frame_bgr: np.ndarray,
    polygon: np.ndarray,
    plane: PlaneEquation | None,
    animation_phase: float,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    output = frame_bgr.copy()
    overlay = np.zeros_like(frame_bgr, dtype=np.float32)
    center = polygon.mean(axis=0)
    radius = max(float(np.linalg.norm(polygon - center, axis=1).mean()), 1.0)

    normal_x = 0.0
    normal_y = 0.0
    if plane is not None:
        normal_x, normal_y, _ = plane.normal
    normal_length = max((normal_x * normal_x + normal_y * normal_y) ** 0.5, 1e-6)
    direction_x = normal_x / normal_length
    direction_y = normal_y / normal_length
    tilt = min(normal_length, 1.0)
    if tilt < 0.08:
        direction_x = 0.0
        direction_y = -1.0

    top_left, bottom_left, bottom_right, top_right = polygon
    phase = animation_phase * 0.035
    particle_count = 90
    travel_limit = radius * (0.85 + 0.65 * max(tilt, 0.35))
    cv2.polylines(output, [polygon.astype(np.int32).reshape((-1, 1, 2))], True, (130, 40, 210), 1, cv2.LINE_AA)

    for index in range(particle_count):
        seed = float(index + 1)
        u = (seed * 0.61803398875 + 0.17) % 1.0
        v = (seed * 0.41421356237 + 0.31) % 1.0
        offset = (seed * 0.75487766625) % 1.0
        progress = (phase * (0.72 + (seed % 7.0) * 0.035) + offset) % 1.0

        top = top_left * (1.0 - u) + top_right * u
        bottom = bottom_left * (1.0 - u) + bottom_right * u
        origin = top * (1.0 - v) + bottom * v
        radial = origin - center
        radial_length = max(float(np.linalg.norm(radial)), 1e-6)
        radial /= radial_length

        travel = progress * travel_limit
        drift = np.array([direction_x, direction_y], dtype=np.float32) * travel
        spread = radial * travel * (0.18 + 0.2 * (1.0 - tilt))
        wobble_angle = seed * 1.91 + animation_phase * 0.09
        wobble = np.array([np.cos(wobble_angle), np.sin(wobble_angle)], dtype=np.float32)
        position = origin + drift + spread + wobble * radius * 0.018

        x = int(round(float(position[0])))
        y = int(round(float(position[1])))
        if not (0 <= x < width and 0 <= y < height):
            continue

        size = int(1 + (1.0 - progress) * 4.0)
        alpha = float(np.sin(progress * np.pi) * (1.0 - progress * 0.35))
        color = np.array(
            [
                180.0 + 55.0 * (1.0 - progress),
                220.0 + 30.0 * np.sin(seed),
                255.0,
            ],
            dtype=np.float32,
        )

        trail_position = position - np.array([direction_x, direction_y], dtype=np.float32) * max(size * 3.0, travel * 0.08)
        cv2.line(
            overlay,
            tuple(np.rint(trail_position).astype(int)),
            (x, y),
            tuple(float(channel * alpha * 0.45) for channel in color),
            max(1, size),
            cv2.LINE_AA,
        )
        cv2.circle(
            overlay,
            (x, y),
            size,
            tuple(float(channel * alpha) for channel in color),
            -1,
            cv2.LINE_AA,
        )

    combined = output.astype(np.float32) + overlay
    return np.clip(combined, 0, 255).astype(np.uint8)


def _odd_at_least_three(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1
