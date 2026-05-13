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
    "portal",
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

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    if config.mode == "portal":
        effected = _apply_portal_effect(
            frame_bgr,
            polygon.reshape((-1, 2)).astype(np.float32),
            plane,
            animation_phase,
        )
    else:
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
    if config.mode == "portal":
        cv2.polylines(output, [polygon], True, (190, 25, 255), 2, cv2.LINE_AA)
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


def _apply_portal_effect(
    frame_bgr: np.ndarray,
    polygon: np.ndarray,
    plane: PlaneEquation | None,
    animation_phase: float,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    y_indices, x_indices = np.indices((height, width), dtype=np.float32)
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

    dx = (x_indices - center[0]) / radius
    dy = (y_indices - center[1]) / radius
    dx -= direction_x * tilt * 0.18
    dy -= direction_y * tilt * 0.18

    distance = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)
    direction = dx * direction_x + dy * direction_y
    cross = dx * direction_y - dy * direction_x

    phase = animation_phase * 0.18
    swirl = 0.5 + 0.5 * np.sin(angle * 11.0 + distance * 17.0 - phase * 3.4)
    streaks = 0.5 + 0.5 * np.sin(angle * 23.0 - distance * 13.0 + phase * 5.2)
    rings = 0.5 + 0.5 * np.sin(distance * 34.0 - phase * 7.0)
    core = np.exp(-(distance * 2.6) ** 2)
    rim = np.exp(-((distance - 0.92) ** 2) / 0.018)
    jet = (
        np.clip(direction + 0.25, 0.0, 1.0)
        * np.exp(-(cross * 2.8) ** 2)
        * np.exp(-distance * 0.9)
        * tilt
    )
    energy = np.clip(swirl * 0.55 + streaks * rings * 0.35 + rim * 0.8 + jet, 0.0, 1.0)

    portal = np.zeros_like(frame_bgr, dtype=np.float32)
    portal[:, :, 0] = 50.0 + 135.0 * energy + 80.0 * rim + 180.0 * core
    portal[:, :, 1] = 8.0 + 28.0 * energy + 220.0 * core
    portal[:, :, 2] = 70.0 + 185.0 * energy + 90.0 * rim + 180.0 * core

    darkness = np.clip(distance - 0.25, 0.0, 1.0)
    portal *= 1.12 - darkness[:, :, None] * 0.28
    return np.clip(portal, 0, 255).astype(np.uint8)


def _odd_at_least_three(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1
