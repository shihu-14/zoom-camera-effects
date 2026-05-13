"""Video effects for polygon-localized processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import cv2
import numpy as np

from .geometry import Point, normalized_to_pixels

EffectMode = Literal["blur", "mosaic", "invert", "grayscale"]


@dataclass(frozen=True)
class BlurConfig:
    mode: EffectMode = "invert"
    kernel_size: int = 35
    edge_feather_px: int = 3
    mosaic_block_size: int = 18


def apply_polygon_blur(
    frame_bgr: np.ndarray,
    normalized_points: Sequence[Point] | None,
    config: BlurConfig,
) -> np.ndarray:
    """Apply the selected effect only inside the polygon."""
    if normalized_points is None:
        return frame_bgr.copy()

    height, width = frame_bgr.shape[:2]
    polygon = normalized_to_pixels(normalized_points, width, height)
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
    return np.clip(output, 0, 255).astype(np.uint8)


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

    raise ValueError(f"unsupported effect mode: {config.mode}")


def _odd_at_least_three(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1
