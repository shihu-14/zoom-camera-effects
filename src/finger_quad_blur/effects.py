"""Video effects for polygon-localized blur."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .geometry import Point, normalized_to_pixels


@dataclass(frozen=True)
class BlurConfig:
    kernel_size: int = 35
    edge_feather_px: int = 3


def apply_polygon_blur(
    frame_bgr: np.ndarray,
    normalized_points: Sequence[Point] | None,
    config: BlurConfig,
) -> np.ndarray:
    """Blur only the polygon interior; return the original frame when inactive."""
    if normalized_points is None:
        return frame_bgr.copy()

    height, width = frame_bgr.shape[:2]
    polygon = normalized_to_pixels(normalized_points, width, height)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    kernel_size = _odd_at_least_three(config.kernel_size)
    blurred = cv2.GaussianBlur(frame_bgr, (kernel_size, kernel_size), 0)

    if config.edge_feather_px > 0:
        feather_size = _odd_at_least_three(config.edge_feather_px * 2 + 1)
        mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0)

    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    output = frame_bgr.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def _odd_at_least_three(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1
