"""Frame-level processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .detection import DetectionResult
from .effects import BlurConfig, apply_polygon_blur


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        """Return the current fingertip quadrilateral state."""


@dataclass(frozen=True)
class ProcessedFrame:
    frame_bgr: np.ndarray
    detection: DetectionResult


class FrameProcessor:
    def __init__(
        self,
        detector: Detector,
        blur_config: BlurConfig,
        smoothing_factor: float = 0.35,
    ) -> None:
        self._detector = detector
        self._blur_config = blur_config
        self._smoothing_factor = min(max(float(smoothing_factor), 0.0), 1.0)
        self._previous_points: tuple[tuple[float, float], ...] | None = None

    def process(self, frame_bgr: np.ndarray) -> ProcessedFrame:
        detection = self._detector.detect(frame_bgr)
        detection = self._smooth_detection(detection)
        processed = apply_polygon_blur(
            frame_bgr,
            detection.points if detection.active else None,
            self._blur_config,
        )
        return ProcessedFrame(frame_bgr=processed, detection=detection)

    def _smooth_detection(self, detection: DetectionResult) -> DetectionResult:
        if not detection.active or detection.points is None:
            self._previous_points = None
            return detection

        if self._previous_points is None or self._smoothing_factor <= 0.0:
            self._previous_points = detection.points
            return detection

        alpha = self._smoothing_factor
        smoothed = tuple(
            (
                previous[0] * (1.0 - alpha) + current[0] * alpha,
                previous[1] * (1.0 - alpha) + current[1] * alpha,
            )
            for previous, current in zip(self._previous_points, detection.points)
        )
        self._previous_points = smoothed
        return DetectionResult(True, points=smoothed, reason=detection.reason)
