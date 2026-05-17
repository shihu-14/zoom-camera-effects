"""Frame-level processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .detection import DetectionResult
from .effects import EffectConfig, apply_polygon_effect
from .geometry import Point3D, estimate_plane_equation


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
        effect_config: EffectConfig,
        smoothing_factor: float = 0.35,
    ) -> None:
        self._detector = detector
        self._effect_config = effect_config
        self._smoothing_factor = min(max(float(smoothing_factor), 0.0), 1.0)
        self._previous_points: tuple[tuple[float, float], ...] | None = None
        self._previous_points_3d: tuple[Point3D, ...] | None = None
        self._frame_index = 0

    @property
    def effect_config(self) -> EffectConfig:
        return self._effect_config

    def set_effect_config(self, effect_config: EffectConfig) -> None:
        self._effect_config = effect_config

    def process(self, frame_bgr: np.ndarray) -> ProcessedFrame:
        detection = self._detector.detect(frame_bgr)
        detection = self._smooth_detection(detection)
        self._frame_index += 1
        processed = apply_polygon_effect(
            frame_bgr,
            detection.points if detection.active else None,
            self._effect_config,
            animation_phase=float(self._frame_index),
        )
        return ProcessedFrame(frame_bgr=processed, detection=detection)

    def _smooth_detection(self, detection: DetectionResult) -> DetectionResult:
        if not detection.active or detection.points is None:
            self._previous_points = None
            self._previous_points_3d = None
            return detection

        if self._previous_points is None or self._smoothing_factor <= 0.0:
            self._previous_points = detection.points
            self._previous_points_3d = detection.points_3d
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

        if self._previous_points_3d is None or detection.points_3d is None:
            self._previous_points_3d = detection.points_3d
            return DetectionResult(
                True,
                points=smoothed,
                reason=detection.reason,
                points_3d=detection.points_3d,
                plane=detection.plane,
            )

        smoothed_3d = tuple(
            (
                previous[0] * (1.0 - alpha) + current[0] * alpha,
                previous[1] * (1.0 - alpha) + current[1] * alpha,
                previous[2] * (1.0 - alpha) + current[2] * alpha,
            )
            for previous, current in zip(self._previous_points_3d, detection.points_3d)
        )
        self._previous_points_3d = smoothed_3d
        try:
            plane = estimate_plane_equation(smoothed_3d)
        except ValueError:
            plane = None
        return DetectionResult(
            True,
            points=tuple((x, y) for x, y, _ in smoothed_3d),
            reason=detection.reason,
            points_3d=smoothed_3d,
            plane=plane,
        )
