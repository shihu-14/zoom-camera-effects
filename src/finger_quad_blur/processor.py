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
    def __init__(self, detector: Detector, blur_config: BlurConfig) -> None:
        self._detector = detector
        self._blur_config = blur_config

    def process(self, frame_bgr: np.ndarray) -> ProcessedFrame:
        detection = self._detector.detect(frame_bgr)
        processed = apply_polygon_blur(
            frame_bgr,
            detection.points if detection.active else None,
            self._blur_config,
        )
        return ProcessedFrame(frame_bgr=processed, detection=detection)
