"""MediaPipe adapter for fingertip quadrilateral detection."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import cv2

from .detection import (
    DetectionConfig,
    DetectionResult,
    LandmarkPoint,
    RawHand,
    build_quad_detection,
)


class HandPointDetector:
    """Detect both thumbs and index fingertips using MediaPipe Hands."""

    def __init__(
        self,
        detection_config: DetectionConfig,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence: float = 0.75,
    ) -> None:
        mp = _import_mediapipe()
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            raise RuntimeError(
                "This app requires the MediaPipe solutions hand tracker. "
                "Install the pinned dependencies with `python3 -m pip install -e .`."
            )

        self._detection_config = detection_config
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def detect(self, frame_bgr) -> DetectionResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        landmarks = results.multi_hand_landmarks or []
        handedness = results.multi_handedness or []
        hands = [
            _solutions_raw_hand(
                hand_landmarks,
                handedness[index] if index < len(handedness) else None,
            )
            for index, hand_landmarks in enumerate(landmarks)
        ]
        return build_quad_detection(hands, self._detection_config)

    def __enter__(self) -> "HandPointDetector":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _solutions_raw_hand(hand_landmarks: Any, handedness: Any | None) -> RawHand:
    classification = None
    if handedness and getattr(handedness, "classification", None):
        classification = handedness.classification[0]

    score = float(getattr(classification, "score", 0.0)) if classification else 0.0
    label = str(getattr(classification, "label", "")) if classification else None
    landmarks = [_landmark_point(landmark) for landmark in hand_landmarks.landmark]
    return RawHand(landmarks=landmarks, score=score, label=label or None)


def _landmark_point(landmark: Any) -> LandmarkPoint:
    return LandmarkPoint(
        x=float(landmark.x),
        y=float(landmark.y),
        z=float(getattr(landmark, "z", 0.0)),
        presence=_optional_landmark_score(landmark, "presence"),
        visibility=_optional_landmark_score(landmark, "visibility"),
    )


def _optional_landmark_score(landmark: Any, field_name: str) -> float | None:
    if hasattr(landmark, "HasField"):
        try:
            if not landmark.HasField(field_name):
                return None
        except ValueError:
            return None
    value = getattr(landmark, field_name, None)
    return None if value is None else float(value)


def _import_mediapipe() -> Any:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "finger_quad_effect_matplotlib"),
    )
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required for webcam hand tracking. "
            "Install dependencies with `python3 -m pip install -e .`."
        ) from exc
    return mp
