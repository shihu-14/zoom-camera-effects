"""MediaPipe adapter for fingertip quadrilateral detection."""

from __future__ import annotations

from typing import Any

import cv2

from .detection import DetectionConfig, DetectionResult, LandmarkPoint, RawHand, build_quad_detection


class HandPointDetector:
    """Detect both thumbs and index fingertips using MediaPipe Hands."""

    def __init__(
        self,
        detection_config: DetectionConfig,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence: float = 0.75,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required for webcam hand tracking. "
                "Install dependencies with `python3 -m pip install -e .`."
            ) from exc

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
            _to_raw_hand(hand_landmarks, handedness[index] if index < len(handedness) else None)
            for index, hand_landmarks in enumerate(landmarks)
        ]
        return build_quad_detection(hands, self._detection_config)

    def __enter__(self) -> "HandPointDetector":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _to_raw_hand(hand_landmarks: Any, handedness: Any | None) -> RawHand:
    classification = None
    if handedness and getattr(handedness, "classification", None):
        classification = handedness.classification[0]

    score = float(getattr(classification, "score", 1.0)) if classification else 1.0
    label = str(getattr(classification, "label", "")) if classification else None
    landmarks = [
        LandmarkPoint(
            x=float(landmark.x),
            y=float(landmark.y),
            presence=_optional_float(getattr(landmark, "presence", None)),
            visibility=_optional_float(getattr(landmark, "visibility", None)),
        )
        for landmark in hand_landmarks.landmark
    ]
    return RawHand(landmarks=landmarks, score=score, label=label or None)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
