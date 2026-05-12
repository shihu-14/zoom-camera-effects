"""Strict hand-point activation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .geometry import (
    Point,
    clamp_normalized_points,
    normalized_points_in_bounds,
    sort_quad_vertices,
)

THUMB_TIP = 4
INDEX_TIP = 8


@dataclass(frozen=True)
class LandmarkPoint:
    x: float
    y: float
    presence: float | None = None
    visibility: float | None = None


@dataclass(frozen=True)
class RawHand:
    landmarks: Sequence[LandmarkPoint]
    score: float
    label: str | None = None


@dataclass(frozen=True)
class DetectionConfig:
    min_hand_score: float = 0.75
    min_point_score: float = 0.5
    point_bounds_margin: float = 0.08
    require_distinct_handedness: bool = True


@dataclass(frozen=True)
class DetectionResult:
    active: bool
    points: tuple[Point, Point, Point, Point] | None = None
    reason: str = "inactive"


def build_quad_detection(
    hands: Sequence[RawHand],
    config: DetectionConfig,
) -> DetectionResult:
    """Build a strict four-point detection result from raw hand landmarks."""
    if len(hands) != 2:
        return DetectionResult(False, reason="requires exactly two hands")

    if config.require_distinct_handedness:
        labels = [hand.label for hand in hands]
        if any(label is None for label in labels):
            return DetectionResult(False, reason="requires handedness labels")
        if labels[0] == labels[1]:
            return DetectionResult(False, reason="requires left and right hands")

    raw_points: list[Point] = []
    for hand in hands:
        if hand.score < config.min_hand_score:
            return DetectionResult(False, reason="hand confidence too low")
        if len(hand.landmarks) <= INDEX_TIP:
            return DetectionResult(False, reason="missing required landmarks")

        for index in (THUMB_TIP, INDEX_TIP):
            landmark = hand.landmarks[index]
            if not _point_confident(landmark, config.min_point_score):
                return DetectionResult(False, reason="point confidence too low")
            raw_points.append((landmark.x, landmark.y))

    if not normalized_points_in_bounds(raw_points, margin=config.point_bounds_margin):
        return DetectionResult(False, reason="required fingertips outside frame")

    try:
        ordered = sort_quad_vertices(clamp_normalized_points(raw_points))
    except ValueError as exc:
        return DetectionResult(False, reason=str(exc))

    return DetectionResult(True, points=ordered, reason="active")


def _point_confident(point: LandmarkPoint, minimum: float) -> bool:
    for score in (point.presence, point.visibility):
        if score is not None and score < minimum:
            return False
    return True
