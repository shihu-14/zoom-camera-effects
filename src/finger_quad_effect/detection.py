"""Strict hand-point activation logic."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Sequence

from .geometry import (
    PlaneEquation,
    Point,
    Point3D,
    clamp_normalized_points,
    estimate_plane_equation,
    normalized_points_in_bounds,
    sort_quad_vertices_3d,
)

THUMB_TIP = 4
INDEX_TIP = 8
MIN_TWO_POINT_DIAGONAL = 0.18


@dataclass(frozen=True)
class LandmarkPoint:
    x: float
    y: float
    z: float = 0.0
    presence: float | None = None
    visibility: float | None = None


@dataclass(frozen=True)
class RawHand:
    landmarks: Sequence[LandmarkPoint]
    score: float
    label: str | None = None


@dataclass(frozen=True)
class DetectionConfig:
    min_hand_score: float = 0.55
    min_point_score: float = 0.5
    point_bounds_margin: float = 0.12
    require_distinct_handedness: bool = False


@dataclass(frozen=True)
class DetectionResult:
    active: bool
    points: tuple[Point, Point, Point, Point] | None = None
    reason: str = "inactive"
    points_3d: tuple[Point3D, Point3D, Point3D, Point3D] | None = None
    plane: PlaneEquation | None = None


def build_quad_detection(
    hands: Sequence[RawHand],
    config: DetectionConfig,
) -> DetectionResult:
    """Build a strict four-point detection result from raw hand landmarks."""
    if len(hands) == 1:
        if config.require_distinct_handedness:
            return DetectionResult(False, reason="requires left and right hands")
        return _build_two_point_detection(hands[0], config)

    if len(hands) != 2:
        return DetectionResult(False, reason="requires exactly two hands")

    if config.require_distinct_handedness:
        labels = [hand.label for hand in hands]
        if any(label is None for label in labels):
            return DetectionResult(False, reason="requires handedness labels")
        if labels[0] == labels[1]:
            return DetectionResult(False, reason="requires left and right hands")

    raw_points: list[Point] = []
    raw_points_3d: list[Point3D] = []
    for hand in hands:
        points, points_3d, reason = _required_fingertips(hand, config)
        if reason is not None:
            return DetectionResult(False, reason=reason)
        raw_points.extend(points)
        raw_points_3d.extend(points_3d)

    if not normalized_points_in_bounds(raw_points, margin=config.point_bounds_margin):
        return DetectionResult(False, reason="required fingertips outside frame")

    try:
        clamped_points = clamp_normalized_points(raw_points)
        clamped_points_3d = tuple(
            (point[0], point[1], raw_points_3d[index][2])
            for index, point in enumerate(clamped_points)
        )
        ordered_3d = sort_quad_vertices_3d(clamped_points_3d)
        ordered = tuple((x, y) for x, y, _ in ordered_3d)
    except ValueError as exc:
        return DetectionResult(False, reason=str(exc))

    try:
        plane = estimate_plane_equation(ordered_3d)
    except ValueError:
        plane = None

    return DetectionResult(
        True,
        points=ordered,
        reason="active",
        points_3d=ordered_3d,
        plane=plane,
    )


def _build_two_point_detection(
    hand: RawHand,
    config: DetectionConfig,
) -> DetectionResult:
    points, points_3d, reason = _required_fingertips(hand, config)
    if reason is not None:
        return DetectionResult(False, reason=reason)

    if not normalized_points_in_bounds(points, margin=config.point_bounds_margin):
        return DetectionResult(False, reason="required fingertips outside frame")

    inferred_points = _infer_quad_from_two_points(points[0], points[1])
    inferred_points_3d = _infer_quad_3d_from_two_points(points_3d[0], points_3d[1])

    try:
        clamped_points = clamp_normalized_points(inferred_points)
        clamped_points_3d = tuple(
            (point[0], point[1], inferred_points_3d[index][2])
            for index, point in enumerate(clamped_points)
        )
        ordered_3d = sort_quad_vertices_3d(clamped_points_3d)
        ordered = tuple((x, y) for x, y, _ in ordered_3d)
    except ValueError as exc:
        return DetectionResult(False, reason=str(exc))

    try:
        plane = estimate_plane_equation(ordered_3d)
    except ValueError:
        plane = None

    return DetectionResult(
        True,
        points=ordered,
        reason="active: inferred from one hand",
        points_3d=ordered_3d,
        plane=plane,
    )


def _required_fingertips(
    hand: RawHand,
    config: DetectionConfig,
) -> tuple[list[Point], list[Point3D], str | None]:
    if hand.score < config.min_hand_score:
        return [], [], "hand confidence too low"
    if len(hand.landmarks) <= INDEX_TIP:
        return [], [], "missing required landmarks"

    points: list[Point] = []
    points_3d: list[Point3D] = []
    for index in (THUMB_TIP, INDEX_TIP):
        landmark = hand.landmarks[index]
        if not _point_confident(landmark, config.min_point_score):
            return [], [], "point confidence too low"
        points.append((landmark.x, landmark.y))
        points_3d.append((landmark.x, landmark.y, landmark.z))
    return points, points_3d, None


def _infer_quad_from_two_points(
    first: Point,
    second: Point,
) -> tuple[Point, Point, Point, Point]:
    center_x = (first[0] + second[0]) * 0.5
    center_y = (first[1] + second[1]) * 0.5
    half_x = (second[0] - first[0]) * 0.5
    half_y = (second[1] - first[1]) * 0.5
    half_diagonal = hypot(half_x, half_y)
    minimum_half_diagonal = MIN_TWO_POINT_DIAGONAL * 0.5
    if half_diagonal < 1e-6:
        half_x = minimum_half_diagonal
        half_y = 0.0
    elif half_diagonal < minimum_half_diagonal:
        scale = minimum_half_diagonal / half_diagonal
        half_x *= scale
        half_y *= scale

    perpendicular_x = -half_y
    perpendicular_y = half_x
    return (
        (center_x - half_x, center_y - half_y),
        (center_x - perpendicular_x, center_y - perpendicular_y),
        (center_x + half_x, center_y + half_y),
        (center_x + perpendicular_x, center_y + perpendicular_y),
    )


def _infer_quad_3d_from_two_points(
    first: Point3D,
    second: Point3D,
) -> tuple[Point3D, Point3D, Point3D, Point3D]:
    z = (first[2] + second[2]) * 0.5
    first_2d = (first[0], first[1])
    second_2d = (second[0], second[1])
    inferred = _infer_quad_from_two_points(first_2d, second_2d)
    return tuple((x, y, z) for x, y in inferred)  # type: ignore[return-value]


def _point_confident(point: LandmarkPoint, minimum: float) -> bool:
    for score in (point.presence, point.visibility):
        if score is not None and score < minimum:
            return False
    return True
