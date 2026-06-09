"""Geometry helpers for stable quadrilateral masks."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite
from typing import Iterable, Sequence

import numpy as np

Point = tuple[float, float]
Point3D = tuple[float, float, float]


@dataclass(frozen=True)
class PlaneEquation:
    normal: Point3D
    offset: float
    residual: float


def polygon_area(points: Sequence[Point]) -> float:
    """Return absolute polygon area in normalized or pixel coordinates."""
    if len(points) < 3:
        return 0.0

    coords = np.asarray(points, dtype=np.float64)
    x_values = coords[:, 0]
    y_values = coords[:, 1]
    signed = np.dot(x_values, np.roll(y_values, -1)) - np.dot(
        y_values, np.roll(x_values, -1)
    )
    return float(abs(signed) * 0.5)


def sort_quad_vertices(points: Iterable[Point]) -> tuple[Point, Point, Point, Point]:
    """Sort four points into a stable, non-crossing polygon order.

    The first point is the top-left-most vertex, followed clockwise in image
    coordinates. This keeps the OpenCV mask stable when hands swap positions or
    move vertically.
    """
    point_list = tuple((float(x), float(y)) for x, y in points)
    if len(point_list) != 4:
        raise ValueError("exactly four points are required")
    if any(not (isfinite(x) and isfinite(y)) for x, y in point_list):
        raise ValueError("points must be finite")

    centroid_x = sum(x for x, _ in point_list) / 4.0
    centroid_y = sum(y for _, y in point_list) / 4.0

    # Image coordinates have y pointing down. Sorting by atan2 around the
    # centroid gives a simple polygon; reversing makes the order clockwise.
    ordered = sorted(
        point_list,
        key=lambda point: atan2(point[1] - centroid_y, point[0] - centroid_x),
    )
    ordered.reverse()

    start_index = min(
        range(4),
        key=lambda index: (ordered[index][0] + ordered[index][1], ordered[index][1]),
    )
    rotated = ordered[start_index:] + ordered[:start_index]
    return tuple(rotated)  # type: ignore[return-value]


def sort_quad_vertices_3d(
    points: Iterable[Point3D],
) -> tuple[Point3D, Point3D, Point3D, Point3D]:
    """Sort four 3D points by their image-plane coordinates."""
    point_list = tuple((float(x), float(y), float(z)) for x, y, z in points)
    if len(point_list) != 4:
        raise ValueError("exactly four points are required")

    ordered_2d = sort_quad_vertices((x, y) for x, y, _ in point_list)
    remaining = set(range(4))
    ordered_3d: list[Point3D] = []
    for target_x, target_y in ordered_2d:
        selected = min(
            remaining,
            key=lambda index: (
                (point_list[index][0] - target_x) ** 2
                + (point_list[index][1] - target_y) ** 2
            ),
        )
        remaining.remove(selected)
        ordered_3d.append(point_list[selected])

    return tuple(ordered_3d)  # type: ignore[return-value]


def estimate_plane_equation(points: Sequence[Point3D]) -> PlaneEquation:
    """Estimate the least-squares plane equation for 3D points."""
    if len(points) < 3:
        raise ValueError("at least three points are required")

    coords = np.asarray(points, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("3D points are required")
    if not np.isfinite(coords).all():
        raise ValueError("points must be finite")

    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if np.linalg.norm(normal) < 1e-12 or singular_values[1] < 1e-12:
        raise ValueError("points must not be collinear")

    winding_normal = _ordered_polygon_normal(coords)
    if np.linalg.norm(winding_normal) > 1e-12:
        if float(np.dot(normal, winding_normal)) < 0.0:
            normal = -normal
    elif normal[2] > 0.0:
        normal = -normal

    normal = normal / np.linalg.norm(normal)
    offset = -float(np.dot(normal, centroid))
    distances = coords @ normal + offset
    residual = float(np.sqrt(np.mean(distances * distances)))
    return PlaneEquation(
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        offset=offset,
        residual=residual,
    )


def _ordered_polygon_normal(coords: np.ndarray) -> np.ndarray:
    normal = np.zeros(3, dtype=np.float64)
    for index in range(len(coords)):
        current = coords[index]
        following = coords[(index + 1) % len(coords)]
        normal[0] += (current[1] - following[1]) * (current[2] + following[2])
        normal[1] += (current[2] - following[2]) * (current[0] + following[0])
        normal[2] += (current[0] - following[0]) * (current[1] + following[1])
    return normal


def normalized_points_in_bounds(points: Sequence[Point], margin: float = 0.0) -> bool:
    """Return true when every point is inside the normalized frame."""
    lower = 0.0 - margin
    upper = 1.0 + margin
    return all(lower <= x <= upper and lower <= y <= upper for x, y in points)


def clamp_normalized_points(points: Sequence[Point]) -> tuple[Point, ...]:
    """Clamp normalized points to the visible image area."""
    return tuple((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)) for x, y in points)


def normalized_to_pixels(points: Sequence[Point], width: int, height: int) -> np.ndarray:
    """Convert normalized points to integer pixel coordinates for OpenCV."""
    coords = np.asarray(points, dtype=np.float32)
    scale = np.asarray([max(width - 1, 0), max(height - 1, 0)], dtype=np.float32)
    pixels = np.rint(coords * scale).astype(np.int32)
    return pixels.reshape((-1, 1, 2))
