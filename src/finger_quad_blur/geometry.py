"""Geometry helpers for stable quadrilateral masks."""

from __future__ import annotations

from math import atan2, isfinite
from typing import Iterable, Sequence

import numpy as np

Point = tuple[float, float]


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


def normalized_points_in_bounds(points: Sequence[Point], margin: float = 0.0) -> bool:
    """Return true when every point is inside the normalized frame."""
    lower = 0.0 - margin
    upper = 1.0 + margin
    return all(lower <= x <= upper and lower <= y <= upper for x, y in points)


def normalized_to_pixels(points: Sequence[Point], width: int, height: int) -> np.ndarray:
    """Convert normalized points to integer pixel coordinates for OpenCV."""
    coords = np.asarray(points, dtype=np.float32)
    scale = np.asarray([max(width - 1, 0), max(height - 1, 0)], dtype=np.float32)
    pixels = np.rint(coords * scale).astype(np.int32)
    return pixels.reshape((-1, 1, 2))
