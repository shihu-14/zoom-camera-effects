from itertools import permutations

from finger_quad_effect.geometry import (
    estimate_plane_equation,
    normalized_points_in_bounds,
    polygon_area,
    sort_quad_vertices,
    sort_quad_vertices_3d,
)


def test_sort_quad_vertices_is_stable_for_permutations():
    points = ((0.8, 0.2), (0.2, 0.7), (0.8, 0.7), (0.2, 0.2))
    expected = sort_quad_vertices(points)

    for shuffled in permutations(points):
        assert sort_quad_vertices(shuffled) == expected


def test_sort_quad_vertices_starts_at_top_left():
    result = sort_quad_vertices(((0.8, 0.2), (0.2, 0.7), (0.8, 0.7), (0.2, 0.2)))

    assert result[0] == (0.2, 0.2)
    assert polygon_area(result) > 0.0


def test_normalized_points_in_bounds():
    assert normalized_points_in_bounds(((0.0, 0.0), (1.0, 1.0)))
    assert not normalized_points_in_bounds(((-0.01, 0.0), (1.0, 1.0)))


def test_sort_quad_vertices_3d_preserves_depth_order():
    points = ((0.8, 0.2, 1.0), (0.2, 0.7, 2.0), (0.8, 0.7, 3.0), (0.2, 0.2, 4.0))

    result = sort_quad_vertices_3d(points)

    assert result == ((0.2, 0.2, 4.0), (0.2, 0.7, 2.0), (0.8, 0.7, 3.0), (0.8, 0.2, 1.0))


def test_estimate_plane_equation_returns_oriented_unit_normal():
    points = (
        (0.0, 0.0, 0.10),
        (0.0, 1.0, 0.35),
        (1.0, 1.0, 0.85),
        (1.0, 0.0, 0.60),
    )

    plane = estimate_plane_equation(points)

    assert plane.normal[0] > 0.0
    assert plane.normal[1] > 0.0
    assert plane.normal[2] < 0.0
    assert abs(sum(component * component for component in plane.normal) - 1.0) < 1e-12
    assert plane.residual < 1e-12
    for x, y, z in points:
        assert abs(plane.normal[0] * x + plane.normal[1] * y + plane.normal[2] * z + plane.offset) < 1e-12
