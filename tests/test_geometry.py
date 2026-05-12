from itertools import permutations

from finger_quad_blur.geometry import (
    normalized_points_in_bounds,
    polygon_area,
    sort_quad_vertices,
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
