from finger_quad_effect.detection import (
    DetectionConfig,
    LandmarkPoint,
    RawHand,
    build_quad_detection,
)


def _hand(thumb, index, label="Left", score=0.95, presence=None, visibility=None):
    landmarks = [LandmarkPoint(0.5, 0.5) for _ in range(21)]
    landmarks[4] = LandmarkPoint(*thumb, presence=presence, visibility=visibility)
    landmarks[8] = LandmarkPoint(*index, presence=presence, visibility=visibility)
    return RawHand(landmarks=landmarks, score=score, label=label)


def test_detection_requires_two_hands_and_four_visible_points():
    result = build_quad_detection(
        [
            _hand((0.2, 0.7), (0.2, 0.2), label="Left"),
            _hand((0.8, 0.7), (0.8, 0.2), label="Right"),
        ],
        DetectionConfig(),
    )

    assert result.active is True
    assert result.points is not None
    assert len(result.points) == 4
    assert result.points_3d is not None
    assert len(result.points_3d) == 4
    assert result.plane is not None


def test_detection_falls_back_when_one_hand_is_missing():
    result = build_quad_detection(
        [_hand((0.2, 0.7), (0.2, 0.2), label="Left")],
        DetectionConfig(),
    )

    assert result.active is False
    assert result.reason == "requires exactly two hands"


def test_detection_falls_back_when_hand_confidence_is_low():
    result = build_quad_detection(
        [
            _hand((0.2, 0.7), (0.2, 0.2), label="Left", score=0.74),
            _hand((0.8, 0.7), (0.8, 0.2), label="Right"),
        ],
        DetectionConfig(min_hand_score=0.75),
    )

    assert result.active is False
    assert result.reason == "hand confidence too low"


def test_detection_falls_back_when_a_required_point_is_outside_frame():
    result = build_quad_detection(
        [
            _hand((-0.2, 0.7), (0.2, 0.2), label="Left"),
            _hand((0.8, 0.7), (0.8, 0.2), label="Right"),
        ],
        DetectionConfig(),
    )

    assert result.active is False
    assert result.reason == "required fingertips outside frame"


def test_detection_accepts_and_clamps_slight_edge_overshoot():
    result = build_quad_detection(
        [
            _hand((-0.03, 0.9), (-0.03, 0.1), label="Left"),
            _hand((1.03, 0.9), (1.03, 0.1), label="Right"),
        ],
        DetectionConfig(point_bounds_margin=0.08),
    )

    assert result.active is True
    assert result.points is not None
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in result.points)


def test_detection_falls_back_when_point_confidence_is_low():
    result = build_quad_detection(
        [
            _hand((0.2, 0.7), (0.2, 0.2), label="Left", visibility=0.49),
            _hand((0.8, 0.7), (0.8, 0.2), label="Right"),
        ],
        DetectionConfig(min_point_score=0.5),
    )

    assert result.active is False
    assert result.reason == "point confidence too low"


def test_detection_rejects_duplicate_handedness_when_strict():
    result = build_quad_detection(
        [
            _hand((0.2, 0.7), (0.2, 0.2), label="Left"),
            _hand((0.8, 0.7), (0.8, 0.2), label="Left"),
        ],
        DetectionConfig(require_distinct_handedness=True),
    )

    assert result.active is False
    assert result.reason == "requires left and right hands"


def test_detection_accepts_duplicate_handedness_by_default():
    result = build_quad_detection(
        [
            _hand((0.2, 0.7), (0.2, 0.2), label="Left"),
            _hand((0.8, 0.7), (0.8, 0.2), label="Left"),
        ],
        DetectionConfig(),
    )

    assert result.active is True


def test_detection_rejects_missing_handedness_when_strict():
    result = build_quad_detection(
        [
            _hand((0.2, 0.7), (0.2, 0.2), label=None),
            _hand((0.8, 0.7), (0.8, 0.2), label="Right"),
        ],
        DetectionConfig(require_distinct_handedness=True),
    )

    assert result.active is False
    assert result.reason == "requires handedness labels"


def test_detection_can_allow_ambiguous_handedness():
    result = build_quad_detection(
        [
            _hand((0.2, 0.7), (0.2, 0.2), label="Left"),
            _hand((0.8, 0.7), (0.8, 0.2), label="Left"),
        ],
        DetectionConfig(require_distinct_handedness=False),
    )

    assert result.active is True


def test_detection_does_not_reject_small_quadrilateral_by_area():
    result = build_quad_detection(
        [
            _hand((0.500, 0.500), (0.501, 0.500), label="Left"),
            _hand((0.501, 0.501), (0.500, 0.501), label="Right"),
        ],
        DetectionConfig(),
    )

    assert result.active is True


def test_detection_stays_active_when_plane_is_degenerate():
    result = build_quad_detection(
        [
            _hand((0.5, 0.5, 0.0), (0.5, 0.5, 0.0), label="Left"),
            _hand((0.5, 0.5, 0.0), (0.5, 0.5, 0.0), label="Right"),
        ],
        DetectionConfig(),
    )

    assert result.active is True
    assert result.points_3d is not None
    assert result.plane is None


def test_detection_estimates_plane_from_landmark_depth():
    result = build_quad_detection(
        [
            _hand((0.2, 0.7, 0.22), (0.2, 0.2, 0.12), label="Left"),
            _hand((0.8, 0.7, 0.52), (0.8, 0.2, 0.42), label="Right"),
        ],
        DetectionConfig(),
    )

    assert result.active is True
    assert result.points_3d is not None
    assert result.plane is not None
    assert result.plane.residual < 1e-12
    for x, y, z in result.points_3d:
        value = (
            result.plane.normal[0] * x
            + result.plane.normal[1] * y
            + result.plane.normal[2] * z
            + result.plane.offset
        )
        assert abs(value) < 1e-12
