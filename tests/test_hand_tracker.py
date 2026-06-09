from zoom_camera_effects.hand_tracker import _import_mediapipe


def test_mediapipe_solutions_hands_api_is_available():
    mp = _import_mediapipe()

    assert hasattr(mp, "solutions")
    assert hasattr(mp.solutions, "hands")
