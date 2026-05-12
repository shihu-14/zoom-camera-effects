import numpy as np

from finger_quad_blur.detection import DetectionResult
from finger_quad_blur.effects import BlurConfig
from finger_quad_blur.processor import FrameProcessor


class SequenceDetector:
    def __init__(self, results):
        self._results = list(results)

    def detect(self, _frame_bgr):
        return self._results.pop(0)


def test_processor_removes_blur_on_first_inactive_frame():
    y_indices, x_indices = np.indices((80, 80))
    checker = ((x_indices + y_indices) % 2 * 255).astype(np.uint8)
    frame = np.dstack([checker, 255 - checker, checker])
    active = DetectionResult(
        True,
        points=((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)),
    )
    inactive = DetectionResult(False)
    processor = FrameProcessor(
        SequenceDetector([active, inactive]),
        BlurConfig(kernel_size=21, edge_feather_px=0),
    )

    active_frame = processor.process(frame).frame_bgr
    inactive_frame = processor.process(frame).frame_bgr

    assert not np.array_equal(active_frame, frame)
    assert np.array_equal(inactive_frame, frame)


def test_processor_smooths_active_points_and_resets_on_inactive_frame():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    first = DetectionResult(
        True,
        points=((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
    )
    second = DetectionResult(
        True,
        points=((0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)),
    )
    inactive = DetectionResult(False)
    third = DetectionResult(
        True,
        points=((0.4, 0.4), (0.4, 0.6), (0.6, 0.6), (0.6, 0.4)),
    )
    processor = FrameProcessor(
        SequenceDetector([first, second, inactive, third]),
        BlurConfig(mode="invert", edge_feather_px=0),
        smoothing_factor=0.5,
    )

    assert processor.process(frame).detection.points == first.points
    assert processor.process(frame).detection.points == (
        (0.1, 0.1),
        (0.1, 0.9),
        (0.9, 0.9),
        (0.9, 0.1),
    )
    assert processor.process(frame).detection.active is False
    assert processor.process(frame).detection.points == third.points
