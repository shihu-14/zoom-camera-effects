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
