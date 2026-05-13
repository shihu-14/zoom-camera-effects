"""Runtime loop for webcam processing and virtual camera output."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2

from .detection import DetectionConfig
from .effects import BlurConfig
from .hand_tracker import HandPointDetector
from .processor import FrameProcessor
from .virtual_camera import VirtualCameraWriter


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    preview: bool = False
    virtual_camera: bool = True
    mirror: bool = True
    max_frames: int | None = None
    smoothing_factor: float = 0.25
    detection: DetectionConfig = DetectionConfig()
    blur: BlurConfig = BlurConfig()
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.5


def run_app(config: AppConfig) -> int:
    if not config.virtual_camera and not config.preview:
        raise RuntimeError("enable --preview when --no-virtual-camera is used")

    capture = cv2.VideoCapture(config.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera index {config.camera_index}")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    capture.set(cv2.CAP_PROP_FPS, config.fps)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.width
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.height

    writer: VirtualCameraWriter | None = None
    detector: HandPointDetector | None = None
    try:
        writer = (
            VirtualCameraWriter(actual_width, actual_height, config.fps)
            if config.virtual_camera
            else None
        )
        detector = HandPointDetector(
            config.detection,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        processor = FrameProcessor(detector, config.blur, config.smoothing_factor)
        with detector:
            return _loop(capture, processor, writer, config)
    finally:
        capture.release()
        if writer is not None:
            writer.close()
        if config.preview:
            cv2.destroyAllWindows()


def _loop(
    capture: cv2.VideoCapture,
    processor: FrameProcessor,
    writer: VirtualCameraWriter | None,
    config: AppConfig,
) -> int:
    last_report = perf_counter()
    frames = 0
    total_frames = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("camera frame read failed")

        if config.mirror:
            frame = cv2.flip(frame, 1)

        processed_frame = processor.process(frame)
        processed = processed_frame.frame_bgr

        if writer is not None:
            writer.send_bgr(processed)

        if config.preview:
            cv2.imshow("Finger Quad Blur", processed)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                return 0

        frames += 1
        total_frames += 1
        if config.max_frames is not None and total_frames >= config.max_frames:
            return 0

        now = perf_counter()
        if now - last_report >= 5.0:
            observed_fps = frames / (now - last_report)
            if observed_fps < 10.0:
                print(
                    f"warning: observed FPS is {observed_fps:.1f}; "
                    "blur removal may exceed 0.1 seconds"
                )
            last_report = now
            frames = 0
