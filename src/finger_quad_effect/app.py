"""Runtime loop for webcam processing and virtual camera output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2

from .control import EffectControlReader, default_control_file
from .detection import DetectionConfig
from .effects import EffectConfig
from .hand_tracker import HandPointDetector
from .processor import FrameProcessor
from .ui import OverlayControlUI
from .virtual_camera import VirtualCameraWriter

DISPLAY_WINDOW_NAME = "Finger Quad Effect"


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
    effect: EffectConfig = EffectConfig()
    control_file: Path | None = default_control_file()
    ui: bool = True
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.5


def run_app(config: AppConfig) -> int:
    if not config.virtual_camera and not config.preview and not config.ui:
        raise RuntimeError("enable preview, UI, or virtual camera output")

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
        processor = FrameProcessor(detector, config.effect, config.smoothing_factor)
        with detector:
            return _loop(capture, processor, writer, config)
    finally:
        capture.release()
        if writer is not None:
            writer.close()
        if config.preview or config.ui:
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
    control_reader = (
        EffectControlReader(config.control_file)
        if config.control_file is not None
        else None
    )
    if control_reader is not None:
        control_reader.reset_current()

    overlay_ui = OverlayControlUI() if config.ui else None
    show_display = config.preview or overlay_ui is not None
    if show_display:
        cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(DISPLAY_WINDOW_NAME, 20, 20)
        if overlay_ui is not None:
            cv2.setMouseCallback(DISPLAY_WINDOW_NAME, overlay_ui.handle_mouse)

    while True:
        if control_reader is not None:
            try:
                effect_config = control_reader.read_config(processor.effect_config)
            except ValueError as exc:
                print(f"warning: ignoring runtime control command: {exc}")
                effect_config = None
            if effect_config is not None and effect_config != processor.effect_config:
                processor.set_effect_config(effect_config)
                print(f"effect switched: {effect_config.mode}")

        if overlay_ui is not None:
            effect_config = overlay_ui.consume_pending_config(processor.effect_config)
            if effect_config is not None and effect_config != processor.effect_config:
                processor.set_effect_config(effect_config)

        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("camera frame read failed")

        if config.mirror:
            frame = cv2.flip(frame, 1)

        processed_frame = processor.process(frame)
        processed = processed_frame.frame_bgr

        if writer is not None:
            writer.send_bgr(processed)

        if show_display:
            display = processed
            if overlay_ui is not None:
                display = overlay_ui.render(processed, processor.effect_config)
            cv2.imshow(DISPLAY_WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                return 0
            if cv2.getWindowProperty(DISPLAY_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
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
                    "effect removal may exceed 0.1 seconds"
                )
            last_report = now
            frames = 0
