"""Runtime loop for webcam processing and virtual camera output."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2

from .control import EffectControlReader, default_control_file, write_effect_config
from .detection import DetectionConfig
from .effects import EffectConfig
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
    effect: EffectConfig = EffectConfig()
    control_file: Path | None = default_control_file()
    ui: bool = True
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.5


def run_app(config: AppConfig) -> int:
    if not config.virtual_camera and not config.preview:
        raise RuntimeError("enable preview or virtual camera output")

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
    ui_process: subprocess.Popen | None = None
    try:
        if config.ui and config.control_file is not None:
            write_effect_config(config.effect, config.control_file)
            ui_process = _start_control_ui(config.control_file)
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
        if config.preview:
            cv2.destroyAllWindows()
        if ui_process is not None and ui_process.poll() is None:
            ui_process.terminate()
            try:
                ui_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                ui_process.kill()


def _start_control_ui(control_file: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "finger_quad_effect.ui",
            "--control-file",
            str(control_file),
        ],
        close_fds=True,
    )


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
        control_reader.ignore_current()

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
            cv2.imshow("Finger Quad Effect", processed)
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
                    "effect removal may exceed 0.1 seconds"
                )
            last_report = now
            frames = 0
