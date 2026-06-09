"""Runtime environment checks."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from .effects import EffectConfig, apply_polygon_effect


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_doctor(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    print_line: Callable[[str], None] = print,
) -> int:
    results = [
        _check_imports(),
        _check_mediapipe_hands(),
        _check_camera_input(camera_index),
        _check_effect_pipeline(width, height, fps),
        _check_virtual_camera(width, height, fps),
    ]
    if platform.system() == "Darwin":
        results.append(_check_macos_camera_extension())

    for result in results:
        status = "ok" if result.ok else "fail"
        print_line(f"{status}: {result.name}: {result.detail}")

    return 0 if all(result.ok for result in results) else 1


def _check_imports() -> CheckResult:
    try:
        import cv2
        import mediapipe
        import numpy
        import pyvirtualcam
    except ImportError as exc:
        return CheckResult("dependencies", False, str(exc))

    return CheckResult(
        "dependencies",
        True,
        (
            f"mediapipe={mediapipe.__version__}, cv2={cv2.__version__}, "
            f"numpy={numpy.__version__}, pyvirtualcam={pyvirtualcam.__version__}"
        ),
    )


def _check_mediapipe_hands() -> CheckResult:
    try:
        import mediapipe as mp
    except ImportError as exc:
        return CheckResult("hand tracker", False, str(exc))

    ok = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
    detail = "MediaPipe solutions hands API available" if ok else "hands API missing"
    return CheckResult("hand tracker", ok, detail)


def _check_camera_input(camera_index: int) -> CheckResult:
    try:
        import cv2

        capture = cv2.VideoCapture(camera_index)
        opened = capture.isOpened()
        ok, frame = capture.read() if opened else (False, None)
        capture.release()
    except Exception as exc:
        return CheckResult("camera input", False, str(exc))

    if not opened:
        return CheckResult(
            "camera input",
            False,
            (
                f"camera index {camera_index} did not open; grant Camera access "
                "to Terminal/Python in System Settings > Privacy & Security > Camera"
            ),
        )
    if not ok or frame is None:
        return CheckResult("camera input", False, f"camera index {camera_index} did not read a frame")

    return CheckResult("camera input", True, f"camera index {camera_index} frame {frame.shape}")


def _check_virtual_camera(width: int, height: int, fps: int) -> CheckResult:
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(width=width, height=height, fps=fps)
        device = camera.device
        camera.close()
    except Exception as exc:
        return CheckResult("virtual camera", False, str(exc).splitlines()[0])

    return CheckResult("virtual camera", True, device)


def _check_effect_pipeline(width: int, height: int, target_fps: int) -> CheckResult:
    frame = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))
    config = EffectConfig()

    inactive = apply_polygon_effect(frame, None, config)
    if not np.array_equal(inactive, frame):
        return CheckResult("effect pipeline", False, "inactive frame changed pixels")

    for _ in range(3):
        apply_polygon_effect(frame, points, config)

    frames = 20
    started = perf_counter()
    for _ in range(frames):
        apply_polygon_effect(frame, points, config)
    elapsed = perf_counter() - started
    observed_fps = frames / elapsed if elapsed > 0 else float("inf")

    if observed_fps < target_fps:
        return CheckResult(
            "effect pipeline",
            False,
            f"{observed_fps:.1f} FPS below target {target_fps}",
        )

    return CheckResult(
        "effect pipeline",
        True,
        f"{observed_fps:.1f} FPS at {width}x{height}",
    )


def _check_macos_camera_extension() -> CheckResult:
    try:
        completed = subprocess.run(
            ["systemextensionsctl", "list"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return CheckResult("macOS camera extension", False, str(exc))

    output = completed.stdout + completed.stderr
    if completed.returncode != 0 and not output.strip():
        return CheckResult("macOS camera extension", False, "systemextensionsctl list failed")
    if "com.obsproject.obs-studio.mac-camera-extension" not in output:
        if "OSSystemExtensionErrorDomain error 1" in output:
            return CheckResult(
                "macOS camera extension",
                False,
                "systemextensionsctl list is blocked in this process; run `systemextensionsctl list` directly",
            )
        return CheckResult(
            "macOS camera extension",
            False,
            "OBS Virtual Camera extension is not registered",
        )
    if "[activated waiting for user]" in output:
        return CheckResult(
            "macOS camera extension",
            False,
            "approve OBS Virtual Camera in System Settings > General > Login Items & Extensions > Camera Extensions",
        )
    if "[activated enabled]" in output or "\t*\t" in output:
        return CheckResult("macOS camera extension", True, "OBS Virtual Camera registered")

    return CheckResult("macOS camera extension", False, output.strip().splitlines()[-1])
