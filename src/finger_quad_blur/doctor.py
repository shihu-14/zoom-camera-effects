"""Runtime environment checks."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_doctor(
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    print_line: Callable[[str], None] = print,
) -> int:
    results = [
        _check_imports(),
        _check_mediapipe_hands(),
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


def _check_virtual_camera(width: int, height: int, fps: int) -> CheckResult:
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(width=width, height=height, fps=fps)
        device = camera.device
        camera.close()
    except Exception as exc:
        return CheckResult("virtual camera", False, str(exc).splitlines()[0])

    return CheckResult("virtual camera", True, device)


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
    if "com.obsproject.obs-studio.mac-camera-extension" not in output:
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
