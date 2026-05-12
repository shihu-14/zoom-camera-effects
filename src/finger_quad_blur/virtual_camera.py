"""Virtual camera output wrapper."""

from __future__ import annotations

import cv2


class VirtualCameraWriter:
    def __init__(self, width: int, height: int, fps: int) -> None:
        try:
            import pyvirtualcam
            from pyvirtualcam import PixelFormat
        except ImportError as exc:
            raise RuntimeError(
                "pyvirtualcam is required for virtual camera output. "
                "Install dependencies with `python3 -m pip install -e .`."
            ) from exc

        self._pyvirtualcam = pyvirtualcam
        self._camera = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=fps,
            fmt=PixelFormat.RGB,
        )
        print(f"virtual camera active: {self._camera.device}")

    def send_bgr(self, frame_bgr) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._camera.send(frame_rgb)
        self._camera.sleep_until_next_frame()

    def close(self) -> None:
        self._camera.close()

    def __enter__(self) -> "VirtualCameraWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
