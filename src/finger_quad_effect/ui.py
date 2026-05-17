"""Runtime OpenCV control UI for Finger Quad Effect."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import monotonic, sleep

import cv2
import numpy as np

from .control import default_control_file, parse_effect_config, write_effect_config
from .effects import COLORMAPS, EFFECT_MODES, EffectConfig

WINDOW_NAME = "Finger Quad Effect Controls"
COLORMAP_NAMES = tuple(COLORMAPS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the Finger Quad Effect control UI.")
    parser.add_argument("--control-file", type=Path, default=default_control_file())
    args = parser.parse_args()
    run_control_ui(args.control_file)
    return 0


def run_control_ui(control_file: Path) -> None:
    config = _load_initial_config(control_file)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 420, 620)
    cv2.moveWindow(WINDOW_NAME, 20, 20)
    _create_trackbars(config)

    last_config: EffectConfig | None = None
    last_write = 0.0
    while True:
        config = _read_trackbars()
        now = monotonic()
        if config != last_config and now - last_write >= 0.05:
            write_effect_config(config, control_file)
            last_config = config
            last_write = now

        cv2.imshow(WINDOW_NAME, _render_panel(config, control_file))
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        sleep(0.01)
    cv2.destroyWindow(WINDOW_NAME)


def _load_initial_config(control_file: Path) -> EffectConfig:
    try:
        line = control_file.read_text(encoding="utf-8").strip().splitlines()[0]
    except (FileNotFoundError, IndexError):
        return EffectConfig()
    try:
        return parse_effect_config(line)
    except ValueError:
        return EffectConfig()


def _create_trackbars(config: EffectConfig) -> None:
    cv2.createTrackbar(
        "effect", WINDOW_NAME, _effect_index(config.mode), len(EFFECT_MODES) - 1, _noop
    )
    cv2.createTrackbar(
        "kernel", WINDOW_NAME, _kernel_to_pos(config.kernel_size), 49, _noop
    )
    cv2.createTrackbar(
        "sigma x10",
        WINDOW_NAME,
        _clamp_int(round(config.sigma * 10), 0, 200),
        200,
        _noop,
    )
    cv2.createTrackbar(
        "block size",
        WINDOW_NAME,
        _clamp_int(config.mosaic_block_size, 1, 64),
        64,
        _noop,
    )
    cv2.createTrackbar(
        "feather", WINDOW_NAME, _clamp_int(config.edge_feather_px, 0, 40), 40, _noop
    )
    cv2.createTrackbar(
        "threshold low",
        WINDOW_NAME,
        _clamp_int(config.edge_low_threshold, 0, 255),
        255,
        _noop,
    )
    cv2.createTrackbar(
        "threshold high",
        WINDOW_NAME,
        _clamp_int(config.edge_high_threshold, 0, 255),
        255,
        _noop,
    )
    cv2.createTrackbar(
        "colormap",
        WINDOW_NAME,
        _colormap_index(config.thermal_colormap),
        len(COLORMAP_NAMES) - 1,
        _noop,
    )
    cv2.createTrackbar(
        "strength x100",
        WINDOW_NAME,
        _clamp_int(round(config.noise_strength * 100), 0, 100),
        100,
        _noop,
    )
    cv2.createTrackbar(
        "thickness",
        WINDOW_NAME,
        _clamp_int(config.outline_thickness, 0, 30),
        30,
        _noop,
    )
    blue, green, red = config.outline_color_bgr
    cv2.createTrackbar("color R", WINDOW_NAME, _clamp_int(red, 0, 255), 255, _noop)
    cv2.createTrackbar("color G", WINDOW_NAME, _clamp_int(green, 0, 255), 255, _noop)
    cv2.createTrackbar("color B", WINDOW_NAME, _clamp_int(blue, 0, 255), 255, _noop)


def _read_trackbars() -> EffectConfig:
    mode_index = _clamp_int(
        cv2.getTrackbarPos("effect", WINDOW_NAME), 0, len(EFFECT_MODES) - 1
    )
    colormap_index = _clamp_int(
        cv2.getTrackbarPos("colormap", WINDOW_NAME), 0, len(COLORMAP_NAMES) - 1
    )
    mode = EFFECT_MODES[mode_index]
    red = cv2.getTrackbarPos("color R", WINDOW_NAME)
    green = cv2.getTrackbarPos("color G", WINDOW_NAME)
    blue = cv2.getTrackbarPos("color B", WINDOW_NAME)
    return EffectConfig(
        mode=mode,
        kernel_size=_pos_to_kernel(cv2.getTrackbarPos("kernel", WINDOW_NAME)),
        sigma=cv2.getTrackbarPos("sigma x10", WINDOW_NAME) / 10.0,
        edge_feather_px=cv2.getTrackbarPos("feather", WINDOW_NAME),
        mosaic_block_size=max(1, cv2.getTrackbarPos("block size", WINDOW_NAME)),
        edge_low_threshold=float(cv2.getTrackbarPos("threshold low", WINDOW_NAME)),
        edge_high_threshold=float(cv2.getTrackbarPos("threshold high", WINDOW_NAME)),
        thermal_colormap=COLORMAP_NAMES[colormap_index],
        noise_strength=cv2.getTrackbarPos("strength x100", WINDOW_NAME) / 100.0,
        outline_thickness=cv2.getTrackbarPos("thickness", WINDOW_NAME),
        outline_color_bgr=(blue, green, red),
    )


def _render_panel(config: EffectConfig, control_file: Path) -> np.ndarray:
    image = np.full((360, 420, 3), (30, 34, 38), dtype=np.uint8)
    lines = [
        "Finger Quad Effect",
        f"effect: {config.mode}",
        f"kernel: {config.kernel_size}  sigma: {config.sigma:.1f}",
        f"block: {config.mosaic_block_size}  feather: {config.edge_feather_px}",
        f"threshold: {config.edge_low_threshold:.0f}-{config.edge_high_threshold:.0f}",
        f"colormap: {config.thermal_colormap}",
        f"strength: {config.noise_strength:.2f}  thickness: {config.outline_thickness}",
        (
            f"color RGB: {config.outline_color_bgr[2]}, "
            f"{config.outline_color_bgr[1]}, {config.outline_color_bgr[0]}"
        ),
        "",
        "Use the sliders to update in real time.",
        "Minimize this window from the title bar.",
        "Press q or Esc to close this panel.",
        "",
        f"control file: {control_file.name}",
    ]
    y = 32
    for index, line in enumerate(lines):
        scale = 0.58 if index == 0 else 0.45
        color = (235, 245, 255) if index == 0 else (205, 214, 224)
        cv2.putText(
            image,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 28 if index == 0 else 23
    return image


def _effect_index(mode: str) -> int:
    try:
        return EFFECT_MODES.index(mode)
    except ValueError:
        return 0


def _colormap_index(name: str) -> int:
    try:
        return COLORMAP_NAMES.index(name)
    except ValueError:
        return 0


def _kernel_to_pos(kernel_size: int) -> int:
    return max(0, min(49, (max(3, int(kernel_size)) - 3) // 2))


def _pos_to_kernel(position: int) -> int:
    return 3 + max(0, int(position)) * 2


def _clamp_int(value: float, minimum: int, maximum: int) -> int:
    return min(max(int(value), minimum), maximum)


def _noop(_value: int) -> None:
    return None


if __name__ == "__main__":
    raise SystemExit(main())
