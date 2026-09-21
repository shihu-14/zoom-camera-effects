"""Option definitions and value updates for the overlay controls."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .effects import (
    COLOR_NAMES_BGR,
    COLORMAPS,
    EffectConfig,
    EffectMode,
    format_color_hex,
)


COLOR_CHOICES = (
    "black",
    "white",
    "cyan",
    "magenta",
    "yellow",
    "red",
    "green",
    "blue",
)


COLORMAP_CHOICES = tuple(COLORMAPS)


@dataclass(frozen=True)
class NumericOption:
    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    integer: bool = False
    odd: bool = False
    decimals: int = 0


NUMERIC_OPTIONS = {
    "kernel_size": NumericOption(
        "kernel_size", "kernel", 3, 101, 2, integer=True, odd=True
    ),
    "mosaic_block_size": NumericOption(
        "mosaic_block_size", "block", 1, 64, 1, integer=True
    ),
    "edge_low_threshold": NumericOption(
        "edge_low_threshold", "low", 0, 255, 5, decimals=0
    ),
    "edge_high_threshold": NumericOption(
        "edge_high_threshold", "high", 0, 255, 5, decimals=0
    ),
    "noise_strength": NumericOption(
        "noise_strength", "strength", 0.0, 1.0, 0.05, decimals=2
    ),
    "outline_thickness": NumericOption(
        "outline_thickness", "thickness", 0, 30, 1, integer=True
    ),
}


EFFECT_OPTIONS: dict[EffectMode, tuple[str, ...]] = {
    "none": (),
    "blur": ("kernel_size",),
    "mosaic": ("mosaic_block_size",),
    "invert": (),
    "grayscale": (),
    "edge": ("edge_low_threshold", "edge_high_threshold"),
    "thermal": ("thermal_colormap",),
    "noise": ("noise_strength",),
    "outline": (
        "outline_thickness",
        "outline_color_bgr",
        "outline_fill",
        "outline_fill_color_bgr",
    ),
    "neon": (
        "edge_low_threshold",
        "edge_high_threshold",
        "noise_strength",
        "outline_color_bgr",
    ),
    "glitch": ("noise_strength",),
    "cartoon": (),
    "sketch": (),
}


def _set_numeric_from_slider(
    config: EffectConfig,
    key: str,
    x: int,
    rect: tuple[int, int, int, int],
) -> EffectConfig:
    option = NUMERIC_OPTIONS[key]
    ratio = (x - rect[0]) / max(rect[2], 1)
    ratio = min(max(ratio, 0.0), 1.0)
    value = option.minimum + (option.maximum - option.minimum) * ratio
    value = (
        option.minimum
        + round((value - option.minimum) / option.step) * option.step
    )
    if option.odd:
        value = int(value)
        if value % 2 == 0:
            value += 1
        value = min(max(value, int(option.minimum)), int(option.maximum))
    elif option.integer:
        value = int(round(value))
    else:
        value = round(value, option.decimals)
    value = min(max(value, option.minimum), option.maximum)
    return replace(config, **{key: value})


def _cycle_option(config: EffectConfig, key: str, direction: int) -> EffectConfig:
    if key == "thermal_colormap":
        value = _cycle_value(config.thermal_colormap, COLORMAP_CHOICES, direction)
        return replace(config, thermal_colormap=value)
    if key == "outline_fill":
        return replace(config, outline_fill=not config.outline_fill)
    if key in {"outline_color_bgr", "outline_fill_color_bgr"}:
        current = _color_label(getattr(config, key))
        value = _cycle_value(current, COLOR_CHOICES, direction)
        return replace(config, **{key: COLOR_NAMES_BGR[value]})
    return config


def _cycle_option_label(key: str) -> str:
    if key == "thermal_colormap":
        return "colormap"
    if key == "outline_fill":
        return "fill"
    if key == "outline_fill_color_bgr":
        return "fill color"
    return "color"


def _cycle_option_value(config: EffectConfig, key: str) -> str:
    if key == "thermal_colormap":
        return config.thermal_colormap
    if key == "outline_fill":
        return "on" if config.outline_fill else "off"
    return _color_label(getattr(config, key))


def _cycle_value(value: str, choices: tuple[str, ...], direction: int) -> str:
    try:
        index = choices.index(value)
    except ValueError:
        index = 0
    return choices[(index + direction) % len(choices)]


def _format_numeric_value(option: NumericOption, value: float | int) -> str:
    if option.key == "outline_thickness" and int(value) == 0:
        return "auto"
    if option.integer:
        return str(int(value))
    return f"{float(value):.{option.decimals}f}"


def _color_label(color_bgr: tuple[int, int, int]) -> str:
    for name in COLOR_CHOICES:
        if COLOR_NAMES_BGR[name] == color_bgr:
            return name
    return format_color_hex(color_bgr)
