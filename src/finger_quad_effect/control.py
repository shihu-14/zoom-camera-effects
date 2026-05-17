"""Runtime control file helpers."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

from .effects import (
    COLORMAPS,
    EffectConfig,
    EffectMode,
    format_color_hex,
    normalize_effect_mode,
    parse_color_bgr,
)


def default_control_file() -> Path:
    return Path(tempfile.gettempdir()) / "finger_quad_effect_control.txt"


def write_effect_command(effect: str, control_file: Path | None = None) -> Path:
    path = control_file or default_control_file()
    mode = normalize_effect_mode(effect)
    _write_control_text(path, f"{mode}\n")
    return path


def write_effect_config(config: EffectConfig, control_file: Path | None = None) -> Path:
    path = control_file or default_control_file()
    payload = asdict(config)
    payload["outline_color"] = format_color_hex(config.outline_color_bgr)
    payload.pop("outline_color_bgr")
    _write_control_text(path, json.dumps(payload, sort_keys=True) + "\n")
    return path


def _write_control_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


@dataclass
class EffectControlReader:
    path: Path
    _last_mtime_ns: int | None = None

    def ignore_current(self) -> None:
        try:
            self._last_mtime_ns = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            self._last_mtime_ns = None

    def reset_current(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._last_mtime_ns = None

    def read_config(self, base_config: EffectConfig | None = None) -> EffectConfig | None:
        try:
            stat_result = self.path.stat()
        except FileNotFoundError:
            return None

        if stat_result.st_mtime_ns == self._last_mtime_ns:
            return None
        self._last_mtime_ns = stat_result.st_mtime_ns

        line = self.path.read_text(encoding="utf-8").strip().splitlines()
        if not line:
            return None
        return parse_effect_config(line[0].strip(), base_config)

    def read_effect(self) -> EffectMode | None:
        config = self.read_config()
        return None if config is None else config.mode


def parse_effect_config(
    raw_value: str,
    base_config: EffectConfig | None = None,
) -> EffectConfig:
    base = base_config or EffectConfig()
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError:
        return replace(base, mode=normalize_effect_mode(raw_value))
    if isinstance(payload, str):
        return replace(base, mode=normalize_effect_mode(payload))
    if not isinstance(payload, Mapping):
        raise ValueError("runtime control command must be an effect name or object")
    return effect_config_from_mapping(payload, base)


def effect_config_from_mapping(
    payload: Mapping[str, Any],
    base_config: EffectConfig | None = None,
) -> EffectConfig:
    values = asdict(base_config or EffectConfig())
    allowed = {field.name for field in fields(EffectConfig)}
    aliases = {
        "kernel": "kernel_size",
        "block_size": "mosaic_block_size",
        "threshold_low": "edge_low_threshold",
        "threshold_high": "edge_high_threshold",
        "colormap": "thermal_colormap",
        "strength": "noise_strength",
        "thickness": "outline_thickness",
    }
    int_fields = {
        "kernel_size",
        "mosaic_block_size",
        "outline_thickness",
    }
    float_fields = {
        "edge_low_threshold",
        "edge_high_threshold",
        "noise_strength",
    }

    for key, value in payload.items():
        target = aliases.get(key, key)
        if target == "outline_color":
            values["outline_color_bgr"] = parse_color_bgr(str(value))
            continue
        if target == "outline_color_bgr":
            values[target] = _coerce_bgr(value)
            continue
        if target not in allowed:
            continue
        if target == "mode":
            values[target] = normalize_effect_mode(str(value))
        elif target == "thermal_colormap":
            colormap = str(value)
            if colormap not in COLORMAPS:
                raise ValueError(f"unsupported colormap: {colormap}")
            values[target] = colormap
        elif target in int_fields:
            values[target] = int(value)
        elif target in float_fields:
            values[target] = float(value)
        else:
            values[target] = value
    return EffectConfig(**values)


def _coerce_bgr(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        return parse_color_bgr(value)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("outline_color_bgr must contain three channels")
    return tuple(min(max(int(channel), 0), 255) for channel in value)
