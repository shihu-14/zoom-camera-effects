"""Runtime control file helpers."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .effects import EffectMode, normalize_effect_mode


def default_control_file() -> Path:
    return Path(tempfile.gettempdir()) / "finger_quad_effect_control.txt"


def write_effect_command(effect: str, control_file: Path | None = None) -> Path:
    path = control_file or default_control_file()
    mode = normalize_effect_mode(effect)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(f"{mode}\n", encoding="utf-8")
    temp_path.replace(path)
    return path


@dataclass
class EffectControlReader:
    path: Path
    _last_mtime_ns: int | None = None

    def ignore_current(self) -> None:
        try:
            self._last_mtime_ns = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            self._last_mtime_ns = None

    def read_effect(self) -> EffectMode | None:
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
        return normalize_effect_mode(line[0].strip())
