import json

import pytest

from zoom_camera_effects.control import (
    EffectControlReader,
    effect_config_from_mapping,
    write_effect_command,
    write_effect_config,
)
from zoom_camera_effects.effects import EffectConfig


def test_write_and_read_runtime_effect_command(tmp_path):
    control_file = tmp_path / "effect.txt"
    reader = EffectControlReader(control_file)

    assert reader.read_effect() is None

    write_effect_command("thermal", control_file)

    assert reader.read_effect() == "thermal"
    assert reader.read_effect() is None


def test_write_and_read_runtime_effect_config(tmp_path):
    control_file = tmp_path / "effect.txt"
    reader = EffectControlReader(control_file)
    config = EffectConfig(
        mode="neon",
        kernel_size=51,
        noise_strength=0.9,
        outline_color_bgr=(255, 255, 0),
        outline_fill=True,
        outline_fill_color_bgr=(32, 32, 32),
    )

    write_effect_config(config, control_file)

    result = reader.read_config()
    assert result == config
    payload = json.loads(control_file.read_text(encoding="utf-8"))
    assert payload["outline_color"] == "#00ffff"
    assert payload["fill_color"] == "#202020"


def test_runtime_effect_reader_can_ignore_existing_command(tmp_path):
    control_file = tmp_path / "effect.txt"
    write_effect_command("thermal", control_file)
    reader = EffectControlReader(control_file)

    reader.ignore_current()

    assert reader.read_effect() is None
    write_effect_command("edge", control_file)
    assert reader.read_effect() == "edge"


def test_runtime_effect_reader_can_reset_existing_command(tmp_path):
    control_file = tmp_path / "effect.txt"
    write_effect_config(
        EffectConfig(mode="neon", kernel_size=51, noise_strength=0.4),
        control_file,
    )
    reader = EffectControlReader(control_file)

    reader.reset_current()

    assert not control_file.exists()
    assert reader.read_config() is None
    write_effect_command("edge", control_file)
    assert reader.read_effect() == "edge"


def test_runtime_effect_command_normalizes_alias(tmp_path):
    control_file = tmp_path / "effect.txt"

    write_effect_command("monochrome", control_file)

    assert EffectControlReader(control_file).read_effect() == "grayscale"


def test_runtime_effect_command_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError):
        write_effect_command("unknown", tmp_path / "effect.txt")


def test_runtime_effect_command_rejects_removed_particles(tmp_path):
    with pytest.raises(ValueError):
        write_effect_command("particles", tmp_path / "effect.txt")


def test_runtime_effect_config_preserves_existing_options():
    base = EffectConfig(mode="blur", kernel_size=51, noise_strength=0.4)

    result = effect_config_from_mapping({"mode": "glitch"}, base)

    assert result.mode == "glitch"
    assert result.kernel_size == 51
    assert result.noise_strength == 0.4


def test_runtime_effect_config_accepts_ui_aliases():
    result = effect_config_from_mapping(
        {
            "mode": "neon",
            "kernel": 31,
            "block_size": 22,
            "threshold_low": 20,
            "threshold_high": 100,
            "colormap": "turbo",
            "strength": 0.7,
            "thickness": 5,
            "outline_color": "#00ffff",
            "fill": "on",
            "fill_color": "red",
            "apply_to": "full",
            "area": "0,0 1,0 1,1 0,1",
        }
    )

    assert result.mode == "neon"
    assert result.scope == "full"
    assert result.area_points == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert result.kernel_size == 31
    assert result.mosaic_block_size == 22
    assert result.edge_low_threshold == 20
    assert result.edge_high_threshold == 100
    assert result.thermal_colormap == "turbo"
    assert result.noise_strength == 0.7
    assert result.outline_thickness == 5
    assert result.outline_color_bgr == (255, 255, 0)
    assert result.outline_fill is True
    assert result.outline_fill_color_bgr == (0, 0, 255)


def test_runtime_effect_config_rejects_unknown_scope():
    with pytest.raises(ValueError):
        effect_config_from_mapping({"scope": "window"})


def test_runtime_effect_config_rejects_too_few_area_points():
    with pytest.raises(ValueError):
        effect_config_from_mapping({"area_points": [[0, 0], [1, 0]]})


def test_runtime_effect_config_ignores_sigma():
    result = effect_config_from_mapping({"sigma": 7})

    assert not hasattr(result, "sigma")


def test_runtime_effect_config_ignores_feather():
    result = effect_config_from_mapping({"feather": 7})

    assert not hasattr(result, "edge_feather_px")
