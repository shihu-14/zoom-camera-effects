from finger_quad_effect.control import write_effect_config
from finger_quad_effect.effects import EffectConfig
from finger_quad_effect.ui import (
    _clamp_int,
    _kernel_to_pos,
    _load_initial_config,
    _pos_to_kernel,
    _render_panel,
)


def test_ui_kernel_position_round_trip():
    assert _pos_to_kernel(_kernel_to_pos(51)) == 51
    assert _pos_to_kernel(_kernel_to_pos(2)) == 3


def test_ui_clamps_trackbar_values():
    assert _clamp_int(-1, 0, 10) == 0
    assert _clamp_int(20, 0, 10) == 10


def test_ui_loads_initial_config(tmp_path):
    control_file = tmp_path / "control.txt"
    config = EffectConfig(mode="neon", noise_strength=0.9)

    write_effect_config(config, control_file)

    assert _load_initial_config(control_file) == config


def test_ui_panel_renders_status_image(tmp_path):
    image = _render_panel(EffectConfig(mode="glitch"), tmp_path / "control.txt")

    assert image.shape == (360, 420, 3)
    assert image.max() > image.min()
