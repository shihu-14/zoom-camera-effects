import pytest

from finger_quad_blur.control import EffectControlReader, write_effect_command


def test_write_and_read_runtime_effect_command(tmp_path):
    control_file = tmp_path / "effect.txt"
    reader = EffectControlReader(control_file)

    assert reader.read_effect() is None

    write_effect_command("particles", control_file)

    assert reader.read_effect() == "particles"
    assert reader.read_effect() is None


def test_runtime_effect_command_normalizes_alias(tmp_path):
    control_file = tmp_path / "effect.txt"

    write_effect_command("monochrome", control_file)

    assert EffectControlReader(control_file).read_effect() == "grayscale"


def test_runtime_effect_command_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError):
        write_effect_command("unknown", tmp_path / "effect.txt")
