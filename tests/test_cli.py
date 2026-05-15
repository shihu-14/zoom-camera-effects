import pytest

from finger_quad_effect.cli import _build_parser, _format_effect_list


def test_cli_defaults_to_blur_and_mirrored_image():
    args = _build_parser().parse_args([])

    assert args.effect == "blur"
    assert args.mirror is True


def test_cli_can_disable_mirror():
    args = _build_parser().parse_args(["--no-mirror"])

    assert args.mirror is False


@pytest.mark.parametrize("mode", ["edge", "thermal", "noise", "outline", "particles"])
def test_cli_accepts_added_effect_modes(mode):
    args = _build_parser().parse_args(["--effect", mode])

    assert args.effect == mode


def test_cli_accepts_runtime_effect_command(tmp_path):
    control_file = tmp_path / "effect.txt"
    args = _build_parser().parse_args(
        ["--set-effect", "particles", "--control-file", str(control_file)]
    )

    assert args.set_effect == "particles"
    assert args.control_file == control_file


def test_cli_accepts_effect_list_help_alias():
    args = _build_parser().parse_args(["--help-effects"])

    assert args.list_effects is True


def test_effect_list_help_includes_all_modes():
    output = _format_effect_list()

    assert "Available effects:" in output
    assert "blur" in output
    assert "mosaic" in output
    assert "particles" in output
    assert "python3 -m finger_quad_effect --effect <name>" in output
