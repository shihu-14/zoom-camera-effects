import pytest

from finger_quad_effect.cli import (
    _build_parser,
    _format_help_epilog,
    _parse_color_bgr,
    main,
)


def test_cli_defaults_to_blur_and_mirrored_image():
    args = _build_parser().parse_args([])

    assert args.effect == "blur"
    assert args.mirror is True


def test_cli_can_disable_mirror():
    args = _build_parser().parse_args(["--no-mirror"])

    assert args.mirror is False


def test_preview_disables_virtual_camera(monkeypatch):
    captured = {}

    def fake_run_app(config):
        captured["config"] = config
        return 0

    monkeypatch.setattr("finger_quad_effect.cli.run_app", fake_run_app)
    monkeypatch.setattr("sys.argv", ["finger_quad_effect", "--preview"])

    assert main() == 0
    assert captured["config"].preview is True
    assert captured["config"].virtual_camera is False


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


def test_cli_accepts_short_effect_parameters():
    args = _build_parser().parse_args(
        [
            "--effect",
            "thermal",
            "--kernel",
            "51",
            "--sigma",
            "7",
            "--block-size",
            "24",
            "--feather",
            "5",
            "--threshold-low",
            "30",
            "--threshold-high",
            "90",
            "--colormap",
            "turbo",
            "--strength",
            "0.4",
            "--thickness",
            "6",
            "--color",
            "#00ffff",
        ]
    )

    assert args.kernel == 51
    assert args.sigma == 7
    assert args.block_size == 24
    assert args.feather == 5
    assert args.threshold_low == 30
    assert args.threshold_high == 90
    assert args.colormap == "turbo"
    assert args.strength == 0.4
    assert args.thickness == 6
    assert args.color == "#00ffff"


def test_cli_keeps_legacy_parameter_aliases():
    args = _build_parser().parse_args(
        ["--blur-kernel", "51", "--mosaic-block-size", "24", "--edge-feather", "5"]
    )

    assert args.kernel == 51
    assert args.block_size == 24
    assert args.feather == 5


def test_removed_effect_list_aliases_are_not_accepted():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--list-effects"])

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--help-effects"])


def test_removed_preview_virtual_camera_option_is_not_accepted():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--preview", "--no-virtual-camera"])


def test_help_epilog_includes_effects_and_parameter_notes():
    output = _format_help_epilog()

    assert "Effects:" in output
    assert "blur" in output
    assert "mosaic" in output
    assert "particles" in output
    assert "--kernel" in output
    assert "--threshold-low" in output


def test_parse_color_converts_rgb_hex_to_bgr():
    assert _parse_color_bgr("#00ffff") == (255, 255, 0)
    assert _parse_color_bgr("red") == (0, 0, 255)
