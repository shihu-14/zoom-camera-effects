import pytest

from zoom_camera_effects.cli import (
    _build_parser,
    _format_effect_help,
    _parse_color_bgr,
    main,
)


def test_cli_defaults_to_blur_and_mirrored_image():
    args = _build_parser().parse_args([])

    assert args.effect == "blur"
    assert args.scope == "finger"
    assert args.area == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert args.mirror is True


def test_cli_can_disable_mirror():
    args = _build_parser().parse_args(["--no-mirror"])

    assert args.mirror is False


def test_cli_accepts_full_effect_scope():
    args = _build_parser().parse_args(["--scope", "full"])

    assert args.scope == "full"


def test_cli_accepts_partial_effect_scope_and_area():
    args = _build_parser().parse_args(
        ["--scope", "partial", "--area", "0,0", "1,0", "1,1", "0,1"]
    )

    assert args.scope == "partial"
    assert args.area == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def test_cli_passes_full_effect_scope(monkeypatch):
    captured = {}

    def fake_run_app(config):
        captured["config"] = config
        return 0

    monkeypatch.setattr("zoom_camera_effects.cli.run_app", fake_run_app)
    monkeypatch.setattr("sys.argv", ["zoom_camera_effects", "--scope", "full"])

    assert main() == 0
    assert captured["config"].effect.scope == "full"


def test_cli_passes_partial_area(monkeypatch):
    captured = {}

    def fake_run_app(config):
        captured["config"] = config
        return 0

    monkeypatch.setattr("zoom_camera_effects.cli.run_app", fake_run_app)
    monkeypatch.setattr(
        "sys.argv",
        [
            "zoom_camera_effects",
            "--scope",
            "partial",
            "--area",
            "0,0",
            "1,0",
            "1,1",
        ],
    )

    assert main() == 0
    assert captured["config"].effect.scope == "partial"
    assert captured["config"].effect.area_points == (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
    )


def test_cli_passes_outline_fill_options(monkeypatch):
    captured = {}

    def fake_run_app(config):
        captured["config"] = config
        return 0

    monkeypatch.setattr("zoom_camera_effects.cli.run_app", fake_run_app)
    monkeypatch.setattr(
        "sys.argv",
        [
            "zoom_camera_effects",
            "--effect",
            "outline",
            "--fill",
            "--fill-color",
            "#202020",
        ],
    )

    assert main() == 0
    assert captured["config"].effect.outline_fill is True
    assert captured["config"].effect.outline_fill_color_bgr == (32, 32, 32)


def test_cli_rejects_too_few_area_points(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["zoom_camera_effects", "--scope", "partial", "--area", "0,0", "1,0"],
    )

    with pytest.raises(SystemExit):
        main()


def test_preview_disables_virtual_camera(monkeypatch):
    captured = {}

    def fake_run_app(config):
        captured["config"] = config
        return 0

    monkeypatch.setattr("zoom_camera_effects.cli.run_app", fake_run_app)
    monkeypatch.setattr("sys.argv", ["zoom_camera_effects", "--preview"])

    assert main() == 0
    assert captured["config"].preview is True
    assert captured["config"].virtual_camera is False
    assert captured["config"].ui is True


def test_cli_can_disable_ui(monkeypatch):
    captured = {}

    def fake_run_app(config):
        captured["config"] = config
        return 0

    monkeypatch.setattr("zoom_camera_effects.cli.run_app", fake_run_app)
    monkeypatch.setattr("sys.argv", ["zoom_camera_effects", "--no-ui"])

    assert main() == 0
    assert captured["config"].ui is False


def test_no_control_keeps_overlay_ui(monkeypatch):
    captured = {}

    def fake_run_app(config):
        captured["config"] = config
        return 0

    monkeypatch.setattr("zoom_camera_effects.cli.run_app", fake_run_app)
    monkeypatch.setattr("sys.argv", ["zoom_camera_effects", "--no-control"])

    assert main() == 0
    assert captured["config"].control_file is None
    assert captured["config"].ui is True


@pytest.mark.parametrize(
    "mode",
    [
        "none",
        "edge",
        "thermal",
        "noise",
        "outline",
        "neon",
        "glitch",
        "cartoon",
        "sketch",
    ],
)
def test_cli_accepts_added_effect_modes(mode):
    args = _build_parser().parse_args(["--effect", mode])

    assert args.effect == mode


def test_cli_accepts_runtime_effect_command(tmp_path):
    control_file = tmp_path / "effect.txt"
    args = _build_parser().parse_args(
        ["--set-effect", "thermal", "--control-file", str(control_file)]
    )

    assert args.set_effect == "thermal"
    assert args.control_file == control_file


def test_cli_accepts_short_effect_parameters():
    args = _build_parser().parse_args(
        [
            "--effect",
            "thermal",
            "--kernel",
            "51",
            "--block-size",
            "24",
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
            "--fill",
            "--fill-color",
            "red",
        ]
    )

    assert args.kernel == 51
    assert args.block_size == 24
    assert args.threshold_low == 30
    assert args.threshold_high == 90
    assert args.colormap == "turbo"
    assert args.strength == 0.4
    assert args.thickness == 6
    assert args.color == "#00ffff"
    assert args.fill is True
    assert args.fill_color == "red"


def test_cli_keeps_legacy_parameter_aliases():
    args = _build_parser().parse_args(
        ["--blur-kernel", "51", "--mosaic-block-size", "24"]
    )

    assert args.kernel == 51
    assert args.block_size == 24


def test_removed_effect_list_aliases_are_not_accepted():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--list-effects"])

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--help-effects"])


def test_removed_preview_virtual_camera_option_is_not_accepted():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--preview", "--no-virtual-camera"])


def test_removed_sigma_option_is_not_accepted():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--sigma", "7"])


def test_removed_feather_options_are_not_accepted():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--feather", "5"])

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--edge-feather", "5"])


def test_removed_particles_effect_is_not_accepted():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--effect", "particles"])


def test_effect_help_includes_effects_and_parameter_notes():
    output = _format_effect_help()

    assert "Syntax:" in output
    assert "Syntax rules:" in output
    assert "Effects:" in output
    assert "Effect options:" in output
    assert "none      No effect. Use: --effect none" in output
    assert "blur      Gaussian blur in the selected area. Use: --effect blur" in output
    assert "mosaic" in output
    assert "particles" not in output
    assert "neon" in output
    assert "glitch" in output
    assert "cartoon" in output
    assert "sketch" in output
    assert "--kernel" in output
    assert "--threshold-low" in output


def test_help_output_places_effects_before_advanced_without_duplicate_sections():
    output = _build_parser().format_help()

    assert "usage: python3 -m zoom_camera_effects [options]" in output
    assert "\neffects:" not in output
    assert output.count("Syntax:") == 1
    assert output.count("Syntax rules:") == 1
    assert output.count("Effects:") == 1
    assert output.count("Effect options:") == 1
    assert output.index("Effects:") < output.index("advanced:")
    assert output.index("Effect options:") < output.index("advanced:")
    assert "[--kernel {kernel}]                    default: 35" in output
    assert "[--scope {scope}]        default: finger" in output
    assert "[--area {x,y x,y x,y [x,y]...}]" in output
    assert "default: 0,0 1,0 1,1 0,1; 3..10 points" in output
    assert "[--fill | --no-fill]                   default: --no-fill" in output
    assert "[--fill-color {color}]                 default: #ffffff" in output
    assert "--sigma" not in output


def test_parse_color_converts_rgb_hex_to_bgr():
    assert _parse_color_bgr("#00ffff") == (255, 255, 0)
    assert _parse_color_bgr("red") == (0, 0, 255)
