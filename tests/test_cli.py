from finger_quad_blur.cli import _build_parser


def test_cli_defaults_to_blur_and_mirrored_image():
    args = _build_parser().parse_args([])

    assert args.effect == "blur"
    assert args.mirror is True


def test_cli_can_disable_mirror():
    args = _build_parser().parse_args(["--no-mirror"])

    assert args.mirror is False
