"""Command line entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from .app import AppConfig, run_app
from .control import default_control_file, write_effect_command
from .detection import DetectionConfig
from .doctor import run_doctor
from .effects import (
    COLORMAPS,
    EFFECT_DESCRIPTIONS,
    EFFECT_MODES,
    EffectConfig,
    normalize_effect_mode,
)

EFFECT_CHOICES = (*EFFECT_MODES, "monochrome")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.set_effect:
        control_file = write_effect_command(args.set_effect, args.control_file)
        print(f"effect command written: {args.set_effect} -> {control_file}")
        return 0

    if args.doctor:
        return run_doctor(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )

    try:
        outline_color_bgr = _parse_color_bgr(args.color)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    config = AppConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        preview=args.preview,
        virtual_camera=not args.preview,
        mirror=args.mirror,
        max_frames=args.max_frames,
        smoothing_factor=args.smoothing_factor,
        detection=DetectionConfig(
            min_hand_score=args.min_hand_score,
            min_point_score=args.min_point_score,
            point_bounds_margin=args.point_bounds_margin,
            require_distinct_handedness=args.require_distinct_handedness,
        ),
        effect=EffectConfig(
            mode=_normalize_effect_mode(args.effect),
            kernel_size=args.kernel,
            sigma=args.sigma,
            edge_feather_px=args.feather,
            mosaic_block_size=args.block_size,
            edge_low_threshold=args.threshold_low,
            edge_high_threshold=args.threshold_high,
            thermal_colormap=args.colormap,
            noise_strength=args.strength,
            outline_thickness=args.thickness,
            outline_color_bgr=outline_color_bgr,
        ),
        control_file=None if args.no_control else args.control_file,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    try:
        return run_app(config)
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def _build_parser() -> argparse.ArgumentParser:
    default_effect = EffectConfig()
    parser = argparse.ArgumentParser(
        description="Apply an effect inside the quadrilateral formed by both thumbs and index fingers.",
        epilog=_format_help_epilog(),
        formatter_class=_HelpFormatter,
    )

    run_group = parser.add_argument_group("run")
    run_group.add_argument("--camera-index", type=int, default=0)
    run_group.add_argument("--width", type=int, default=1280)
    run_group.add_argument("--height", type=int, default=720)
    run_group.add_argument("--fps", type=int, default=30)
    run_group.add_argument("--doctor", action="store_true", help="check runtime dependencies and exit")
    run_group.add_argument(
        "--preview",
        action="store_true",
        help="open a local preview window without virtual camera output",
    )
    run_group.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mirror the camera image horizontally",
    )
    run_group.add_argument("--max-frames", type=int, help="process this many frames and exit")
    effect_group = parser.add_argument_group("effects")
    effect_group.add_argument(
        "--effect",
        choices=EFFECT_CHOICES,
        default=default_effect.mode,
        help="effect to apply",
    )
    effect_group.add_argument(
        "--set-effect",
        choices=EFFECT_CHOICES,
        help="write a runtime effect switch command and exit",
    )
    effect_group.add_argument(
        "--kernel",
        dest="kernel",
        type=int,
        default=default_effect.kernel_size,
        help="Gaussian blur kernel size",
    )
    effect_group.add_argument(
        "--blur-kernel",
        dest="kernel",
        type=int,
        help=argparse.SUPPRESS,
    )
    effect_group.add_argument(
        "--sigma",
        type=float,
        default=default_effect.sigma,
        help="Gaussian blur sigma; 0 lets OpenCV infer it from kernel",
    )
    effect_group.add_argument(
        "--block-size",
        dest="block_size",
        type=int,
        default=default_effect.mosaic_block_size,
        help="mosaic pixel block size",
    )
    effect_group.add_argument(
        "--mosaic-block-size",
        dest="block_size",
        type=int,
        help=argparse.SUPPRESS,
    )
    effect_group.add_argument(
        "--feather",
        dest="feather",
        type=int,
        default=default_effect.edge_feather_px,
        help="polygon edge feather radius in pixels",
    )
    effect_group.add_argument(
        "--edge-feather",
        dest="feather",
        type=int,
        help=argparse.SUPPRESS,
    )
    effect_group.add_argument(
        "--threshold-low",
        type=float,
        default=default_effect.edge_low_threshold,
        help="Canny low threshold for edge effect",
    )
    effect_group.add_argument(
        "--threshold-high",
        type=float,
        default=default_effect.edge_high_threshold,
        help="Canny high threshold for edge effect",
    )
    effect_group.add_argument(
        "--colormap",
        choices=tuple(COLORMAPS),
        default=default_effect.thermal_colormap,
        help="OpenCV colormap for thermal effect",
    )
    effect_group.add_argument(
        "--strength",
        type=float,
        default=default_effect.noise_strength,
        help="noise blend strength from 0 to 1",
    )
    effect_group.add_argument(
        "--thickness",
        type=int,
        default=default_effect.outline_thickness,
        help="outline thickness in pixels; 0 uses automatic thickness",
    )
    effect_group.add_argument(
        "--color",
        default="#000000",
        metavar="COLOR",
        help="outline color as #RRGGBB or a basic color name",
    )

    advanced_group = parser.add_argument_group("advanced")
    advanced_group.add_argument(
        "--control-file",
        type=Path,
        default=default_control_file(),
        help="file used for runtime effect switching commands",
    )
    advanced_group.add_argument(
        "--no-control",
        action="store_true",
        help="disable runtime effect switching from the control file",
    )
    advanced_group.add_argument(
        "--smoothing-factor",
        type=float,
        default=0.25,
        help="fingertip point smoothing factor from 0 to 1",
    )
    advanced_group.add_argument(
        "--min-hand-score",
        type=float,
        default=0.55,
        help="minimum detected hand score accepted by the app",
    )
    advanced_group.add_argument(
        "--min-point-score",
        type=float,
        default=0.5,
        help="minimum fingertip point score accepted by the app",
    )
    advanced_group.add_argument(
        "--point-bounds-margin",
        type=float,
        default=0.12,
        help="normalized margin allowed around the frame for fingertip points",
    )
    advanced_group.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.55,
        help="MediaPipe minimum detection confidence",
    )
    advanced_group.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum tracking confidence",
    )
    parser.set_defaults(require_distinct_handedness=False)
    advanced_group.add_argument(
        "--require-distinct-handedness",
        dest="require_distinct_handedness",
        action="store_true",
        help="require MediaPipe to label the two hands as different sides",
    )
    advanced_group.add_argument(
        "--allow-ambiguous-handedness",
        dest="require_distinct_handedness",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    return parser


def _normalize_effect_mode(value: str) -> str:
    return normalize_effect_mode(value)


def _format_help_epilog() -> str:
    lines = ["Effects:"]
    for mode in EFFECT_MODES:
        lines.append(f"  {mode:<9} {EFFECT_DESCRIPTIONS[mode]}")
    lines.append("")
    lines.extend(
        (
            "Examples:",
            "  python3 -m finger_quad_effect",
            "  python3 -m finger_quad_effect --preview --effect edge",
            "  python3 -m finger_quad_effect --effect blur --kernel 51 --sigma 7",
            "  python3 -m finger_quad_effect --effect mosaic --block-size 24",
            "  python3 -m finger_quad_effect --effect thermal --colormap turbo",
            "  python3 -m finger_quad_effect --effect noise --strength 0.5",
            '  python3 -m finger_quad_effect --effect outline --thickness 6 --color "#00ffff"',
            "",
            "Effect option notes:",
            "  blur: --kernel, --sigma",
            "  mosaic: --block-size",
            "  edge: --threshold-low, --threshold-high",
            "  thermal: --colormap",
            "  noise: --strength",
            "  outline: --thickness, --color",
            "  blur/mosaic/invert/grayscale/edge/thermal/noise: --feather",
        )
    )
    return "\n".join(lines)


def _parse_color_bgr(value: str) -> tuple[int, int, int]:
    colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "yellow": (0, 255, 255),
    }
    normalized = value.strip().lower()
    if normalized in colors:
        return colors[normalized]

    hex_value = normalized.removeprefix("#")
    if len(hex_value) != 6:
        raise argparse.ArgumentTypeError("color must be #RRGGBB or a basic color name")
    try:
        red = int(hex_value[0:2], 16)
        green = int(hex_value[2:4], 16)
        blue = int(hex_value[4:6], 16)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("color must be #RRGGBB or a basic color name") from exc
    return (blue, green, red)


if __name__ == "__main__":
    raise SystemExit(main())
