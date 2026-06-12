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
    EFFECT_SCOPES,
    EffectConfig,
    normalize_area_points,
    normalize_effect_mode,
    normalize_effect_scope,
    parse_color_bgr,
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
        outline_fill_color_bgr = _parse_color_bgr(args.fill_color)
        area_points = _parse_area_points(args.area)
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
            scope=_normalize_effect_scope(args.scope),
            area_points=area_points,
            kernel_size=args.kernel,
            mosaic_block_size=args.block_size,
            edge_low_threshold=args.threshold_low,
            edge_high_threshold=args.threshold_high,
            thermal_colormap=args.colormap,
            noise_strength=args.strength,
            outline_thickness=args.thickness,
            outline_color_bgr=outline_color_bgr,
            outline_fill=args.fill,
            outline_fill_color_bgr=outline_fill_color_bgr,
        ),
        control_file=None if args.no_control else args.control_file,
        ui=args.ui,
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


class _EffectHelpParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        help_text = super().format_help()
        marker = "\nadvanced:\n"
        if marker not in help_text:
            return help_text
        before_advanced, advanced = help_text.split(marker, 1)
        advanced = advanced.rstrip()
        return f"{before_advanced.rstrip()}\n\n{_format_effect_help()}\n\nadvanced:\n{advanced}\n"


def _build_parser() -> argparse.ArgumentParser:
    default_effect = EffectConfig()
    parser = _EffectHelpParser(
        prog="python3 -m zoom_camera_effects",
        usage="%(prog)s [options]",
        description=(
            "Apply a camera effect to the finger quadrilateral, full frame, "
            "or partial area."
        ),
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
        help="show local preview only without virtual camera output",
    )
    run_group.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mirror the camera image horizontally",
    )
    run_group.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the on-frame runtime control overlay",
    )
    run_group.add_argument("--max-frames", type=int, help="process this many frames and exit")
    parser.add_argument(
        "--effect",
        choices=EFFECT_CHOICES,
        default=default_effect.mode,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--scope",
        choices=EFFECT_SCOPES,
        default=default_effect.scope,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--area",
        "--area-points",
        nargs="+",
        type=_parse_area_point,
        default=default_effect.area_points,
        metavar="X,Y",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--set-effect",
        choices=EFFECT_CHOICES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--kernel",
        dest="kernel",
        type=int,
        default=default_effect.kernel_size,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--blur-kernel",
        dest="kernel",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--block-size",
        dest="block_size",
        type=int,
        default=default_effect.mosaic_block_size,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mosaic-block-size",
        dest="block_size",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--threshold-low",
        type=float,
        default=default_effect.edge_low_threshold,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--threshold-high",
        type=float,
        default=default_effect.edge_high_threshold,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--colormap",
        choices=tuple(COLORMAPS),
        default=default_effect.thermal_colormap,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=default_effect.noise_strength,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=default_effect.outline_thickness,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--color",
        default="#000000",
        metavar="COLOR",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fill",
        action=argparse.BooleanOptionalAction,
        default=default_effect.outline_fill,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fill-color",
        default="#ffffff",
        metavar="COLOR",
        help=argparse.SUPPRESS,
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


def _normalize_effect_scope(value: str) -> str:
    return normalize_effect_scope(value)


def _parse_area_point(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("area point must use x,y")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("area point must use numeric x,y") from exc


def _parse_area_points(
    values: tuple[tuple[float, float], ...] | list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    try:
        return normalize_area_points(values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _format_effect_help() -> str:
    default_effect = EffectConfig()
    effect_choices = tuple(EFFECT_CHOICES)
    scope_choices = " | ".join(EFFECT_SCOPES)
    default_area = " ".join(f"{x:g},{y:g}" for x, y in default_effect.area_points)
    colormap_choices = tuple(COLORMAPS)
    lines = [
        "Syntax:",
        "  python3 -m zoom_camera_effects [run-option]... [effect-selection]",
        "                                 [effect-option]... [advanced-option]...",
        "",
        "  effect-selection:",
        "    [--effect {effect}]",
    ]
    lines.extend(_format_choice_lines("      effect: ", effect_choices, "              ", 7))
    lines.extend(
        (
        "    [--scope {scope}]",
        f"      scope: {scope_choices}",
        "    [--area {x,y x,y x,y [x,y]...}]",
        "    [--set-effect {effect}]",
        "",
        "Syntax rules:",
        "  [ ]   optional parameter",
        "  { }   mandatory value or choice when the option is used",
        "  |     choose one item",
        "  ...   repeatable parameter",
        "  x,y   lowercase variables are values you provide",
        "",
        "Effects:",
        )
    )
    for mode in EFFECT_MODES:
        lines.append(f"  {mode:<9} {EFFECT_DESCRIPTIONS[mode]} Use: --effect {mode}")
    lines.append("  monochrome Alias for grayscale. Use: --effect monochrome")
    lines.extend(
        (
            "",
            "Effect selection:",
            f"  [--effect {{effect}}]      default: {default_effect.mode}",
        )
    )
    lines.extend(
        _format_choice_lines("                          choices: ", effect_choices, "                                   ", 7)
    )
    lines.extend(
        (
            f"  [--scope {{scope}}]        default: {default_effect.scope}",
            f"                          choices: {scope_choices}",
            f"  [--area {{x,y x,y x,y [x,y]...}}]",
            f"                          default: {default_area}; 3..10 points, clockwise from top-left",
            "  [--set-effect {effect}]  switch the effect in a running app and exit",
            "",
            "Effect options:",
            "  none:",
            "    no extra options",
            "  blur:",
            f"    [--kernel {{kernel}}]                    default: {default_effect.kernel_size}",
            "  mosaic:",
            f"    [--block-size {{block-size}}]            default: {default_effect.mosaic_block_size}",
            "  invert:",
            "    no extra options",
            "  grayscale:",
            "    no extra options",
            "  edge:",
            f"    [--threshold-low {{threshold}}]          default: {default_effect.edge_low_threshold}",
            f"    [--threshold-high {{threshold}}]         default: {default_effect.edge_high_threshold}",
            "  thermal:",
            f"    [--colormap {{colormap}}]                default: {default_effect.thermal_colormap}",
        )
    )
    lines.extend(
        _format_choice_lines(
            "                                           choices: ",
            colormap_choices,
            "                                                    ",
            6,
        )
    )
    lines.extend(
        (
            "  noise:",
            f"    [--strength {{strength}}]                default: {default_effect.noise_strength}; range: 0..1",
            "  outline:",
            f"    [--thickness {{thickness}}]              default: {default_effect.outline_thickness} (auto)",
            "    [--color {color}]                      default: #000000",
            f"    [--fill | --no-fill]                   default: --{'fill' if default_effect.outline_fill else 'no-fill'}",
            "    [--fill-color {color}]                 default: #ffffff",
            "  neon:",
            f"    [--threshold-low {{threshold}}]          default: {default_effect.edge_low_threshold}",
            f"    [--threshold-high {{threshold}}]         default: {default_effect.edge_high_threshold}",
            "    [--color {color}]                      default: #000000 (uses cyan fallback)",
            f"    [--strength {{strength}}]                default: {default_effect.noise_strength}; range: 0..1",
            "  glitch:",
            f"    [--strength {{strength}}]                default: {default_effect.noise_strength}; range: 0..1",
            "  cartoon:",
            "    no extra options",
            "  sketch:",
            "    no extra options",
            "",
            "Examples:",
            "  python3 -m zoom_camera_effects",
            "  python3 -m zoom_camera_effects --preview --effect edge",
            "  python3 -m zoom_camera_effects --effect thermal --scope full",
            "  python3 -m zoom_camera_effects --scope partial "
            "--area 0.1,0.1 0.9,0.1 0.9,0.8 0.1,0.8",
            "  python3 -m zoom_camera_effects --effect outline --fill "
            "--fill-color '#202020'",
            "  python3 -m zoom_camera_effects --effect blur --kernel 51",
            "  python3 -m zoom_camera_effects --effect neon --color cyan --strength 0.9",
        )
    )
    return "\n".join(lines)


def _format_choice_lines(
    prefix: str,
    choices: tuple[str, ...],
    subsequent_indent: str,
    per_line: int,
) -> list[str]:
    chunks = [
        " | ".join(choices[index : index + per_line])
        for index in range(0, len(choices), per_line)
    ]
    if not chunks:
        return [prefix.rstrip()]
    lines = []
    for index, chunk in enumerate(chunks):
        line_prefix = prefix if index == 0 else subsequent_indent
        delimiter = " |" if index < len(chunks) - 1 else ""
        lines.append(line_prefix + chunk + delimiter)
    return lines


def _parse_color_bgr(value: str) -> tuple[int, int, int]:
    try:
        return parse_color_bgr(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
