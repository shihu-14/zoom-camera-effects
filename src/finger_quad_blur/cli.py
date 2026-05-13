"""Command line entrypoint."""

from __future__ import annotations

import argparse

from .app import AppConfig, run_app
from .detection import DetectionConfig
from .doctor import run_doctor
from .effects import BlurConfig


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.doctor:
        return run_doctor(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )

    config = AppConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        preview=args.preview,
        virtual_camera=not args.no_virtual_camera,
        mirror=args.mirror,
        max_frames=args.max_frames,
        smoothing_factor=args.smoothing_factor,
        detection=DetectionConfig(
            min_hand_score=args.min_hand_score,
            min_point_score=args.min_point_score,
            point_bounds_margin=args.point_bounds_margin,
            require_distinct_handedness=args.require_distinct_handedness,
        ),
        blur=BlurConfig(
            mode=_normalize_effect_mode(args.effect),
            kernel_size=args.blur_kernel,
            edge_feather_px=args.edge_feather,
            mosaic_block_size=args.mosaic_block_size,
        ),
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    try:
        return run_app(config)
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply an effect inside the quadrilateral formed by both thumbs and index fingers."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--no-virtual-camera", action="store_true")
    parser.set_defaults(mirror=True)
    parser.add_argument(
        "--mirror",
        dest="mirror",
        action="store_true",
        help="mirror the camera image horizontally (default)",
    )
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="do not mirror the camera image horizontally",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--smoothing-factor", type=float, default=0.25)
    parser.add_argument(
        "--effect",
        choices=(
            "blur",
            "mosaic",
            "invert",
            "grayscale",
            "monochrome",
            "edge",
            "thermal",
            "noise",
            "outline",
            "particles",
        ),
        default="blur",
    )
    parser.add_argument("--blur-kernel", type=int, default=35)
    parser.add_argument("--mosaic-block-size", type=int, default=18)
    parser.add_argument("--edge-feather", type=int, default=3)
    parser.add_argument("--min-hand-score", type=float, default=0.55)
    parser.add_argument("--min-point-score", type=float, default=0.5)
    parser.add_argument("--point-bounds-margin", type=float, default=0.12)
    parser.add_argument("--min-detection-confidence", type=float, default=0.55)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.set_defaults(require_distinct_handedness=False)
    parser.add_argument(
        "--require-distinct-handedness",
        dest="require_distinct_handedness",
        action="store_true",
        help="require MediaPipe to label the two hands as different sides",
    )
    parser.add_argument(
        "--allow-ambiguous-handedness",
        dest="require_distinct_handedness",
        action="store_false",
        help="deprecated; ambiguous handedness is allowed by default",
    )
    return parser


def _normalize_effect_mode(value: str) -> str:
    return "grayscale" if value == "monochrome" else value


if __name__ == "__main__":
    raise SystemExit(main())
