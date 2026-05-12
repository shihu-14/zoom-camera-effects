"""Command line entrypoint."""

from __future__ import annotations

import argparse

from .app import AppConfig, run_app
from .detection import DetectionConfig
from .effects import BlurConfig


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = AppConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        preview=args.preview,
        virtual_camera=not args.no_virtual_camera,
        mirror=args.mirror,
        detection=DetectionConfig(
            min_hand_score=args.min_hand_score,
            min_point_score=args.min_point_score,
            min_area_ratio=args.min_area_ratio,
            require_distinct_handedness=not args.allow_ambiguous_handedness,
        ),
        blur=BlurConfig(
            kernel_size=args.blur_kernel,
            edge_feather_px=args.edge_feather,
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
        description="Blur the quadrilateral formed by both thumbs and index fingers."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--no-virtual-camera", action="store_true")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--blur-kernel", type=int, default=35)
    parser.add_argument("--edge-feather", type=int, default=3)
    parser.add_argument("--min-hand-score", type=float, default=0.75)
    parser.add_argument("--min-point-score", type=float, default=0.5)
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--min-detection-confidence", type=float, default=0.75)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.75)
    parser.add_argument(
        "--allow-ambiguous-handedness",
        action="store_true",
        help="accept two detected hands even if MediaPipe reports the same hand label",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
