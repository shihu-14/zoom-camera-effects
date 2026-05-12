"""Create macOS app launchers for Finger Quad Blur."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from string import Template


INFO_PLIST = Template(
    """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>${display_name}</string>
  <key>CFBundleExecutable</key>
  <string>${executable_name}</string>
  <key>CFBundleIdentifier</key>
  <string>${bundle_id}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>${display_name}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSCameraUsageDescription</key>
  <string>Finger Quad Blur uses the webcam to detect thumbs and index fingers for the selected privacy effect.</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
"""
)


LAUNCHER = Template(
    """#!/bin/zsh
set -e
export PATH="${python_dir}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
cd "${repo_root}"
exec "${python_executable}" -m finger_quad_blur ${args}
"""
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create macOS .app launchers.")
    parser.add_argument("--output-dir", default="dist")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    python_executable = Path(sys.executable).resolve()

    _create_app(
        output_dir=output_dir,
        repo_root=repo_root,
        python_executable=python_executable,
        app_name="Finger Quad Blur",
        bundle_id="jp.local.finger-quad-blur",
        launcher_args="",
    )
    _create_app(
        output_dir=output_dir,
        repo_root=repo_root,
        python_executable=python_executable,
        app_name="Finger Quad Blur Preview",
        bundle_id="jp.local.finger-quad-blur.preview",
        launcher_args="--preview --no-virtual-camera",
    )

    print(f"created app launchers in {output_dir}")
    return 0


def _create_app(
    *,
    output_dir: Path,
    repo_root: Path,
    python_executable: Path,
    app_name: str,
    bundle_id: str,
    launcher_args: str,
) -> None:
    app_root = output_dir / f"{app_name}.app"
    contents_dir = app_root / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    executable_name = "finger-quad-blur"

    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    (contents_dir / "Info.plist").write_text(
        INFO_PLIST.substitute(
            display_name=app_name,
            executable_name=executable_name,
            bundle_id=bundle_id,
        ),
        encoding="utf-8",
    )

    launcher_path = macos_dir / executable_name
    launcher_path.write_text(
        LAUNCHER.substitute(
            python_dir=str(python_executable.parent),
            repo_root=str(repo_root),
            python_executable=str(python_executable),
            args=launcher_args,
        ),
        encoding="utf-8",
    )
    launcher_path.chmod(launcher_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


if __name__ == "__main__":
    raise SystemExit(main())
