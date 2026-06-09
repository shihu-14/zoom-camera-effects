# Zoom Camera Effects

Real-time webcam filtering that applies a selected effect to an interactive
scope: the fingertip quadrilateral, the full frame, or an editable partial
area. The processed stream can be sent to a virtual camera for use in Zoom,
Teams, and other video meeting apps.

## Requirements

- Python 3.10+
- A webcam
- MediaPipe `0.10.21`, installed through this package
- A virtual camera backend supported by `pyvirtualcam`
  - macOS: OBS Virtual Camera is recommended
  - Linux: `v4l2loopback`
  - Windows: OBS Virtual Camera

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## Run

On macOS, create and launch the app bundle so Camera permission belongs to
Zoom Camera Effects instead of Terminal:

```bash
python3 scripts/create_macos_app.py
open "dist/Zoom Camera Effects.app"
```

The runtime control UI is drawn on top of the local video window by default.
Click the gear button in the upper-left corner to expand or collapse it. The
expanded panel switches effects and shows only the options for the active
effect. Disable it when needed:

```bash
python3 -m zoom_camera_effects --no-ui
```

Open the local preview without virtual camera output:

```bash
python3 -m zoom_camera_effects --preview
```

Command-line launch is still available for debugging, but macOS will attribute
webcam access to Terminal or Python:

```bash
python3 -m zoom_camera_effects
```

For Zoom self-view, avoid double mirroring by disabling app-side mirroring when
launching from the command line:

```bash
python3 -m zoom_camera_effects --no-mirror
```

Run a short smoke test that opens the webcam, processes frames, sends them to
the virtual camera, then exits:

```bash
python3 -m zoom_camera_effects --max-frames 30
```

## Scope Behavior

- `finger` applies the selected effect when exactly two hands provide thumb tips and index fingertips.
- `full` applies the selected effect to the entire frame.
- `partial` applies the selected effect to a fixed editable polygon area.
- If `finger` has no required fingertip data, the current frame is output unchanged.
- The fingertip points are sorted geometrically, so swapped hand positions and vertical movement still produce a stable polygon.
- The quadrilateral has no minimum size gate; any four valid fingertip points are accepted.
- No temporal hold is used; the effect disappears on the next processed frame after detection fails. At 30 FPS this is about 33 ms.

## Effects

The default effect is Gaussian blur. Choose another area effect with
`--effect`. Run `--help` to see every effect, tuning option, and default value:

```bash
python3 -m zoom_camera_effects --help
```

```bash
python3 -m zoom_camera_effects
python3 -m zoom_camera_effects --effect blur
python3 -m zoom_camera_effects --effect mosaic
python3 -m zoom_camera_effects --effect invert
python3 -m zoom_camera_effects --effect grayscale
python3 -m zoom_camera_effects --effect edge
python3 -m zoom_camera_effects --effect thermal
python3 -m zoom_camera_effects --effect noise
python3 -m zoom_camera_effects --effect outline
python3 -m zoom_camera_effects --effect neon
python3 -m zoom_camera_effects --effect glitch
python3 -m zoom_camera_effects --effect cartoon
python3 -m zoom_camera_effects --effect sketch
```

Scope examples:

```bash
python3 -m zoom_camera_effects --scope finger
python3 -m zoom_camera_effects --scope full
python3 -m zoom_camera_effects --scope partial --area 0.1,0.1 0.9,0.1 0.9,0.8 0.1,0.8
```

The camera image is mirrored horizontally by default so hand movement matches
the preview direction. Use `--no-mirror` if you need unmirrored output. The
Zoom app launcher uses `--no-mirror` because Zoom mirrors your own self-view.

Switch the effect while the app is running:

```bash
python3 -m zoom_camera_effects --set-effect thermal
python3 -m zoom_camera_effects --set-effect blur
```

The control UI is the preferred way to switch effects and tune options in real
time. Numeric options use sliders. `--set-effect` remains available for
terminal-driven changes.

Tuning options:

```bash
python3 -m zoom_camera_effects --effect blur --kernel 51
python3 -m zoom_camera_effects --effect mosaic --block-size 24
python3 -m zoom_camera_effects --effect edge --threshold-low 30 --threshold-high 90
python3 -m zoom_camera_effects --effect thermal --colormap turbo
python3 -m zoom_camera_effects --effect noise --strength 0.5
python3 -m zoom_camera_effects --effect outline --thickness 6 --color "#00ffff"
python3 -m zoom_camera_effects --effect outline --fill --fill-color "#202020"
python3 -m zoom_camera_effects --effect neon --color cyan --strength 0.9
python3 -m zoom_camera_effects --effect glitch --strength 0.7
```

For large quadrilaterals and hands spread far apart, the app accepts small
landmark overshoots, does not require MediaPipe's left/right labels to be
perfect, and smooths detected points over time. You can tune this behavior:

```bash
python3 -m zoom_camera_effects --point-bounds-margin 0.12 --smoothing-factor 0.25
python3 -m zoom_camera_effects --require-distinct-handedness
```

## Zoom or Teams Setup

1. Install and enable the OS virtual camera backend.
2. Create or refresh the macOS app launcher with `python3 scripts/create_macos_app.py`.
3. Start this app with `open "dist/Zoom Camera Effects.app"`.
4. In the meeting app, select the virtual camera named by the backend.

Use `--preview` to validate the local processed feed without virtual camera output.

## Troubleshooting

Check the runtime environment:

```bash
python3 -m zoom_camera_effects --doctor
```

If virtual camera startup fails on macOS, open OBS once, choose
`Start Virtual Camera`, approve the system extension in System Settings, then
restart the app. You can still validate gesture tracking locally with:

```bash
python3 -m zoom_camera_effects --preview
```

If camera input fails, grant Camera access to `Zoom Camera Effects` in
`System Settings > Privacy & Security > Camera`. For command-line debugging,
grant access to Terminal or the Python executable instead.
