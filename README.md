# Finger Quad Effect

Real-time webcam filtering that applies a selected effect only inside the quadrilateral formed by both thumbs and index fingers. The processed stream can be sent to a virtual camera for use in Zoom, Teams, and other video meeting apps.

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
Finger Quad Effect instead of Terminal:

```bash
python3 scripts/create_macos_app.py
open "dist/Finger Quad Effect.app"
```

Open the preview window without virtual camera output:

```bash
python3 -m finger_quad_effect --preview --no-virtual-camera
```

Command-line launch is still available for debugging, but macOS will attribute
webcam access to Terminal or Python:

```bash
python3 -m finger_quad_effect
```

For Zoom self-view, avoid double mirroring by disabling app-side mirroring when
launching from the command line:

```bash
python3 -m finger_quad_effect --no-mirror
```

Run a short smoke test that opens the webcam, processes frames, sends them to
the virtual camera, then exits:

```bash
python3 -m finger_quad_effect --max-frames 30
```

## Gesture Behavior

- The selected effect activates only when exactly two hands are detected and both thumb tips and index fingertips are visible.
- If either hand or any required fingertip is lost, the current frame is output unchanged.
- The four fingertip points are sorted geometrically, so swapped hand positions and vertical movement still produce a stable polygon.
- The quadrilateral has no minimum size gate; any four valid fingertip points are accepted.
- No temporal hold is used; the effect disappears on the next processed frame after detection fails. At 30 FPS this is about 33 ms.

## Effects

The default effect is Gaussian blur. Choose another area effect with
`--effect`. Print the available effects with `--list-effects` or
`--help-effects`:

```bash
python3 -m finger_quad_effect --list-effects
```

```bash
python3 -m finger_quad_effect
python3 -m finger_quad_effect --effect blur
python3 -m finger_quad_effect --effect mosaic
python3 -m finger_quad_effect --effect invert
python3 -m finger_quad_effect --effect grayscale
python3 -m finger_quad_effect --effect edge
python3 -m finger_quad_effect --effect thermal
python3 -m finger_quad_effect --effect noise
python3 -m finger_quad_effect --effect outline
python3 -m finger_quad_effect --effect particles
```

The `particles` effect uses MediaPipe's relative landmark depth to estimate the
fingertip plane equation and emits animated particles from the quadrilateral.

The camera image is mirrored horizontally by default so hand movement matches
the preview direction. Use `--no-mirror` if you need unmirrored output. The
Zoom app launcher uses `--no-mirror` because Zoom mirrors your own self-view.

Switch the effect while the app is running:

```bash
python3 -m finger_quad_effect --set-effect particles
python3 -m finger_quad_effect --set-effect thermal
python3 -m finger_quad_effect --set-effect blur
```

Tuning options:

```bash
python3 -m finger_quad_effect --effect blur --blur-kernel 51
python3 -m finger_quad_effect --effect mosaic --mosaic-block-size 24
```

For large quadrilaterals and hands spread far apart, the app accepts small
landmark overshoots, does not require MediaPipe's left/right labels to be
perfect, and smooths detected points over time. You can tune this behavior:

```bash
python3 -m finger_quad_effect --point-bounds-margin 0.12 --smoothing-factor 0.25
python3 -m finger_quad_effect --require-distinct-handedness
```

## Zoom or Teams Setup

1. Install and enable the OS virtual camera backend.
2. Create or refresh the macOS app launcher with `python3 scripts/create_macos_app.py`.
3. Start this app with `open "dist/Finger Quad Effect.app"`.
4. In the meeting app, select the virtual camera named by the backend.

Use `--preview` to see the local processed feed while sending frames to the virtual camera.

## Troubleshooting

Check the runtime environment:

```bash
python3 -m finger_quad_effect --doctor
```

If virtual camera startup fails on macOS, open OBS once, choose
`Start Virtual Camera`, approve the system extension in System Settings, then
restart the app. You can still validate gesture tracking locally with:

```bash
python3 -m finger_quad_effect --preview --no-virtual-camera
```

If camera input fails, grant Camera access to `Finger Quad Effect` in
`System Settings > Privacy & Security > Camera`. For command-line debugging,
grant access to Terminal or the Python executable instead.
