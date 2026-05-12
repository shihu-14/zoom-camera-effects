# Finger Quad Blur

Real-time webcam filtering that blurs only the quadrilateral formed by both thumbs and index fingers. The processed stream can be sent to a virtual camera for use in Zoom, Teams, and other video meeting apps.

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

Send processed video to a virtual camera:

```bash
python3 -m finger_quad_blur
```

Preview without virtual camera output:

```bash
python3 -m finger_quad_blur --preview --no-virtual-camera
```

Run a short smoke test that opens the webcam, processes frames, sends them to
the virtual camera, then exits:

```bash
python3 -m finger_quad_blur --max-frames 30
```

## Gesture Behavior

- The selected effect activates only when exactly two hands are detected and both thumb tips and index fingertips are visible.
- If either hand or any required fingertip is lost, the current frame is output unchanged.
- The four fingertip points are sorted geometrically, so swapped hand positions and vertical movement still produce a stable polygon.
- The quadrilateral has no minimum size gate; any four valid fingertip points are accepted.
- No temporal hold is used; the blur disappears on the next processed frame after detection fails. At 30 FPS this is about 33 ms.

## Effects

Choose the area effect with `--effect`:

```bash
python3 -m finger_quad_blur --effect blur
python3 -m finger_quad_blur --effect mosaic
python3 -m finger_quad_blur --effect invert
python3 -m finger_quad_blur --effect grayscale
```

Tuning options:

```bash
python3 -m finger_quad_blur --effect blur --blur-kernel 51
python3 -m finger_quad_blur --effect mosaic --mosaic-block-size 24
```

## Zoom or Teams Setup

1. Install and enable the OS virtual camera backend.
2. Start this app with `python3 -m finger_quad_blur`.
3. In the meeting app, select the virtual camera named by the backend.

Use `--preview` to see the local processed feed while sending frames to the virtual camera.

## Troubleshooting

Check the runtime environment:

```bash
python3 -m finger_quad_blur --doctor
```

If virtual camera startup fails on macOS, open OBS once, choose
`Start Virtual Camera`, approve the system extension in System Settings, then
restart the app. You can still validate gesture tracking locally with:

```bash
python3 -m finger_quad_blur --preview --no-virtual-camera
```

If camera input fails, grant Camera access to Terminal or the Python executable
in `System Settings > Privacy & Security > Camera`.
