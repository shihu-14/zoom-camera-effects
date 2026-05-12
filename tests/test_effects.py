import numpy as np

from finger_quad_blur.effects import BlurConfig, apply_polygon_blur


def test_inactive_blur_returns_pixel_exact_copy():
    frame = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)

    output = apply_polygon_blur(frame, None, BlurConfig())

    assert np.array_equal(output, frame)
    assert output is not frame


def test_blur_changes_only_polygon_region():
    y_indices, x_indices = np.indices((80, 80))
    checker = ((x_indices + y_indices) % 2 * 255).astype(np.uint8)
    frame = np.dstack([checker, 255 - checker, checker])
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_blur(
        frame,
        points,
        BlurConfig(kernel_size=21, edge_feather_px=0),
    )

    assert np.array_equal(output[5, 5], frame[5, 5])
    assert not np.array_equal(output[40, 40], frame[40, 40])
