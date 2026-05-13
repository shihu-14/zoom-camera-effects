import numpy as np

from finger_quad_blur.effects import BlurConfig, apply_polygon_blur


def test_default_effect_is_blur():
    assert BlurConfig().mode == "blur"


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


def test_mosaic_changes_only_polygon_region():
    y_indices, x_indices = np.indices((80, 80))
    frame = np.dstack(
        [
            (x_indices * 3).astype(np.uint8),
            (y_indices * 3).astype(np.uint8),
            ((x_indices + y_indices) * 2).astype(np.uint8),
        ]
    )
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_blur(
        frame,
        points,
        BlurConfig(mode="mosaic", mosaic_block_size=12, edge_feather_px=0),
    )

    assert np.array_equal(output[5, 5], frame[5, 5])
    assert not np.array_equal(output[40, 40], frame[40, 40])


def test_invert_changes_only_polygon_region():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_blur(
        frame,
        points,
        BlurConfig(mode="invert", edge_feather_px=0),
    )

    assert np.array_equal(output[1, 1], frame[1, 1])
    assert np.array_equal(output[10, 10], np.array([245, 235, 225], dtype=np.uint8))


def test_grayscale_changes_only_polygon_region():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = (20, 80, 200)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_blur(
        frame,
        points,
        BlurConfig(mode="grayscale", edge_feather_px=0),
    )

    assert np.array_equal(output[1, 1], frame[1, 1])
    assert output[10, 10, 0] == output[10, 10, 1] == output[10, 10, 2]
    assert not np.array_equal(output[10, 10], frame[10, 10])


def test_edge_changes_only_polygon_region():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    frame[:, 30:] = 255
    points = ((0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2))

    output = apply_polygon_blur(
        frame,
        points,
        BlurConfig(mode="edge", edge_feather_px=0),
    )

    assert np.array_equal(output[5, 5], frame[5, 5])
    assert not np.array_equal(output[20:40, 20:40], frame[20:40, 20:40])


def test_thermal_changes_only_polygon_region():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = (20, 80, 200)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_blur(
        frame,
        points,
        BlurConfig(mode="thermal", edge_feather_px=0),
    )

    assert np.array_equal(output[1, 1], frame[1, 1])
    assert not np.array_equal(output[10, 10], frame[10, 10])


def test_noise_changes_only_polygon_region():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_blur(
        frame,
        points,
        BlurConfig(mode="noise", edge_feather_px=0),
    )

    assert np.array_equal(output[1, 1], frame[1, 1])
    assert not np.array_equal(output[10, 10], frame[10, 10])


def test_outline_fill_changes_only_polygon_region():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = (80, 120, 160)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_blur(
        frame,
        points,
        BlurConfig(mode="outline-fill", edge_feather_px=0),
    )

    assert np.array_equal(output[1, 1], frame[1, 1])
    assert np.array_equal(output[10, 10], np.array([24, 24, 24], dtype=np.uint8))
