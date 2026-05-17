import numpy as np

from finger_quad_effect.effects import EffectConfig, apply_polygon_effect


def test_default_effect_is_blur():
    assert EffectConfig().mode == "blur"


def test_inactive_effect_returns_pixel_exact_copy():
    frame = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)

    output = apply_polygon_effect(frame, None, EffectConfig())

    assert np.array_equal(output, frame)
    assert output is not frame


def test_blur_changes_only_polygon_region():
    y_indices, x_indices = np.indices((80, 80))
    checker = ((x_indices + y_indices) % 2 * 255).astype(np.uint8)
    frame = np.dstack([checker, 255 - checker, checker])
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(kernel_size=21),
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

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="mosaic", mosaic_block_size=12),
    )

    assert np.array_equal(output[5, 5], frame[5, 5])
    assert not np.array_equal(output[40, 40], frame[40, 40])


def test_invert_changes_only_polygon_region():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="invert"),
    )

    assert np.array_equal(output[1, 1], frame[1, 1])
    assert np.array_equal(output[10, 10], np.array([245, 235, 225], dtype=np.uint8))


def test_grayscale_changes_only_polygon_region():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = (20, 80, 200)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="grayscale"),
    )

    assert np.array_equal(output[1, 1], frame[1, 1])
    assert output[10, 10, 0] == output[10, 10, 1] == output[10, 10, 2]
    assert not np.array_equal(output[10, 10], frame[10, 10])


def test_edge_changes_only_polygon_region():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    frame[:, 30:] = 255
    points = ((0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2))

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="edge"),
    )

    assert np.array_equal(output[5, 5], frame[5, 5])
    assert not np.array_equal(output[20:40, 20:40], frame[20:40, 20:40])


def test_edge_thresholds_change_edge_output():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    frame[:, 30:] = 255
    points = ((0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2))

    low_threshold = apply_polygon_effect(
        frame,
        points,
        EffectConfig(
            mode="edge",
            edge_low_threshold=1,
            edge_high_threshold=2,
        ),
    )
    high_threshold = apply_polygon_effect(
        frame,
        points,
        EffectConfig(
            mode="edge",
            edge_low_threshold=1000,
            edge_high_threshold=1200,
        ),
    )

    assert not np.array_equal(low_threshold, high_threshold)


def test_thermal_changes_only_polygon_region():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = (20, 80, 200)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="thermal"),
    )

    assert np.array_equal(output[1, 1], frame[1, 1])
    assert not np.array_equal(output[10, 10], frame[10, 10])


def test_thermal_colormap_changes_output():
    y_indices, x_indices = np.indices((20, 20))
    frame = np.dstack(
        [
            (x_indices * 10).astype(np.uint8),
            (y_indices * 10).astype(np.uint8),
            ((x_indices + y_indices) * 5).astype(np.uint8),
        ]
    )
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    jet = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="thermal", thermal_colormap="jet"),
    )
    turbo = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="thermal", thermal_colormap="turbo"),
    )

    assert not np.array_equal(jet[10, 10], turbo[10, 10])


def test_noise_changes_only_polygon_region():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="noise"),
    )

    assert np.array_equal(output[1, 1], frame[1, 1])
    assert not np.array_equal(output[10, 10], frame[10, 10])


def test_noise_strength_zero_keeps_frame_unchanged():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="noise", noise_strength=0),
    )

    assert np.array_equal(output, frame)


def test_outline_draws_black_polygon_border_only():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = (80, 120, 160)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="outline"),
    )

    assert np.array_equal(output[1, 1], frame[1, 1])
    assert np.array_equal(output[10, 10], frame[10, 10])
    assert np.array_equal(output[5, 5], np.array([0, 0, 0], dtype=np.uint8))


def test_outline_accepts_thickness_and_color():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = (80, 120, 160)
    points = ((0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25))

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(
            mode="outline",
            outline_thickness=1,
            outline_color_bgr=(255, 255, 0),
        ),
    )

    assert np.array_equal(output[5, 5], np.array([255, 255, 0], dtype=np.uint8))


def test_neon_changes_only_polygon_region():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    frame[:, 30:] = 255
    points = ((0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2))

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="neon"),
    )

    assert np.array_equal(output[5, 5], frame[5, 5])
    assert not np.array_equal(output[20:40, 20:40], frame[20:40, 20:40])


def test_glitch_changes_only_polygon_region_and_animates():
    y_indices, x_indices = np.indices((80, 80))
    frame = np.dstack(
        [
            (x_indices * 3).astype(np.uint8),
            (y_indices * 3).astype(np.uint8),
            ((x_indices + y_indices) * 2).astype(np.uint8),
        ]
    )
    points = ((0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2))

    first = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="glitch"),
        animation_phase=1.0,
    )
    second = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="glitch"),
        animation_phase=10.0,
    )

    assert np.array_equal(first[5, 5], frame[5, 5])
    assert not np.array_equal(first[20:60, 20:60], frame[20:60, 20:60])
    assert not np.array_equal(first, second)


def test_cartoon_changes_only_polygon_region():
    y_indices, x_indices = np.indices((50, 50))
    frame = np.dstack(
        [
            ((x_indices * 5) % 256).astype(np.uint8),
            ((y_indices * 5) % 256).astype(np.uint8),
            (((x_indices + y_indices) * 4) % 256).astype(np.uint8),
        ]
    )
    points = ((0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2))

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="cartoon"),
    )

    assert np.array_equal(output[5, 5], frame[5, 5])
    assert not np.array_equal(output[20:40, 20:40], frame[20:40, 20:40])


def test_sketch_changes_only_polygon_region():
    y_indices, x_indices = np.indices((50, 50))
    checker = ((x_indices + y_indices) % 2 * 255).astype(np.uint8)
    frame = np.dstack([checker, 255 - checker, checker])
    points = ((0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2))

    output = apply_polygon_effect(
        frame,
        points,
        EffectConfig(mode="sketch"),
    )

    assert np.array_equal(output[5, 5], frame[5, 5])
    assert not np.array_equal(output[20:40, 20:40], frame[20:40, 20:40])
