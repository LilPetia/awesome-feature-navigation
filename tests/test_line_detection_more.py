from __future__ import annotations

import numpy as np
import pytest

from awesome_feature_navigation.line_detection import (
    LineDetector,
    _blend_angles,
    _fit_direction,
    _fit_local_direction,
    _measure_bottom_x,
    _normalize_hsv_ranges,
    _vector_to_angle,
    resolve_hsv_ranges,
    resolve_target_color,
)


def test_line_geometry_helpers_handle_degenerate_and_windowed_inputs() -> None:
    vertical = np.column_stack((np.full(20, 10.0), np.arange(20.0)))

    assert np.isnan(_fit_direction(np.zeros((1, 2), dtype=float)))
    assert _fit_direction(vertical) == pytest.approx(0.0, abs=1e-6)
    assert _vector_to_angle(np.array([0.0, -1.0], dtype=float)) == pytest.approx(0.0)
    assert np.isnan(_vector_to_angle(np.array([0.0, 0.0], dtype=float)))
    assert _fit_local_direction(vertical, fraction=0.2, min_points=3) == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(_fit_local_direction(vertical[:1], fraction=0.2, min_points=3))
    assert np.isnan(_fit_local_direction(vertical[:2], fraction=0.05, min_points=0))
    assert np.isnan(_fit_local_direction(np.zeros((3, 2), dtype=float), fraction=1.0, min_points=1))
    assert _measure_bottom_x(vertical, fraction=0.5, min_points=4, statistic='mean') == pytest.approx(10.0)
    assert _measure_bottom_x(vertical, fraction=0.5, min_points=4, statistic='median') == pytest.approx(10.0)
    assert _measure_bottom_x(vertical, fraction=0.0, min_points=4, statistic='median') == pytest.approx(10.0)
    assert _measure_bottom_x(vertical[:0], fraction=0.5, min_points=4, statistic='median') != _measure_bottom_x(vertical[:0], fraction=0.5, min_points=4, statistic='median')


def test_hsv_range_normalization_clips_and_ignores_bad_ranges() -> None:
    ranges = _normalize_hsv_ranges(
        [
            [200, -1, 30, 10, 300, 20],
            [1, 2],
        ]
    )

    assert len(ranges) == 1
    low, high = ranges[0]
    np.testing.assert_array_equal(low, [10, 0, 20])
    np.testing.assert_array_equal(high, [180, 255, 30])
    explicit = resolve_hsv_ranges({'hsv_ranges': [[130, 255, 255, 95, 100, 60]]})
    np.testing.assert_array_equal(explicit[0][0], [95, 100, 60])
    legacy_red = resolve_hsv_ranges({'target_color': 'red', 'hsv_red1_low': [0, 1, 2], 'hsv_red1_high': [3, 4, 5]})
    assert len(legacy_red) == 1
    assert resolve_target_color({'target_color': 'not-a-color'}) == 'blue'


def test_angle_blending_falls_back_for_nan_values() -> None:
    assert _blend_angles(float('nan'), 0.25, 0.5) == pytest.approx(0.25)
    assert _blend_angles(0.5, float('nan'), 0.5) == pytest.approx(0.5)
    assert _blend_angles(0.0, np.pi / 2.0, 0.5) == pytest.approx(np.pi / 4.0)
    assert _blend_angles(0.0, np.pi, 0.5) == pytest.approx(0.0)


def test_line_detector_internal_helpers_cover_smoothing_rendering_and_tuning() -> None:
    detector = LineDetector(resize_width=120)
    points = np.column_stack((np.arange(5.0), np.arange(5.0))).astype(np.float32)

    np.testing.assert_allclose(detector._smooth_centerline_points(points, smooth_window=0, width=10), points)
    smoothed = detector._smooth_centerline_points(points, smooth_window=2, width=10)
    assert smoothed.shape == points.shape

    empty_line = detector._render_centerline((8, 8), np.zeros((0, 2), dtype=np.float32))
    single_line = detector._render_centerline((8, 8), np.array([[3.0, 4.0]], dtype=np.float32))
    assert np.count_nonzero(empty_line) == 0
    assert single_line[4, 3] == 255

    ranges = resolve_hsv_ranges({'target_color': 'blue'})
    hsv_empty = np.zeros((4, 4, 3), dtype=np.uint8)
    assert detector._auto_tune_ranges(hsv_empty, ranges, 'blue', 2, 2) == ranges

    hsv_sparse = np.zeros((20, 20, 3), dtype=np.uint8)
    hsv_sparse[10, 10] = [110, 180, 200]
    sparse_ranges = detector._auto_tune_ranges(hsv_sparse, ranges, 'blue', 0, 20)
    np.testing.assert_array_equal(sparse_ranges[0][0], ranges[0][0])

    hsv_dense = np.zeros((30, 30, 3), dtype=np.uint8)
    hsv_dense[:, :] = [110, 180, 200]
    dense_ranges = detector._auto_tune_ranges(hsv_dense, ranges, 'blue', 0, 30)
    assert dense_ranges[0][0][0] >= ranges[0][0][0]
    assert dense_ranges[0][1][0] <= ranges[0][1][0]


def test_line_detector_processes_blue_line_and_reuses_previous_centerline() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[50:90, 58:63] = [255, 0, 0]
    detector = LineDetector(resize_width=120)

    obs = detector.process(
        frame,
        {
            'target_color': 'blue',
            'line_angle_mode': 'bottom_segment',
            'centerline_smooth_window': 3,
        },
    )

    assert obs.centerline_px.shape[0] >= detector.min_pixels
    assert obs.bottom_x == pytest.approx(60.0, abs=2.0)
    assert np.isfinite(obs.angle_rad)
    assert np.count_nonzero(obs.centerline_mask) > 0

    pca = detector.process(frame, {'target_color': 'blue'})
    assert np.isfinite(pca.angle_rad)

    empty = np.zeros_like(frame)
    fallback = detector.process(empty, {'target_color': 'blue'})

    assert fallback.centerline_px.shape == (0, 2)
    assert np.count_nonzero(fallback.centerline_mask) > 0


def test_line_detector_blend_mode_and_white_auto_tune() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[52:88, 50:70] = [240, 240, 240]
    detector = LineDetector(resize_width=60)

    obs = detector.process(
        frame,
        {
            'target_color': 'white',
            'auto_color_tune': True,
            'line_angle_mode': 'blend',
            'line_angle_local_weight': 0.5,
            'line_bottom_window_fraction': 0.5,
            'line_bottom_window_statistic': 'mean',
        },
    )

    assert obs.shape_hw == (50, 60)
    assert obs.centerline_px.shape[0] > 0
    assert np.isfinite(obs.bottom_x)
