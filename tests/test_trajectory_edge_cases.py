from __future__ import annotations

import numpy as np
import pytest

import awesome_feature_navigation.trajectory as tr
from awesome_feature_navigation.line_detection import TapeObservation
from awesome_feature_navigation.trajectory import TapeFrameObservation, TapeMotionTerms, TrajectoryPoint


def _traj(points: np.ndarray, dt: float=1.0) -> list[TrajectoryPoint]:
    return [
        TrajectoryPoint(t=float(idx * dt), x=float(pt[0]), y=float(pt[1]), yaw=0.0)
        for idx, pt in enumerate(points)
    ]


def _record(t: float) -> TapeFrameObservation:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50:90, 48:53] = 255
    centerline = np.column_stack((np.full(40, 50.0), np.arange(50.0, 90.0))).astype(np.float32)
    obs = TapeObservation(
        centerline_px=centerline,
        angle_rad=0.0,
        bottom_x=50.0,
        shape_hw=(100, 100),
        mask=mask,
        centerline_mask=mask.copy(),
    )
    return TapeFrameObservation(
        t=t,
        dt=0.0 if t == 0.0 else 1.0,
        obs=obs,
        confidence=1.0,
        delta_yaw=0.0,
        delta_p=np.zeros(3, dtype=float),
    )


def test_anchor_resampling_alignment_and_loop_edge_cases() -> None:
    norm = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)

    assert tr._anchor_scores_from_normalized(norm, 'top_left').tolist() == pytest.approx([1.0, 1.0])
    assert tr._anchor_scores_from_normalized(norm, 'top_right').tolist() == pytest.approx([2.0, 0.0])
    assert tr._anchor_scores_from_normalized(norm, 'bottom_right').tolist() == pytest.approx([1.0, 1.0])
    assert tr._anchor_scores_from_normalized(norm, 'left').tolist() == pytest.approx([0.0, 1.0])
    assert tr._anchor_scores_from_normalized(norm, 'right').tolist() == pytest.approx([1.0, 0.0])
    assert tr._anchor_scores_from_normalized(norm, 'top').tolist() == pytest.approx([1.0, 0.0])
    assert tr._anchor_scores_from_normalized(norm, 'bottom').tolist() == pytest.approx([0.0, 1.0])
    assert tr._anchor_scores_from_normalized(norm, 'unknown').tolist() == pytest.approx([0.0, 2.0])
    assert tr._choose_loop_anchor(np.zeros((0, 2)), 'bottom_left') == 0

    assert tr._resample_polyline_by_arclength(np.zeros((0, 2)), 3).shape == (3, 2)
    np.testing.assert_allclose(
        tr._resample_polyline_by_arclength(np.array([[1.0, 2.0], [1.0, 2.0]], dtype=float), 2),
        [[1.0, 2.0], [1.0, 2.0]],
    )
    assert tr._resample_closed_polyline_by_arclength(np.zeros((0, 2)), 3).shape == (3, 2)
    np.testing.assert_allclose(
        tr._resample_closed_polyline_by_arclength(np.array([[1.0, 2.0]], dtype=float), 2),
        [[1.0, 2.0], [1.0, 2.0]],
    )
    np.testing.assert_allclose(
        tr._resample_closed_polyline_by_arclength(np.array([[1.0, 2.0], [1.0, 2.0]], dtype=float), 2),
        [[1.0, 2.0], [1.0, 2.0]],
    )

    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=float)
    shifted = np.roll(square * 2.0 + 3.0, 1, axis=0)
    shift, aligned, rmse, scale = tr._align_lap_to_template(shifted, square, allow_scale=True, phase_search_fraction=0.5)

    assert shift >= 0
    assert rmse < 1e-6
    assert scale == pytest.approx(0.5)
    np.testing.assert_allclose(aligned, square, atol=1e-6)
    assert tr._robust_keep_mask([0.1, 100.0], min_keep=2, sigma=0.0).tolist() == [True, True]
    assert tr._robust_keep_mask([1.0, 1.0, 1.0], min_keep=2, sigma=2.0).tolist() == [True, True, True]
    assert tr._search_anchor_boundaries(np.array([0.0, 1.0]), np.zeros((2, 2)), period=0.0, anchor='left', search_radius_fraction=0.1, min_gap_fraction=0.5) == []


def test_lap_extraction_and_direction_edge_cases() -> None:
    points = np.column_stack((np.linspace(0.0, 1.0, 20), np.zeros(20)))
    traj = _traj(points, dt=0.1)

    anchor_laps, anchor_splits = tr._extract_anchor_laps(traj, period=10.0, samples_per_lap=16, min_fraction=0.8, anchor='left', search_radius_fraction=0.1)
    assert anchor_laps == []
    assert anchor_splits.shape == (0,)
    manual_laps, manual_splits = tr._extract_manual_laps(traj, [0.2, 0.2001, 0.8], samples_per_lap=8)
    assert len(manual_laps) == 1
    assert manual_splits.tolist() == pytest.approx([0.2001, 0.8])
    periodic_laps, periodic_splits = tr._extract_periodic_laps(traj[:2], period=10.0, samples_per_lap=8, min_fraction=0.5)
    assert periodic_laps == []
    assert periodic_splits.shape == (0,)
    periodic_laps, _ = tr._extract_periodic_laps(traj, period=1.0, samples_per_lap=8, min_fraction=0.95)
    assert len(periodic_laps) == 1

    assert tr._normalize_lap_direction('any') == 'any'
    assert tr._resolve_manual_lap_directions({'manual_lap_direction': 'any'}, 2) == ['forward', 'forward']
    assert tr._resolve_loop_start_anchor({'loop_start_anchor': 'top_right'}) == 'top_right'
    assert tr._normalize_laps_for_common_direction(manual_laps, ['reverse'], {'loop_normalize_reverse_laps': False}) == manual_laps


def test_closed_path_degenerate_and_anchor_branches() -> None:
    triangle = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]], dtype=float)

    np.testing.assert_allclose(tr._fourier_smooth_closed_path(triangle, harmonics=0), triangle)
    np.testing.assert_allclose(tr._distribute_loop_closure_error(triangle[:1]), triangle[:1])
    np.testing.assert_allclose(tr._distribute_loop_closure_error(np.array([[0.0, 0.0], [0.0, 0.0]])), [[0.0, 0.0], [0.0, 0.0]])
    assert tr._sample_closed_catmull_rom(np.zeros((0, 2)), 4).shape == (4, 2)
    np.testing.assert_allclose(tr._sample_closed_catmull_rom(np.array([[1.0, 2.0]]), 3), [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    control, spline = tr._fit_closed_spline(triangle, control_count=6, sample_count=12, anchor='top_right', harmonics=2)
    assert control.shape == (6, 2)
    assert spline.shape == (12, 2)
    mean_control, mean_loop = tr._build_canonical_loop(np.stack((triangle, triangle + 0.1)), harmonics=2, anchor='top', statistic='mean', control_count=6)
    assert mean_control.shape == (6, 2)
    assert mean_loop.shape == (4, 2)
    assert tr._count_self_intersections(np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=float)) == 0
    with pytest.raises(ValueError, match='No representative'):
        tr._build_representative_lap_loop([], [], [], [], harmonics=2, anchor='', control_count=4, reference_period=1.0)
    projected, rmse = tr._project_points_to_closed_path(np.zeros((0, 2)), triangle, search_fraction=0.1)
    assert projected.shape == (0, 2)
    assert rmse == 0.0


def test_canonicalization_rejection_and_fallback_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    circle = np.column_stack(
        (
            np.cos(np.linspace(0.0, 4.0 * np.pi, 120, endpoint=False)),
            np.sin(np.linspace(0.0, 4.0 * np.pi, 120, endpoint=False)),
        )
    )
    traj = _traj(circle, dt=0.1)

    unchanged, debug = tr._canonicalize_periodic_trajectory(traj[:5], {'loop_period_sec': 1.0})
    assert debug is None
    assert len(unchanged) == 5
    unchanged, debug = tr._canonicalize_periodic_trajectory(traj, {'loop_period_sec': 1.0, 'manual_lap_bounds_sec': object()})
    assert debug is None or len(unchanged) > 0
    unchanged, debug = tr._canonicalize_periodic_trajectory(
        traj,
        {
            'loop_period_sec': 6.0,
            'manual_lap_bounds_sec': [0.0, 6.0, 12.0],
            'manual_lap_directions': ['forward', 'reverse'],
            'loop_average_direction': 'reverse',
        },
    )
    assert debug is None
    assert unchanged == traj

    original_representative = tr._build_representative_lap_loop

    def raise_representative(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        raise ValueError('candidate failed')

    monkeypatch.setattr(tr, '_build_representative_lap_loop', raise_representative)
    final, debug = tr._canonicalize_periodic_trajectory(
        traj,
        {
            'loop_period_sec': 6.0,
            'manual_lap_bounds_sec': [0.0, 6.0, 12.0],
            'loop_strategy': 'representative',
            'loop_max_alignment_rmse_ratio': 10.0,
            'loop_max_projection_rmse_ratio': 10.0,
        },
    )
    assert debug is not None
    assert len(final) > 0
    monkeypatch.setattr(tr, '_build_representative_lap_loop', original_representative)

    rejected, debug = tr._canonicalize_periodic_trajectory(
        traj,
        {
            'loop_period_sec': 6.0,
            'manual_lap_bounds_sec': [0.0, 6.0, 12.0],
            'loop_max_alignment_rmse_ratio': 0.0,
            'loop_max_projection_rmse_ratio': 0.0,
        },
    )
    assert debug is None
    assert rejected == traj


def test_speed_scale_solver_and_empty_offline_estimate_cover_full_branches() -> None:
    records = [_record(float(idx)) for idx in range(6)]
    terms = TapeMotionTerms(
        yaw_values=np.zeros(6, dtype=float),
        yaw_moves=np.zeros(5, dtype=float),
        lateral_steps_xy=np.tile(np.array([[0.0, 0.1]], dtype=float), (5, 1)),
    )
    reference = _traj(
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [0.1, 0.0],
            ],
            dtype=float,
        ),
    )

    scales = tr._solve_loop_speed_scales(
        records,
        terms,
        {
            'offline_variable_speed': True,
            'loop_start_anchor': 'manual_start',
            'offline_loop_closure_max_error_ratio': 1.0,
            'offline_speed_min_scale': 0.1,
            'offline_speed_max_scale': 2.0,
        },
        loop_period_sec=5.0,
        reference_traj=reference,
    )
    empty_estimate = tr._build_offline_tape_line_estimate([], {})

    assert scales.shape == (5,)
    assert np.all(scales >= 0.1)
    assert empty_estimate.line_valid_ratio == 0.0
    assert empty_estimate.diagnostics.t.shape == (0,)


def test_tracking_and_transform_failure_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    prev = np.zeros((20, 20), dtype=np.uint8)
    cur = np.zeros((20, 20), dtype=np.uint8)
    pts = np.array([[[5.0, 5.0]], [[10.0, 10.0]]], dtype=np.float32)

    empty_prev, empty_next = tr._track_features_forward_backward(prev, cur, None, {})
    assert empty_prev.shape == (0, 2)
    assert empty_next.shape == (0, 2)

    monkeypatch.setattr(tr.cv2, 'calcOpticalFlowPyrLK', lambda *args, **kwargs: (None, None, None))
    empty_prev, empty_next = tr._track_features_forward_backward(prev, cur, pts, {})
    assert empty_prev.shape == (0, 2)

    calls = {'count': 0}

    def fake_lk(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray, None] | tuple[None, None, None]:
        calls['count'] += 1
        if calls['count'] == 1:
            return (pts.copy(), np.ones((2, 1), dtype=np.uint8), None)
        return (None, None, None)

    monkeypatch.setattr(tr.cv2, 'calcOpticalFlowPyrLK', fake_lk)
    empty_prev, empty_next = tr._track_features_forward_backward(prev, cur, pts, {})
    assert empty_prev.shape == (0, 2)
    assert empty_next.shape == (0, 2)

    H, in_prev, in_next, ratio = tr._estimate_normalized_similarity_transform(
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        shape_hw=(20, 20),
        cfg={'vio_min_points_for_motion': 8},
    )
    assert H is None
    assert in_prev.shape == (0, 2)
    assert in_next.shape == (0, 2)
    assert ratio == 0.0

    monkeypatch.setattr(tr.cv2, 'estimateAffinePartial2D', lambda *args, **kwargs: (None, None))
    H, _, _, ratio = tr._estimate_normalized_similarity_transform(
        np.zeros((8, 2), dtype=np.float32),
        np.zeros((8, 2), dtype=np.float32),
        shape_hw=(20, 20),
        cfg={'vio_min_points_for_motion': 8},
    )
    assert H is None
    assert ratio == 0.0

    monkeypatch.setattr(
        tr.cv2,
        'estimateAffinePartial2D',
        lambda *args, **kwargs: (np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0]], dtype=float), np.ones((8, 1), dtype=np.uint8)),
    )
    H, _, _, ratio = tr._estimate_normalized_similarity_transform(
        np.column_stack((np.arange(8), np.arange(8))).astype(np.float32),
        np.column_stack((np.arange(8) + 10, np.arange(8))).astype(np.float32),
        shape_hw=(20, 20),
        cfg={'vio_min_points_for_motion': 8, 'vio_max_step_norm': 0.01},
    )
    assert H is not None
    assert np.linalg.norm(H[:2, 2]) <= 0.011
    assert ratio == 1.0
