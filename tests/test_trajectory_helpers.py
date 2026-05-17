from __future__ import annotations

import numpy as np
import pytest

import awesome_feature_navigation.trajectory as tr
from awesome_feature_navigation.line_detection import TapeObservation
from awesome_feature_navigation.trajectory import (
    LoopAveragingDebug,
    LoopLapDebug,
    TapeFrameObservation,
    TrajectoryEstimateResult,
    TrajectoryPoint,
)


def _traj(points: np.ndarray, dt: float=1.0) -> list[TrajectoryPoint]:
    return [
        TrajectoryPoint(t=float(idx * dt), x=float(pt[0]), y=float(pt[1]), yaw=0.0)
        for idx, pt in enumerate(points)
    ]


def _tape_observation(angle: float=0.0, bottom_x: float=50.0, points: int=40) -> TapeObservation:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50:90, 48:53] = 255
    centerline = np.column_stack((np.full(points, bottom_x, dtype=float), np.linspace(50.0, 89.0, points)))
    centerline_mask = np.zeros_like(mask)
    centerline_mask[50:90, 50] = 255
    shape_hw = (int(mask.shape[0]), int(mask.shape[1]))
    return TapeObservation(
        centerline_px=centerline.astype(np.float32),
        angle_rad=angle,
        bottom_x=bottom_x,
        shape_hw=shape_hw,
        mask=mask,
        centerline_mask=centerline_mask,
    )


def _record(t: float, angle: float=0.0, bottom_x: float=50.0, confidence: float=1.0) -> TapeFrameObservation:
    return TapeFrameObservation(
        t=t,
        dt=0.0 if t == 0.0 else 1.0,
        obs=_tape_observation(angle=angle, bottom_x=bottom_x),
        confidence=confidence,
        delta_yaw=0.0,
        delta_p=np.array([0.1, 0.0, 0.0], dtype=float),
    )


def test_config_geometry_alignment_and_resampling_helpers() -> None:
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    shifted = square + np.array([2.0, -1.0])

    assert tr._clamp(10.0, 0.0, 1.0) == 1.0
    assert tr._cfg_float({'value': 'bad'}, ['missing', 'value'], 2.0) == 2.0
    assert tr._cfg_bool_any({'flag': 'off'}, ['flag'], default=True) is False
    assert tr._cfg_bool_any({'flag': 1}, ['flag']) is True
    assert tr._build_imu_preintegration_params({'imu_gravity_mps2': 9.7}) is None
    assert tr._blend_angles(float('nan'), 0.5, 0.5) == pytest.approx(0.5)
    np.testing.assert_allclose(tr._rigid_align_points(shifted, square), square, atol=1e-6)
    aligned, scale = tr._similarity_align_points(square * 2.0, square)
    assert scale == pytest.approx(0.5)
    np.testing.assert_allclose(aligned, square, atol=1e-6)
    assert tr._lap_rmse(square, square) == pytest.approx(0.0)
    assert tr._choose_loop_anchor(square, 'bottom_left') == 0
    assert tr._robust_keep_mask([0.1, 0.11, 10.0], min_keep=2, sigma=1.0).tolist() == [True, True, False]
    assert tr._anchor_scores(np.zeros((0, 2)), 'bottom_left').shape == (0,)
    assert tr._resample_polyline_by_arclength(square[:1], 3).shape == (3, 2)
    assert tr._resample_closed_polyline_by_arclength(square, 8).shape == (8, 2)


def test_lap_extraction_direction_and_canonical_loop_helpers() -> None:
    samples = []
    for idx in range(80):
        phase = 2.0 * np.pi * idx / 80.0
        samples.append([np.cos(phase), np.sin(phase)])
    traj = _traj(np.asarray(samples, dtype=float), dt=0.025)

    manual_laps, manual_splits = tr._extract_manual_laps(traj, [0.0, 1.0, 2.0], samples_per_lap=32)
    periodic_laps, periodic_splits = tr._extract_periodic_laps(traj, period=1.0, samples_per_lap=32, min_fraction=0.4)

    assert len(manual_laps) == 2
    assert manual_splits.tolist() == pytest.approx([0.0, 1.0, 1.975])
    assert len(periodic_laps) == 2
    assert periodic_splits[0] == pytest.approx(0.0)
    assert tr._normalize_lap_direction('backwards') == 'reverse'
    assert tr._resolve_manual_lap_directions({'manual_lap_directions': ['forward', 'reverse']}, 3) == ['forward', 'reverse', 'forward']
    assert tr._resolve_loop_start_anchor({'loop_start_anchor': 'manual_start'}) == ''
    normalized = tr._normalize_laps_for_common_direction(manual_laps, ['forward', 'reverse'], {'loop_normalize_reverse_laps': True})
    np.testing.assert_allclose(normalized[1][3][0], manual_laps[1][3][-1])


def test_closed_loop_spline_projection_and_representative_selection() -> None:
    circle = np.column_stack(
        (
            np.cos(np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)),
            np.sin(np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)),
        )
    )
    laps = [
        (0, 0.0, 1.0, circle),
        (1, 1.0, 2.0, circle + np.array([0.02, -0.01])),
    ]
    aligned_laps = [tr._center_points(lap[3]) for lap in laps]

    smoothed = tr._fourier_smooth_closed_path(circle, harmonics=8)
    control, loop = tr._build_canonical_loop(np.stack(aligned_laps), harmonics=8, anchor='', statistic='median', control_count=12)
    rep_control, rep_loop = tr._build_representative_lap_loop(
        laps=laps,
        aligned_laps=aligned_laps,
        alignment_rmses=[0.1, 0.2],
        keep_mask=[True, True],
        harmonics=8,
        anchor='',
        control_count=12,
        reference_period=1.0,
    )
    projected, rmse = tr._project_points_to_closed_path(loop[:-1], loop[:-1], search_fraction=0.1)
    loop_traj = tr._loop_xy_to_trajectory(loop, period=2.0)

    assert smoothed.shape == circle.shape
    assert control.shape == (12, 2)
    assert loop.shape[0] == 65
    assert rep_control.shape == (12, 2)
    assert rep_loop.shape[0] == 65
    assert projected.shape == circle.shape
    assert rmse < 0.2
    assert loop_traj[-1].t == pytest.approx(2.0)
    assert tr._segments_strictly_intersect(np.array([0, 0]), np.array([1, 1]), np.array([0, 1]), np.array([1, 0])) is True
    assert tr._count_self_intersections(np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)) == 1


def test_tape_smoothing_confidence_and_motion_terms() -> None:
    records = [
        _record(0.0, angle=0.0, bottom_x=50.0),
        _record(1.0, angle=0.2, bottom_x=60.0),
        _record(2.0, angle=float('nan'), bottom_x=float('nan'), confidence=0.0),
    ]
    smoother = tr.VisionObservationSmoother(window_frames=3)

    first = smoother.update(_tape_observation(angle=0.1, bottom_x=52.0))
    second = smoother.update(_tape_observation(angle=0.2, bottom_x=54.0))
    confidence = tr._line_observation_confidence(_tape_observation(), {'line_confidence_min_points': 8})
    terms = tr._compute_tape_motion_terms(
        records,
        angles=np.array([0.0, 0.2, 0.0], dtype=float),
        bottom_xs=np.array([50.0, 60.0, 50.0], dtype=float),
        weights=np.array([1.0, 1.0, 0.0], dtype=float),
        cfg={'vision_yaw_gain': 1.0, 'vision_yaw_max_correction': 0.5, 'vision_lateral_gain': 0.01},
    )
    traj, speeds = tr._trajectory_from_tape_motion_terms(records, terms, {'forward_speed_mps': 1.0}, speed_scales=np.array([1.0, 0.5]))

    assert first.bottom_x == pytest.approx(52.0)
    assert second.bottom_x > 52.0
    assert confidence > 0.5
    assert tr._centered_weighted_average(np.array([1.0, 3.0]), np.array([1.0, 1.0]), 3).tolist() == pytest.approx([2.0, 2.0])
    assert np.isfinite(tr._centered_weighted_angle_average(np.array([0.0, np.pi / 2.0]), np.ones(2), 3)).all()
    assert tr._median_dt(np.array([0.0, 0.1, 0.2])) == pytest.approx(0.1)
    assert tr._make_odd_window(4) == 5
    assert tr._resolve_smoothing_window_frames(np.array([0.0, 0.5, 1.0]), {'offline_line_smoothing_frames': 2}) == 3
    assert tr._adaptive_confidence_threshold(np.array([0.0, 0.0]), {'offline_line_min_confidence': 0.2}) == pytest.approx(0.2)
    assert len(traj) == 3
    assert speeds[-1] == pytest.approx(0.5)


def test_loop_boundary_speed_soft_closure_and_output_transforms() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.1, 0.1]], dtype=float)
    traj = _traj(points, dt=1.0)
    records = [_record(float(idx)) for idx in range(len(traj))]
    terms = tr.TapeMotionTerms(
        yaw_values=np.zeros(len(records), dtype=float),
        yaw_moves=np.zeros(len(records) - 1, dtype=float),
        lateral_steps_xy=np.zeros((len(records) - 1, 2), dtype=float),
    )

    assert tr._trajectory_segment_length(points, 0, 2) == pytest.approx(2.0)
    assert tr._loop_boundary_indices(traj, 1.0, {'loop_start_anchor': 'bottom_left'}) != []
    scales = tr._solve_loop_speed_scales(records, terms, {'offline_variable_speed': True}, 1.0, traj)
    closed = tr._apply_soft_loop_closure(
        traj,
        {'offline_soft_loop_closure': True, 'offline_loop_closure_max_error_ratio': 1.0},
        3.0,
    )
    transformed = tr._apply_output_transform_to_trajectory(
        closed,
        {'trajectory_output_flip_x': True, 'trajectory_output_rotation_deg': 90.0},
    )

    assert scales.shape == (len(records) - 1,)
    assert closed[-1].x != pytest.approx(traj[-1].x) or closed[-1].y != pytest.approx(traj[-1].y)
    assert transformed[1].y == pytest.approx(-closed[1].x)
    assert tr._transform_xy_array(np.array([[1.0, 2.0]]), {'trajectory_output_flip_y': True}).tolist() == [[1.0, -2.0]]
    assert tr._transform_yaw(float('nan'), {}) != tr._transform_yaw(float('nan'), {})


def test_loop_debug_and_result_output_transform() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    debug = LoopAveragingDebug(
        period_sec=1.0,
        samples_per_lap=2,
        laps=[
            LoopLapDebug(
                lap_index=0,
                t0=0.0,
                t1=1.0,
                direction='forward',
                raw_xy=xy,
                aligned_xy=xy,
                projected_xy=xy,
                phase_shift=0,
                scale=1.0,
                alignment_rmse=0.0,
                projection_rmse=0.0,
                kept=True,
            )
        ],
        canonical_xy=xy,
        spline_control_xy=xy,
        split_times=np.array([0.0, 1.0], dtype=float),
    )
    result = TrajectoryEstimateResult(
        raw_traj=_traj(xy),
        smoothed_traj=_traj(xy),
        final_traj=_traj(xy),
        loop_debug=debug,
        mode='tape_line',
    )

    transformed = tr._apply_output_transform_to_result(result, {'trajectory_output_flip_x': True})

    assert transformed.final_traj[1].x == pytest.approx(-1.0)
    assert transformed.loop_debug is not None
    assert transformed.loop_debug.canonical_xy[1, 0] == pytest.approx(-1.0)


def test_visual_odometry_helpers_with_synthetic_features() -> None:
    gray = np.zeros((80, 100), dtype=np.uint8)
    gray[20:60, 30:70] = 180
    descriptor = tr._make_frame_descriptor(gray)
    pts = np.array([[[20.0, 20.0]], [[50.0, 20.0]], [[20.0, 50.0]], [[50.0, 50.0]], [[35.0, 35.0]], [[65.0, 35.0]], [[35.0, 65.0]], [[65.0, 65.0]]], dtype=np.float32)
    moved = pts.reshape(-1, 2) + np.array([1.0, 0.0], dtype=np.float32)

    H, inlier_prev, inlier_next, inlier_ratio = tr._estimate_normalized_similarity_transform(
        pts.reshape(-1, 2),
        moved,
        shape_hw=(int(gray.shape[0]), int(gray.shape[1])),
        cfg={'vio_min_points_for_motion': 4},
    )
    period = tr._estimate_loop_period_from_descriptors(
        [float(idx) for idx in range(30)],
        [descriptor if idx % 5 == 0 else np.roll(descriptor, idx) for idx in range(30)],
        {'auto_loop_min_period_sec': 4.0, 'auto_loop_max_period_sec': 6.0, 'auto_loop_min_score': -1.0, 'auto_loop_min_score_gain': -1.0},
    )

    assert descriptor.shape == (32 * 24,)
    assert H is not None
    assert inlier_prev.shape[0] >= 4
    assert inlier_next.shape[0] >= 4
    assert inlier_ratio > 0.0
    assert period is not None
    assert tr._resolve_trajectory_mode({'trajectory_mode': 'generic'}, None) == 'generic_vio'
    assert tr._resolve_trajectory_mode({'trajectory_mode': 'auto'}, [tr.IMUSample(0.0, np.zeros(3), np.zeros(3))]) == 'generic_vio'
    assert tr._generic_vio_result_is_usable(TrajectoryEstimateResult(_traj(np.array([[0.0, 0.0], [1.0, 0.0]] * 5)), [], [], None), {}) is True
    assert tr._tape_line_result_is_usable(TrajectoryEstimateResult(_traj(np.array([[0.0, 0.0], [1.0, 0.0]] * 5)), [], [], None, line_valid_ratio=0.5), {}) is True
    assert tr._resize_frame_keep_aspect(np.zeros((10, 20, 3), dtype=np.uint8), 10).shape[:2] == (5, 10)
