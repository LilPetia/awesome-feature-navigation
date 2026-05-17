import numpy as np
import pytest

from awesome_feature_navigation.line_detection import TapeObservation
from awesome_feature_navigation.trajectory import (
    TapeFrameObservation,
    TrajectoryPoint,
    _adaptive_confidence_threshold,
    _apply_output_transform_to_trajectory,
    _build_offline_tape_line_estimate,
    _build_offline_tape_line_trajectory,
    _canonicalize_periodic_trajectory,
    _distribute_loop_closure_error,
    _line_observation_confidence,
    _resolve_smoothing_window_frames,
)


def _observation(angle_rad: float, bottom_x: float=50.0) -> TapeObservation:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50:90, 48:53] = 255
    centerline_mask = np.zeros_like(mask)
    centerline_mask[50:90, 50] = 255
    centerline = np.column_stack(
        (
            np.full(40, bottom_x, dtype=np.float32),
            np.arange(50, 90, dtype=np.float32),
        )
    )
    return TapeObservation(
        centerline_px=centerline,
        angle_rad=angle_rad,
        bottom_x=bottom_x,
        shape_hw=(100, 100),
        mask=mask,
        centerline_mask=centerline_mask,
    )


def _record(t: float, angle_rad: float, confidence: float) -> TapeFrameObservation:
    return TapeFrameObservation(
        t=t,
        dt=0.0 if t == 0.0 else 1.0,
        obs=_observation(angle_rad=angle_rad),
        confidence=confidence,
        delta_yaw=0.0,
        delta_p=np.zeros(3, dtype=float),
    )


def test_line_observation_confidence_is_positive_for_clean_centerline() -> None:
    confidence = _line_observation_confidence(_observation(angle_rad=0.0), {})

    assert confidence > 0.5


def test_offline_tape_line_smoothing_ignores_low_confidence_angle_outlier() -> None:
    records = [
        _record(0.0, 0.0, 1.0),
        _record(1.0, 0.0, 1.0),
        _record(2.0, 1.5, 0.0),
        _record(3.0, 0.0, 1.0),
        _record(4.0, 0.0, 1.0),
    ]
    cfg = {
        'forward_speed_mps': 1.0,
        'vision_yaw_gain': 1.0,
        'vision_yaw_max_correction': 1.0,
        'vision_lateral_gain': 0.0,
        'offline_tape_smoothing': True,
        'offline_line_smoothing_frames': 5,
        'offline_line_min_confidence': 0.1,
    }

    trajectory, valid_ratio = _build_offline_tape_line_trajectory(records, cfg)

    assert valid_ratio == pytest.approx(0.8)
    assert trajectory[-1].x == pytest.approx(4.0)
    assert trajectory[-1].y == pytest.approx(0.0, abs=1e-9)
    assert trajectory[-1].yaw == pytest.approx(0.0, abs=1e-9)


def test_time_based_smoothing_window_uses_frame_timestamps() -> None:
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)

    assert _resolve_smoothing_window_frames(times, {'offline_line_smoothing_sec': 0.5}) == 5


def test_adaptive_confidence_threshold_uses_video_distribution() -> None:
    confidences = np.array([0.05, 0.2, 0.8, 0.9, 1.0], dtype=float)

    threshold = _adaptive_confidence_threshold(
        confidences,
        {
            'offline_line_min_confidence': 0.15,
            'offline_adaptive_confidence': True,
        },
    )

    assert threshold > 0.15
    assert threshold <= 0.6


def test_offline_tape_line_estimate_separates_raw_smoothed_and_final_outputs() -> None:
    records = [
        _record(0.0, 0.0, 1.0),
        _record(1.0, 0.0, 1.0),
        _record(2.0, 1.5, 0.0),
        _record(3.0, 0.0, 1.0),
        _record(4.0, 0.0, 1.0),
    ]
    cfg = {
        'forward_speed_mps': 1.0,
        'vision_yaw_gain': 1.0,
        'vision_yaw_max_correction': 1.0,
        'vision_lateral_gain': 0.0,
        'offline_tape_smoothing': True,
        'offline_line_smoothing_sec': 2.0,
        'offline_line_min_confidence': 0.1,
        'offline_adaptive_confidence': True,
    }

    estimate = _build_offline_tape_line_estimate(records, cfg)

    assert len(estimate.raw_traj) == len(records)
    assert len(estimate.smoothed_traj) == len(records)
    assert len(estimate.final_traj) == len(records)
    assert estimate.diagnostics.confidence_threshold >= 0.1
    assert estimate.diagnostics.valid_mask.tolist() == [True, True, False, True, True]


def test_loop_canonicalization_rejects_poorly_aligned_laps() -> None:
    traj = []
    for t in np.linspace(0.0, 2.0, 160, endpoint=False):
        phase = 2.0 * np.pi * (t % 1.0)
        radius = 1.0 if t < 1.0 else 2.0
        traj.append(
            TrajectoryPoint(
                t=float(t),
                x=float(radius * np.cos(phase)),
                y=float(radius * np.sin(phase)),
                yaw=float(phase),
            )
        )

    final_traj, loop_debug = _canonicalize_periodic_trajectory(
        traj,
        {
            'loop_period_sec': 1.0,
            'loop_samples': 64,
            'loop_similarity_align': False,
            'loop_strategy': 'representative_lap',
            'loop_max_alignment_rmse_ratio': 0.05,
            'loop_max_projection_rmse_ratio': 0.05,
        },
    )

    assert loop_debug is None
    assert len(final_traj) == len(traj)


def test_loop_canonicalization_can_select_forward_laps_only() -> None:
    traj = []
    samples_per_lap = 80
    for lap_idx in range(3):
        for sample_idx in range(samples_per_lap):
            phase = 2.0 * np.pi * sample_idx / samples_per_lap
            if lap_idx == 2:
                phase = -phase
            t = float(lap_idx + sample_idx / samples_per_lap)
            traj.append(
                TrajectoryPoint(
                    t=t,
                    x=float(np.cos(phase)),
                    y=float(np.sin(phase)),
                    yaw=float(phase),
                )
            )

    final_traj, loop_debug = _canonicalize_periodic_trajectory(
        traj,
        {
            'loop_period_sec': 1.0,
            'loop_samples': 64,
            'loop_similarity_align': False,
            'loop_strategy': 'representative_lap',
            'manual_lap_bounds_sec': [0.0, 1.0, 2.0, 3.0],
            'manual_lap_directions': ['forward', 'forward', 'reverse'],
            'loop_average_direction': 'forward',
            'loop_max_alignment_rmse_ratio': 10.0,
            'loop_max_projection_rmse_ratio': 10.0,
        },
    )

    assert loop_debug is not None
    assert [lap.lap_index for lap in loop_debug.laps] == [0, 1]
    assert [lap.direction for lap in loop_debug.laps] == ['forward', 'forward']
    assert len(final_traj) > 0


def test_loop_canonicalization_can_include_reversed_laps() -> None:
    traj = []
    samples_per_lap = 80
    for lap_idx in range(3):
        for sample_idx in range(samples_per_lap):
            phase = 2.0 * np.pi * sample_idx / samples_per_lap
            if lap_idx == 2:
                phase = -phase
            t = float(lap_idx + sample_idx / samples_per_lap)
            traj.append(
                TrajectoryPoint(
                    t=t,
                    x=float(np.cos(phase)),
                    y=float(np.sin(phase)),
                    yaw=float(phase),
                )
            )

    final_traj, loop_debug = _canonicalize_periodic_trajectory(
        traj,
        {
            'loop_period_sec': 1.0,
            'loop_samples': 64,
            'loop_similarity_align': False,
            'loop_strategy': 'representative_lap',
            'manual_lap_bounds_sec': [0.0, 1.0, 2.0, 3.0],
            'manual_lap_directions': ['forward', 'forward', 'reverse'],
            'loop_average_direction': 'any',
            'loop_normalize_reverse_laps': True,
            'loop_max_alignment_rmse_ratio': 10.0,
            'loop_max_projection_rmse_ratio': 10.0,
        },
    )

    assert loop_debug is not None
    assert [lap.lap_index for lap in loop_debug.laps] == [0, 1, 2]
    assert [lap.direction for lap in loop_debug.laps] == ['forward', 'forward', 'reverse']
    assert loop_debug.laps[2].raw_xy[0, 1] > 0.0
    assert len(final_traj) > 0


def test_distribute_loop_closure_error_removes_single_closing_jump() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 1.0],
        ],
        dtype=float,
    )

    closed = _distribute_loop_closure_error(points)

    assert closed[0].tolist() == pytest.approx([0.0, 0.0])
    assert closed[-1].tolist() == pytest.approx([0.0, 0.0])


def test_output_transform_can_flip_trajectory_x_axis() -> None:
    traj = [
        TrajectoryPoint(t=0.0, x=0.0, y=0.0, yaw=0.0),
        TrajectoryPoint(t=1.0, x=1.5, y=2.0, yaw=0.0),
    ]

    transformed = _apply_output_transform_to_trajectory(
        traj,
        {'trajectory_output_flip_x': True},
    )

    assert transformed[0].x == pytest.approx(0.0)
    assert transformed[0].y == pytest.approx(0.0)
    assert transformed[1].x == pytest.approx(-1.5)
    assert transformed[1].y == pytest.approx(2.0)
    assert abs(transformed[1].yaw) == pytest.approx(np.pi)
