from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import awesome_feature_navigation.cli as cli
from awesome_feature_navigation.auto_config import AutoLineConfig
from awesome_feature_navigation.cli import (
    _cfg_bool,
    _looks_like_absolute_seconds,
    _merge_calibration_cfg,
    _normalize_time_base,
    _parse_lap_bounds,
    _resolve_path,
)
from awesome_feature_navigation.imu_preintegration import IMUSample
from awesome_feature_navigation.trajectory import (
    LoopAveragingDebug,
    LoopLapDebug,
    TapeLineDiagnostics,
    TrajectoryEstimateResult,
    TrajectoryPoint,
)


def test_parse_lap_bounds_supports_seconds_and_clock_notation() -> None:
    assert _parse_lap_bounds('11, 1:02, 2:03:04') == pytest.approx([11.0, 62.0, 7384.0])
    assert _parse_lap_bounds(' , 5, , 0:10 ') == pytest.approx([5.0, 10.0])


def test_cli_bool_absolute_seconds_and_relative_path_resolution(tmp_path: Path) -> None:
    config_path = tmp_path / 'configs' / 'right.yaml'
    config_path.parent.mkdir()
    sibling = config_path.parent / 'data.csv'
    sibling.write_text('x', encoding='utf-8')

    assert _cfg_bool(None, True) is True
    assert _cfg_bool('false', True) is False
    assert _cfg_bool('yes', False) is True
    assert _cfg_bool(0, True) is False
    assert _looks_like_absolute_seconds(1_700_000_000.0) is True
    assert _looks_like_absolute_seconds(123.0) is False
    assert _resolve_path('data.csv', config_path) == sibling
    assert _resolve_path('missing.csv', config_path) == Path('missing.csv')
    assert _resolve_path(str(sibling), None) == sibling


def test_normalize_time_base_handles_absolute_and_relative_timestamps() -> None:
    imu = [
        IMUSample(t=1_700_000_000.5, accel=np.zeros(3), omega=np.zeros(3)),
        IMUSample(t=1_700_000_001.0, accel=np.zeros(3), omega=np.zeros(3)),
    ]

    frames, synced_imu, synced = _normalize_time_base(
        [1_700_000_000.0, 1_700_000_000.5],
        imu,
        {'sync_time_base': 'auto'},
    )

    assert frames == pytest.approx([0.0, 0.5])
    assert synced_imu is not None
    assert [sample.t for sample in synced_imu] == pytest.approx([0.5, 1.0])
    assert synced is True

    relative, _, synced = _normalize_time_base([10.0, 11.0], None, {'frame_timestamp_normalize': True})

    assert relative == pytest.approx([0.0, 1.0])
    assert synced is False
    assert _normalize_time_base(None, imu, {}) == (None, imu, False)
    assert _normalize_time_base([], imu, {}) == ([], imu, False)


def test_merge_calibration_cfg_ignores_missing_path_value() -> None:
    cfg: dict[str, object] = {}

    assert _merge_calibration_cfg(cfg, None, None, lambda path: {'x': 1}) is None
    assert cfg == {}


def test_main_wires_config_calibration_outputs_and_debug_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        '\n'.join(
            [
                'frame_timestamps: frames.csv',
                'imu_calibration: imu.yaml',
                'camchain_calibration: camchain.yaml',
                'imu_apply_camchain_timeshift: true',
                'imu_timeshift_cam_imu_sec: 0.25',
                'loop_period_sec: 1.0',
            ]
        ),
        encoding='utf-8',
    )
    for name in ('frames.csv', 'imu.yaml', 'camchain.yaml', 'imu.csv'):
        (tmp_path / name).write_text('placeholder', encoding='utf-8')
    out_prefix = tmp_path / 'trajectory'
    saved: list[str] = []

    diagnostics = TapeLineDiagnostics(
        t=np.array([0.0], dtype=float),
        confidence=np.array([1.0], dtype=float),
        confidence_threshold=0.5,
        valid_mask=np.array([True]),
        angle_raw=np.array([0.0], dtype=float),
        angle_smooth=np.array([0.0], dtype=float),
        bottom_x_raw=np.array([50.0], dtype=float),
        bottom_x_smooth=np.array([50.0], dtype=float),
        speed_mps=np.array([0.2], dtype=float),
        delta_yaw_imu=np.array([0.0], dtype=float),
    )
    loop_debug = LoopAveragingDebug(
        period_sec=1.0,
        samples_per_lap=2,
        laps=[
            LoopLapDebug(
                lap_index=0,
                t0=0.0,
                t1=1.0,
                direction='forward',
                raw_xy=np.zeros((2, 2), dtype=float),
                aligned_xy=np.zeros((2, 2), dtype=float),
                projected_xy=np.zeros((2, 2), dtype=float),
                phase_shift=0,
                scale=1.0,
                alignment_rmse=0.0,
                projection_rmse=0.0,
                kept=True,
            )
        ],
        canonical_xy=np.zeros((2, 2), dtype=float),
        spline_control_xy=np.zeros((2, 2), dtype=float),
        split_times=np.array([0.0, 1.0], dtype=float),
    )
    result = TrajectoryEstimateResult(
        raw_traj=[TrajectoryPoint(0.0, 0.0, 0.0, 0.0)],
        smoothed_traj=[TrajectoryPoint(0.0, 0.0, 0.0, 0.0)],
        final_traj=[TrajectoryPoint(0.0, 0.0, 0.0, 0.0)],
        loop_debug=loop_debug,
        mode='tape_line',
        relative_scale=True,
        estimated_loop_period_sec=1.0,
        line_valid_ratio=1.0,
        tape_diagnostics=diagnostics,
    )

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'afn-run',
            '--video',
            'video.mp4',
            '--imu',
            str(tmp_path / 'imu.csv'),
            '--config',
            str(config_path),
            '--mode',
            'tape_line',
            '--color',
            'blue',
            '--auto-color',
            '--imu-gyro-scale',
            '2.0',
            '--imu-time-scale',
            '0.5',
            '--imu-use-translation',
            '--frame-timestamps',
            'frames.csv',
            '--frame-time-scale',
            '1.0',
            '--imu-calibration',
            str(tmp_path / 'imu.yaml'),
            '--camchain',
            str(tmp_path / 'camchain.yaml'),
            '--no-auto-config',
            '--lap-bounds',
            '0,1',
            '--save-loop-debug',
            '--save-debug',
            '--out',
            str(out_prefix),
        ],
    )
    monkeypatch.setattr(cli, 'load_imu_calibration_config', lambda path: {'imu_accel_noise_density': 0.1})
    monkeypatch.setattr(
        cli,
        'load_camchain_calibration_config',
        lambda path: {'imu_camera_rotation': np.eye(3).tolist(), 'imu_timeshift_cam_imu_sec': 0.25},
    )
    monkeypatch.setattr(cli, 'load_frame_timestamps_csv', lambda path, time_scale: [1_700_000_000.0])
    monkeypatch.setattr(
        cli,
        'load_imu_csv',
        lambda path, time_scale, gyro_scale: [IMUSample(1_700_000_000.0, np.zeros(3), np.zeros(3))],
    )
    monkeypatch.setattr(cli, 'calibrate_imu_samples', lambda samples, cfg: samples)
    monkeypatch.setattr(
        cli,
        'apply_auto_video_config',
        lambda video_path, cfg: (
            cfg,
            AutoLineConfig('blue', [[95, 100, 60, 130, 255, 255]], valid_ratio=0.9, mean_score=2.0, sample_count=1),
        ),
    )
    monkeypatch.setattr(cli, 'estimate_trajectory_with_details', lambda **kwargs: result)
    monkeypatch.setattr(cli, 'save_trajectory_csv', lambda traj, path: saved.append(path))
    monkeypatch.setattr(cli, 'save_trajectory_plot', lambda traj, path, title, axis_unit: saved.append(path))
    monkeypatch.setattr(cli, 'save_tape_diagnostics_csv', lambda diagnostics, path: saved.append(path))
    monkeypatch.setattr(cli, 'save_tape_diagnostics_plot', lambda diagnostics, path: saved.append(path))
    monkeypatch.setattr(cli, 'save_loop_debug_csv', lambda loop_debug, path: saved.append(path))
    monkeypatch.setattr(cli, 'save_loop_debug_plot', lambda loop_debug, path: saved.append(path))

    cli.main()

    output = capsys.readouterr().out
    assert 'Mode: tape_line' in output
    assert 'Auto line: color=blue' in output
    assert 'Time sync' in output
    assert str(out_prefix.with_suffix('.csv')) in saved
    assert str(Path(str(out_prefix) + '_diagnostics.html')) in saved
    assert str(Path(str(out_prefix) + '_laps.html')) in saved
