from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import awesome_feature_navigation.fusion_cli as fusion_cli
from awesome_feature_navigation.fusion import fuse_trajectories, load_trajectory_csv
from awesome_feature_navigation.plotting import save_trajectory_csv
from awesome_feature_navigation.trajectory import TrajectoryPoint


def _traj(points_xy: np.ndarray, period: float=10.0) -> list[TrajectoryPoint]:
    points = np.asarray(points_xy, dtype=float)
    deltas = np.roll(points, -1, axis=0) - points
    yaw = np.arctan2(deltas[:, 1], deltas[:, 0])
    denom = max(1, points.shape[0] - 1)
    return [
        TrajectoryPoint(
            t=float(period * idx / denom),
            x=float(point[0]),
            y=float(point[1]),
            yaw=float(yaw[idx]),
        )
        for idx, point in enumerate(points)
    ]


def _as_xy(traj: list[TrajectoryPoint]) -> np.ndarray:
    return np.array([[point.x, point.y] for point in traj], dtype=float)


def test_fuse_trajectories_aligns_phase_and_reverse_left_eye() -> None:
    right_loop = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.4, 0.9],
            [1.2, 1.6],
            [-0.2, 1.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    theta = 0.6
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )
    left_loop = np.roll(right_loop[:-1][::-1] @ rotation + np.array([3.0, -1.0]), 2, axis=0)

    result = fuse_trajectories(
        _traj(left_loop),
        _traj(right_loop),
        sample_count=96,
        phase_search_fraction=0.5,
        allow_reverse=True,
    )
    without_reverse = fuse_trajectories(
        _traj(left_loop),
        _traj(right_loop),
        sample_count=96,
        phase_search_fraction=0.5,
        allow_reverse=False,
    )

    assert result.diagnostics.reverse_used is True
    assert result.diagnostics.alignment_rmse < without_reverse.diagnostics.alignment_rmse
    assert result.diagnostics.alignment_rmse < 0.05
    assert len(result.fused_traj) == 97
    np.testing.assert_allclose(_as_xy(result.fused_traj), _as_xy(result.right_resampled_traj), atol=0.05)


def test_fuse_trajectories_validates_inputs() -> None:
    traj = _traj(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float))

    with pytest.raises(ValueError, match='at least two points'):
        fuse_trajectories(traj[:1], traj)
    with pytest.raises(ValueError, match='sample_count'):
        fuse_trajectories(traj, traj, sample_count=4)


def test_load_trajectory_csv_reads_saved_output_and_rejects_bad_files(tmp_path: Path) -> None:
    csv_path = tmp_path / 'trajectory.csv'
    save_trajectory_csv(_traj(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)), str(csv_path))

    loaded = load_trajectory_csv(csv_path)

    assert len(loaded) == 2
    assert loaded[1].x == pytest.approx(1.0)

    missing = tmp_path / 'missing_columns.csv'
    missing.write_text('t,x,y\n0,0,0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='missing columns'):
        load_trajectory_csv(missing)

    non_finite = tmp_path / 'non_finite.csv'
    non_finite.write_text('t,x,y,yaw\n0,0,nan,0\n1,1,0,0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Non-finite'):
        load_trajectory_csv(non_finite)


def test_fusion_cli_saves_fused_and_debug_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    left_csv = tmp_path / 'left.csv'
    right_csv = tmp_path / 'right.csv'
    out_prefix = tmp_path / 'fused'
    left = _traj(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float))
    right = _traj(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float))
    save_trajectory_csv(left, str(left_csv))
    save_trajectory_csv(right, str(right_csv))
    saved: list[str] = []

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'afn-fuse',
            '--left',
            str(left_csv),
            '--right',
            str(right_csv),
            '--out',
            str(out_prefix),
            '--samples',
            '32',
            '--save-aligned-debug',
        ],
    )
    monkeypatch.setattr(fusion_cli, 'save_trajectory_csv', lambda traj, path: saved.append(path))
    monkeypatch.setattr(fusion_cli, 'save_trajectory_plot', lambda traj, path, title, axis_unit: saved.append(path))

    fusion_cli.main()

    output = capsys.readouterr().out
    assert 'LEFT -> RIGHT alignment RMSE' in output
    assert str(out_prefix.with_suffix('.csv')) in saved
    assert str(Path(str(out_prefix) + '_left_aligned.html')) in saved
    assert str(Path(str(out_prefix) + '_right_resampled.csv')) in saved
