from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest

from awesome_feature_navigation.plotting import (
    _write_plot,
    save_loop_debug_csv,
    save_loop_debug_plot,
    save_tape_diagnostics_csv,
    save_tape_diagnostics_plot,
    save_trajectory_csv,
    save_trajectory_plot,
)
from awesome_feature_navigation.trajectory import (
    LoopAveragingDebug,
    LoopLapDebug,
    TapeLineDiagnostics,
    TrajectoryPoint,
)


def _trajectory() -> list[TrajectoryPoint]:
    return [
        TrajectoryPoint(t=0.0, x=0.0, y=0.0, yaw=0.0),
        TrajectoryPoint(t=1.0, x=1.0, y=0.5, yaw=0.25),
    ]


def test_save_trajectory_csv_and_html_export_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / 'trajectory.csv'
    html_path = tmp_path / 'trajectory.html'
    captured: dict[str, Any] = {}

    def fake_write_html(self: go.Figure, path: str, config: dict[str, object]) -> None:
        captured['path'] = path
        captured['config'] = config
        captured['trace_count'] = len(self.data)

    monkeypatch.setattr(go.Figure, 'write_html', fake_write_html)

    save_trajectory_csv(_trajectory(), str(csv_path))
    save_trajectory_plot(_trajectory(), str(html_path))

    assert csv_path.read_text(encoding='utf-8').splitlines()[0] == 't,x,y,yaw'
    assert captured['path'] == str(html_path)
    assert captured['trace_count'] == 3
    options = captured['config']['toImageButtonOptions']
    assert options['filename'] == 'trajectory'
    assert options['scale'] == 3
    assert options['width'] == 2400


def test_write_plot_static_image_handles_missing_kaleido(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_write_image(self: go.Figure, path: str, width: int, height: int, scale: int) -> None:
        raise ValueError('kaleido is missing')

    monkeypatch.setattr(go.Figure, 'write_image', fake_write_image)

    _write_plot(go.Figure(), 'plot.png')

    out = capsys.readouterr().out
    assert 'kaleido is missing' in out
    assert 'pip install kaleido' in out


def test_tape_diagnostics_csv_and_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostics = TapeLineDiagnostics(
        t=np.array([0.0, 1.0], dtype=float),
        confidence=np.array([0.25, 0.9], dtype=float),
        confidence_threshold=0.5,
        valid_mask=np.array([False, True]),
        angle_raw=np.array([0.1, 0.2], dtype=float),
        angle_smooth=np.array([0.15, 0.18], dtype=float),
        bottom_x_raw=np.array([10.0, 12.0], dtype=float),
        bottom_x_smooth=np.array([11.0, 11.5], dtype=float),
        speed_mps=np.array([0.2, 0.3], dtype=float),
        delta_yaw_imu=np.array([0.0, 0.01], dtype=float),
    )
    captured: dict[str, Any] = {}

    def fake_write_html(self: go.Figure, path: str, config: dict[str, object]) -> None:
        captured['path'] = path
        captured['traces'] = len(self.data)
        captured['config'] = config

    monkeypatch.setattr(go.Figure, 'write_html', fake_write_html)

    csv_path = tmp_path / 'diagnostics.csv'
    html_path = tmp_path / 'diagnostics.html'
    save_tape_diagnostics_csv(diagnostics, str(csv_path))
    save_tape_diagnostics_plot(diagnostics, str(html_path))

    assert 'confidence_threshold' in csv_path.read_text(encoding='utf-8').splitlines()[0]
    assert captured['path'] == str(html_path)
    assert captured['traces'] == 5
    assert captured['config']['toImageButtonOptions']['filename'] == 'diagnostics'


def test_loop_debug_csv_and_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
    lap = LoopLapDebug(
        lap_index=0,
        t0=0.0,
        t1=1.0,
        direction='forward',
        raw_xy=xy,
        aligned_xy=xy + 1.0,
        projected_xy=xy + 2.0,
        phase_shift=1,
        scale=1.0,
        alignment_rmse=0.1,
        projection_rmse=0.2,
        kept=True,
    )
    rejected_lap = LoopLapDebug(
        lap_index=1,
        t0=1.0,
        t1=2.0,
        direction='reverse',
        raw_xy=xy + 3.0,
        aligned_xy=xy + 4.0,
        projected_xy=xy + 5.0,
        phase_shift=2,
        scale=0.9,
        alignment_rmse=2.0,
        projection_rmse=3.0,
        kept=False,
    )
    debug = LoopAveragingDebug(
        period_sec=1.0,
        samples_per_lap=xy.shape[0],
        laps=[lap, rejected_lap],
        canonical_xy=np.vstack((xy, xy[0])),
        spline_control_xy=xy,
        split_times=np.array([0.0, 1.0], dtype=float),
    )
    captured: dict[str, Any] = {}

    def fake_write_html(self: go.Figure, path: str, config: dict[str, object]) -> None:
        captured['path'] = path
        captured['traces'] = len(self.data)
        captured['config'] = config

    monkeypatch.setattr(go.Figure, 'write_html', fake_write_html)

    csv_path = tmp_path / 'laps.csv'
    html_path = tmp_path / 'laps.html'
    save_loop_debug_csv(debug, str(csv_path))
    save_loop_debug_plot(debug, str(html_path))

    assert 'direction' in csv_path.read_text(encoding='utf-8').splitlines()[0]
    assert captured['path'] == str(html_path)
    assert captured['traces'] == 9
    assert captured['config']['toImageButtonOptions']['width'] == 3600


def test_empty_plot_inputs_warn_and_skip(capsys: pytest.CaptureFixture[str]) -> None:
    empty_diag = TapeLineDiagnostics(
        t=np.zeros(0, dtype=float),
        confidence=np.zeros(0, dtype=float),
        confidence_threshold=0.0,
        valid_mask=np.zeros(0, dtype=bool),
        angle_raw=np.zeros(0, dtype=float),
        angle_smooth=np.zeros(0, dtype=float),
        bottom_x_raw=np.zeros(0, dtype=float),
        bottom_x_smooth=np.zeros(0, dtype=float),
        speed_mps=np.zeros(0, dtype=float),
        delta_yaw_imu=np.zeros(0, dtype=float),
    )
    empty_debug = LoopAveragingDebug(
        period_sec=0.0,
        samples_per_lap=0,
        laps=[],
        canonical_xy=np.zeros((0, 2), dtype=float),
        spline_control_xy=np.zeros((0, 2), dtype=float),
        split_times=np.zeros(0, dtype=float),
    )

    save_trajectory_plot([], 'empty.html')
    save_tape_diagnostics_plot(empty_diag, 'empty_diag.html')
    save_loop_debug_plot(empty_debug, 'empty_laps.html')

    out = capsys.readouterr().out
    assert 'Trajectory is empty' in out
    assert 'Tape diagnostics are empty' in out
    assert 'Loop debug is empty' in out
