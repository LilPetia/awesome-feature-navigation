from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .trajectory import (
    TrajectoryPoint,
    _align_lap_to_template,
    _lap_rmse,
    _resample_closed_polyline_by_arclength,
)


@dataclass(frozen=True)
class TrajectoryFusionDiagnostics:
    sample_count: int
    phase_shift: int
    alignment_rmse: float
    reverse_used: bool
    scale: float


@dataclass(frozen=True)
class TrajectoryFusionResult:
    fused_traj: list[TrajectoryPoint]
    left_aligned_traj: list[TrajectoryPoint]
    right_resampled_traj: list[TrajectoryPoint]
    diagnostics: TrajectoryFusionDiagnostics


def load_trajectory_csv(path: str | Path) -> list[TrajectoryPoint]:
    csv_path = Path(path)
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f'Trajectory CSV has no header: {csv_path}')
        required = {'t', 'x', 'y', 'yaw'}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f'Trajectory CSV is missing columns {missing}: {csv_path}')
        points: list[TrajectoryPoint] = []
        for row_number, row in enumerate(reader, start=2):
            points.append(
                TrajectoryPoint(
                    t=_finite_float(row, 't', row_number, csv_path),
                    x=_finite_float(row, 'x', row_number, csv_path),
                    y=_finite_float(row, 'y', row_number, csv_path),
                    yaw=_finite_float(row, 'yaw', row_number, csv_path),
                )
            )
    if len(points) < 2:
        raise ValueError(f'Trajectory CSV must contain at least two points: {csv_path}')
    return points


def fuse_trajectories(
    left_traj: Sequence[TrajectoryPoint],
    right_traj: Sequence[TrajectoryPoint],
    *,
    sample_count: int=512,
    phase_search_fraction: float=0.5,
    allow_reverse: bool=True,
) -> TrajectoryFusionResult:
    if len(left_traj) < 2 or len(right_traj) < 2:
        raise ValueError('Both trajectories must contain at least two points.')
    if sample_count < 8:
        raise ValueError('sample_count must be at least 8.')

    right_xy = _resample_closed_polyline_by_arclength(_trajectory_xy(right_traj), sample_count)
    left_xy = _resample_closed_polyline_by_arclength(_trajectory_xy(left_traj), sample_count)

    phase_shift, left_aligned_xy, alignment_rmse, scale = _align_lap_to_template(
        left_xy,
        right_xy,
        allow_scale=False,
        phase_search_fraction=phase_search_fraction,
    )
    reverse_used = False

    if allow_reverse:
        reverse_shift, reverse_aligned_xy, reverse_rmse, reverse_scale = _align_lap_to_template(
            left_xy[::-1].copy(),
            right_xy,
            allow_scale=False,
            phase_search_fraction=phase_search_fraction,
        )
        if reverse_rmse < alignment_rmse:
            phase_shift = reverse_shift
            left_aligned_xy = reverse_aligned_xy
            alignment_rmse = reverse_rmse
            scale = reverse_scale
            reverse_used = True

    fused_xy = 0.5 * (right_xy + left_aligned_xy)
    period = _trajectory_duration(right_traj, left_traj, sample_count=sample_count)

    fused_traj = _closed_xy_to_trajectory(fused_xy, period)
    left_aligned_traj = _closed_xy_to_trajectory(left_aligned_xy, period)
    right_resampled_traj = _closed_xy_to_trajectory(right_xy, period)

    return TrajectoryFusionResult(
        fused_traj=fused_traj,
        left_aligned_traj=left_aligned_traj,
        right_resampled_traj=right_resampled_traj,
        diagnostics=TrajectoryFusionDiagnostics(
            sample_count=sample_count,
            phase_shift=phase_shift,
            alignment_rmse=alignment_rmse,
            reverse_used=reverse_used,
            scale=scale,
        ),
    )


def _finite_float(row: dict[str, str], key: str, row_number: int, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f'Invalid {key!r} value at row {row_number}: {path}') from None
    if not np.isfinite(value):
        raise ValueError(f'Non-finite {key!r} value at row {row_number}: {path}')
    return value


def _trajectory_xy(traj: Sequence[TrajectoryPoint]) -> np.ndarray:
    return np.array([[point.x, point.y] for point in traj], dtype=float)


def _trajectory_duration(
    primary: Sequence[TrajectoryPoint],
    fallback: Sequence[TrajectoryPoint],
    *,
    sample_count: int,
) -> float:
    for traj in (primary, fallback):
        if len(traj) < 2:
            continue
        duration = float(traj[-1].t - traj[0].t)
        if np.isfinite(duration) and duration > 1e-9:
            return duration
    return float(max(1, sample_count - 1))


def _closed_xy_to_trajectory(points_xy: np.ndarray, period: float) -> list[TrajectoryPoint]:
    body = np.asarray(points_xy, dtype=float)
    if body.shape[0] == 0:
        return []
    if not np.allclose(body[0], body[-1]):
        loop_xy = np.vstack((body, body[0]))
    else:
        loop_xy = body.copy()

    deltas = np.roll(loop_xy, -1, axis=0) - loop_xy
    if deltas.shape[0] > 1 and _lap_rmse(loop_xy[-1:], loop_xy[:1]) < 1e-9:
        deltas[-1] = deltas[0]
    yaw = np.arctan2(deltas[:, 1], deltas[:, 0])

    denom = max(1, loop_xy.shape[0] - 1)
    return [
        TrajectoryPoint(
            t=float(period * idx / denom),
            x=float(point[0]),
            y=float(point[1]),
            yaw=float(yaw[idx]),
        )
        for idx, point in enumerate(loop_xy)
    ]
