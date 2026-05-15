from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, cast

import cv2
import numpy as np

from .imu_io import slice_imu
from .imu_preintegration import IMUPreintegrationWrapper, IMUSample, build_default_params
from .line_detection import LineDetector, TapeObservation


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    yaw: float


@dataclass
class LoopLapDebug:
    lap_index: int
    t0: float
    t1: float
    raw_xy: np.ndarray
    aligned_xy: np.ndarray
    projected_xy: np.ndarray
    phase_shift: int
    scale: float
    alignment_rmse: float
    projection_rmse: float
    kept: bool


@dataclass
class LoopAveragingDebug:
    period_sec: float
    samples_per_lap: int
    laps: List[LoopLapDebug]
    canonical_xy: np.ndarray
    spline_control_xy: np.ndarray
    split_times: np.ndarray


@dataclass
class TrajectoryEstimateResult:
    raw_traj: List[TrajectoryPoint]
    final_traj: List[TrajectoryPoint]
    loop_debug: Optional[LoopAveragingDebug]
    mode: str = 'tape_line'
    relative_scale: bool = False
    estimated_loop_period_sec: Optional[float] = None
    line_valid_ratio: Optional[float] = None


@dataclass(frozen=True)
class TapeFrameObservation:
    t: float
    dt: float
    obs: TapeObservation
    confidence: float
    delta_yaw: float
    delta_p: np.ndarray


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _mp4v_fourcc() -> int:
    return int(cast(Any, cv2).VideoWriter_fourcc(*'mp4v'))


def _cfg_float(cfg: Dict, keys: Sequence[str], default: float) -> float:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            try:
                return float(cfg[key])
            except (TypeError, ValueError):
                continue
    return float(default)


def _build_imu_preintegration_params(cfg: Dict) -> object | None:
    return build_default_params(
        gravity_mps2=_cfg_float(cfg, ['imu_gravity_mps2', 'gravity_mps2'], 9.81),
        accel_noise_sigma=_cfg_float(
            cfg,
            ['imu_accel_noise_density', 'imu_accel_noise_sigma', 'accel_noise_sigma'],
            0.2,
        ),
        gyro_noise_sigma=_cfg_float(
            cfg,
            ['imu_gyro_noise_density', 'imu_gyro_noise_sigma', 'gyro_noise_sigma'],
            0.02,
        ),
        accel_bias_rw_sigma=_cfg_float(
            cfg,
            ['imu_accel_random_walk', 'imu_accel_bias_rw_sigma', 'accel_bias_rw_sigma'],
            0.0005,
        ),
        gyro_bias_rw_sigma=_cfg_float(
            cfg,
            ['imu_gyro_random_walk', 'imu_gyro_bias_rw_sigma', 'gyro_bias_rw_sigma'],
            0.0002,
        ),
    )


def _frame_time_at(
    cap: cv2.VideoCapture,
    frame_index: int,
    frame_timestamps: Optional[Sequence[float]],
) -> float:
    if frame_timestamps is not None:
        idx = frame_index - 1
        if 0 <= idx < len(frame_timestamps):
            return float(frame_timestamps[idx])
    return float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0


def _blend_angles(primary: float, secondary: float, secondary_weight: float) -> float:
    if not np.isfinite(primary):
        return float(secondary)
    if not np.isfinite(secondary):
        return float(primary)
    weight = float(np.clip(secondary_weight, 0.0, 1.0))
    primary_vec = np.array([np.cos(primary), np.sin(primary)], dtype=float)
    secondary_vec = np.array([np.cos(secondary), np.sin(secondary)], dtype=float)
    mixed = (1.0 - weight) * primary_vec + weight * secondary_vec
    norm = float(np.linalg.norm(mixed))
    if norm < 1e-9:
        return float(primary)
    mixed /= norm
    return float(np.arctan2(mixed[1], mixed[0]))


def _rigid_align_points(points_xy: np.ndarray, ref_xy: np.ndarray) -> np.ndarray:
    src = np.asarray(points_xy, dtype=float)
    ref = np.asarray(ref_xy, dtype=float)
    src_center = src.mean(axis=0, keepdims=True)
    ref_center = ref.mean(axis=0, keepdims=True)
    src_zero = src - src_center
    ref_zero = ref - ref_center
    H = src_zero.T @ ref_zero
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if float(np.linalg.det(R)) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    t = ref_center - src_center @ R
    return src @ R + t


def _similarity_align_points(
    points_xy: np.ndarray,
    ref_xy: np.ndarray,
    scale_limits: tuple[float, float]=(0.5, 2.0),
) -> tuple[np.ndarray, float]:
    src = np.asarray(points_xy, dtype=float)
    ref = np.asarray(ref_xy, dtype=float)
    src_center = src.mean(axis=0, keepdims=True)
    ref_center = ref.mean(axis=0, keepdims=True)
    src_zero = src - src_center
    ref_zero = ref - ref_center
    H = src_zero.T @ ref_zero
    U, singular_values, Vt = np.linalg.svd(H)
    R = U @ Vt
    if float(np.linalg.det(R)) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    denom = float(np.sum(src_zero * src_zero))
    scale = 1.0 if denom < 1e-12 else float(np.sum(singular_values) / denom)
    scale = float(np.clip(scale, scale_limits[0], scale_limits[1]))
    aligned = scale * (src_zero @ R) + ref_center
    return (aligned, scale)


def _trajectory_to_arrays(traj: Sequence[TrajectoryPoint]) -> tuple[np.ndarray, np.ndarray]:
    ts = np.array([p.t for p in traj], dtype=float)
    xy = np.array([[p.x, p.y] for p in traj], dtype=float)
    return (ts, xy)


def _center_points(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    return pts - pts.mean(axis=0, keepdims=True)


def _lap_rmse(points_xy: np.ndarray, ref_xy: np.ndarray) -> float:
    diff = np.asarray(points_xy, dtype=float) - np.asarray(ref_xy, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _anchor_scores_from_normalized(norm_xy: np.ndarray, anchor: str) -> np.ndarray:
    pts = np.asarray(norm_xy, dtype=float)
    anchor_key = anchor.strip().lower()
    if anchor_key == 'bottom_left':
        return pts[:, 0] + pts[:, 1]
    if anchor_key == 'top_left':
        return pts[:, 0] + (1.0 - pts[:, 1])
    if anchor_key == 'top_right':
        return (1.0 - pts[:, 0]) + (1.0 - pts[:, 1])
    if anchor_key == 'bottom_right':
        return (1.0 - pts[:, 0]) + pts[:, 1]
    if anchor_key == 'left':
        return pts[:, 0]
    if anchor_key == 'right':
        return 1.0 - pts[:, 0]
    if anchor_key == 'top':
        return 1.0 - pts[:, 1]
    if anchor_key == 'bottom':
        return pts[:, 1]
    return pts[:, 0] + pts[:, 1]


def _anchor_scores(points_xy: np.ndarray, anchor: str) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=float)
    min_xy = pts.min(axis=0, keepdims=True)
    span_xy = np.maximum(pts.max(axis=0, keepdims=True) - min_xy, 1e-9)
    norm = (pts - min_xy) / span_xy
    return _anchor_scores_from_normalized(norm, anchor)


def _resample_polyline_by_arclength(points_xy: np.ndarray, sample_count: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros((sample_count, 2), dtype=float)
    if pts.shape[0] == 1:
        return np.repeat(pts, sample_count, axis=0)
    deltas = np.diff(pts, axis=0)
    seg_len = np.sqrt(np.sum(deltas * deltas, axis=1))
    keep = np.concatenate(([True], seg_len > 1e-9))
    pts = pts[keep]
    if pts.shape[0] == 1:
        return np.repeat(pts, sample_count, axis=0)
    deltas = np.diff(pts, axis=0)
    seg_len = np.sqrt(np.sum(deltas * deltas, axis=1))
    cum_len = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = float(cum_len[-1])
    if total_len < 1e-9:
        return np.repeat(pts[:1], sample_count, axis=0)
    query = np.linspace(0.0, total_len, sample_count, endpoint=False)
    x = np.interp(query, cum_len, pts[:, 0])
    y = np.interp(query, cum_len, pts[:, 1])
    return np.column_stack((x, y))


def _resample_closed_polyline_by_arclength(points_xy: np.ndarray, sample_count: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros((sample_count, 2), dtype=float)
    if pts.shape[0] == 1:
        return np.repeat(pts, sample_count, axis=0)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack((pts, pts[0]))
    deltas = np.diff(pts, axis=0)
    seg_len = np.sqrt(np.sum(deltas * deltas, axis=1))
    keep = np.concatenate(([True], seg_len > 1e-9))
    pts = pts[keep]
    if pts.shape[0] <= 2:
        return np.repeat(pts[:1], sample_count, axis=0)
    deltas = np.diff(pts, axis=0)
    seg_len = np.sqrt(np.sum(deltas * deltas, axis=1))
    cum_len = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = float(cum_len[-1])
    if total_len < 1e-9:
        return np.repeat(pts[:1], sample_count, axis=0)
    query = np.linspace(0.0, total_len, sample_count, endpoint=False)
    x = np.interp(query, cum_len, pts[:, 0])
    y = np.interp(query, cum_len, pts[:, 1])
    return np.column_stack((x, y))


def _align_lap_to_template(
    points_xy: np.ndarray,
    template_xy: np.ndarray,
    allow_scale: bool,
    phase_search_fraction: float,
) -> tuple[int, np.ndarray, float, float]:
    sample_count = int(points_xy.shape[0])
    best_shift = 0
    best_rmse = float('inf')
    best_scale = 1.0
    if allow_scale:
        best_aligned, best_scale = _similarity_align_points(points_xy, template_xy)
    else:
        best_aligned = _rigid_align_points(points_xy, template_xy)
    max_shift = sample_count // 2
    if phase_search_fraction > 0.0:
        max_shift = max(1, min(sample_count // 2, int(round(sample_count * phase_search_fraction))))
    for shift_offset in range(-max_shift, max_shift + 1):
        shift = shift_offset % sample_count
        shifted = np.roll(points_xy, -shift, axis=0)
        if allow_scale:
            aligned, scale = _similarity_align_points(shifted, template_xy)
        else:
            aligned = _rigid_align_points(shifted, template_xy)
            scale = 1.0
        rmse = _lap_rmse(aligned, template_xy)
        if rmse < best_rmse:
            best_shift = shift
            best_rmse = rmse
            best_aligned = aligned
            best_scale = scale
    return (best_shift, best_aligned, best_rmse, best_scale)


def _robust_keep_mask(rmses: Sequence[float], min_keep: int=2, sigma: float=2.5) -> np.ndarray:
    values = np.asarray(rmses, dtype=float)
    if values.size <= min_keep or sigma <= 0.0:
        return np.ones(values.shape[0], dtype=bool)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad < 1e-9:
        return np.ones(values.shape[0], dtype=bool)
    robust_z = 0.67448975 * np.abs(values - median) / mad
    keep = robust_z <= sigma
    if int(np.sum(keep)) >= min_keep:
        return keep
    keep = np.zeros(values.shape[0], dtype=bool)
    order = np.argsort(values)
    keep[order[:min_keep]] = True
    return keep


def _choose_loop_anchor(loop_xy: np.ndarray, anchor: str) -> int:
    scores = _anchor_scores(loop_xy, anchor)
    if scores.size == 0:
        return 0
    return int(np.argmin(scores))


def _extract_lap_segment(
    ts: np.ndarray,
    xy: np.ndarray,
    t0: float,
    t1: float,
    samples_per_lap: int,
) -> np.ndarray:
    mask = (ts >= t0) & (ts <= t1)
    lap_xy = xy[mask]
    x0 = float(np.interp(t0, ts, xy[:, 0]))
    y0 = float(np.interp(t0, ts, xy[:, 1]))
    x1 = float(np.interp(t1, ts, xy[:, 0]))
    y1 = float(np.interp(t1, ts, xy[:, 1]))
    lap_xy = np.vstack((np.array([[x0, y0]], dtype=float), lap_xy, np.array([[x1, y1]], dtype=float)))
    return _resample_polyline_by_arclength(lap_xy, samples_per_lap)


def _search_anchor_boundaries(
    ts: np.ndarray,
    xy: np.ndarray,
    period: float,
    anchor: str,
    search_radius_fraction: float,
    min_gap_fraction: float,
) -> List[tuple[int, float]]:
    if period <= 0.0:
        return []
    search_radius = max(period * 0.1, period * search_radius_fraction)
    context_radius = period * 0.6
    min_gap = period * min_gap_fraction
    offset_steps = 64

    def collect_for_targets(targets: np.ndarray) -> tuple[List[tuple[int, float]], List[float]]:
        candidates: List[tuple[int, float, float]] = []
        for target_t in targets:
            search_mask = (ts >= target_t - search_radius) & (ts <= target_t + search_radius)
            context_mask = (ts >= target_t - context_radius) & (ts <= target_t + context_radius)
            search_idx = np.flatnonzero(search_mask)
            context_xy = xy[context_mask]
            if search_idx.size == 0 or context_xy.shape[0] < 4:
                continue
            min_xy = context_xy.min(axis=0, keepdims=True)
            span_xy = np.maximum(context_xy.max(axis=0, keepdims=True) - min_xy, 1e-9)
            norm = (xy[search_idx] - min_xy) / span_xy
            scores = _anchor_scores_from_normalized(norm, anchor)
            best_local = int(np.argmin(scores))
            best_idx = int(search_idx[best_local])
            candidates.append((best_idx, float(ts[best_idx]), float(scores[best_local])))
        candidates.sort(key=lambda item: item[1])
        boundaries: List[tuple[int, float]] = []
        scores_out: List[float] = []
        for idx, t_value, score in candidates:
            if not boundaries:
                boundaries.append((idx, t_value))
                scores_out.append(score)
                continue
            if t_value - boundaries[-1][1] < min_gap:
                if score < scores_out[-1]:
                    boundaries[-1] = (idx, t_value)
                    scores_out[-1] = score
                continue
            boundaries.append((idx, t_value))
            scores_out.append(score)
        return (boundaries, scores_out)

    best_boundaries: List[tuple[int, float]] = []
    best_cost = float('inf')
    best_count = -1
    for offset in np.linspace(0.0, period, offset_steps, endpoint=False):
        targets = np.arange(float(ts[0]) + offset, float(ts[-1]) + period, period, dtype=float)
        boundaries, scores = collect_for_targets(targets)
        if len(boundaries) < 2:
            continue
        durations = np.diff([t_value for _, t_value in boundaries])
        duration_penalty = 0.0
        if durations.size > 0:
            duration_penalty = float(np.mean(np.abs(durations - period)) / max(period, 1e-9))
        score_penalty = float(np.mean(scores)) if scores else 0.0
        cost = score_penalty + 0.75 * duration_penalty
        if len(boundaries) > best_count or (len(boundaries) == best_count and cost < best_cost):
            best_boundaries = boundaries
            best_cost = cost
            best_count = len(boundaries)
    return best_boundaries


def _extract_anchor_laps(
    traj: Sequence[TrajectoryPoint],
    period: float,
    samples_per_lap: int,
    min_fraction: float,
    anchor: str,
    search_radius_fraction: float,
) -> tuple[List[tuple[int, float, float, np.ndarray]], np.ndarray]:
    ts, xy = _trajectory_to_arrays(traj)
    total_time = float(ts[-1] - ts[0])
    if total_time < period * min_fraction:
        return ([], np.zeros(0, dtype=float))
    boundaries = _search_anchor_boundaries(
        ts=ts,
        xy=xy,
        period=period,
        anchor=anchor,
        search_radius_fraction=search_radius_fraction,
        min_gap_fraction=min_fraction * 0.8,
    )
    if len(boundaries) < 2:
        return ([], np.zeros(0, dtype=float))
    max_fraction = max(1.2, float(1.0 / max(min_fraction, 1e-6)))
    laps: List[tuple[int, float, float, np.ndarray]] = []
    split_times: List[float] = []
    for lap_idx, ((_, lap_t0), (_, lap_t1)) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        duration = float(lap_t1 - lap_t0)
        if duration < period * min_fraction or duration > period * max_fraction:
            continue
        lap_xy = _extract_lap_segment(ts=ts, xy=xy, t0=lap_t0, t1=lap_t1, samples_per_lap=samples_per_lap)
        laps.append((lap_idx, lap_t0, lap_t1, lap_xy))
        split_times.append(lap_t0)
    if laps:
        split_times.append(laps[-1][2])
    return (laps, np.asarray(split_times, dtype=float))


def _extract_periodic_laps(
    traj: Sequence[TrajectoryPoint],
    period: float,
    samples_per_lap: int,
    min_fraction: float,
) -> tuple[List[tuple[int, float, float, np.ndarray]], np.ndarray]:
    ts, xy = _trajectory_to_arrays(traj)
    total_time = float(ts[-1] - ts[0])
    if total_time < period * min_fraction:
        return ([], np.zeros(0, dtype=float))
    laps: List[tuple[int, float, float, np.ndarray]] = []
    split_times: List[float] = []
    lap_count = int(np.floor(total_time / period)) + 1
    t_start = float(ts[0])
    for lap_idx in range(lap_count):
        lap_t0 = t_start + lap_idx * period
        lap_t1 = min(lap_t0 + period, float(ts[-1]))
        if lap_t1 - lap_t0 < period * min_fraction:
            continue
        lap_xy = _extract_lap_segment(ts=ts, xy=xy, t0=lap_t0, t1=lap_t1, samples_per_lap=samples_per_lap)
        laps.append((lap_idx, lap_t0, lap_t1, lap_xy))
        split_times.append(lap_t0)
    if laps:
        split_times.append(laps[-1][2])
    return (laps, np.asarray(split_times, dtype=float))


def _fourier_smooth_closed_path(points_xy: np.ndarray, harmonics: int) -> np.ndarray:
    loop_xy = np.asarray(points_xy, dtype=float)
    if harmonics <= 0 or loop_xy.shape[0] == 0:
        return loop_xy
    z = loop_xy[:, 0] + 1j * loop_xy[:, 1]
    coeffs = np.fft.fft(z)
    mask = np.zeros_like(coeffs, dtype=bool)
    mask[:harmonics + 1] = True
    mask[-harmonics:] = True
    coeffs[~mask] = 0.0
    smoothed = np.fft.ifft(coeffs)
    return np.column_stack((smoothed.real, smoothed.imag))


def _sample_closed_catmull_rom(control_xy: np.ndarray, sample_count: int) -> np.ndarray:
    controls = np.asarray(control_xy, dtype=float)
    count = controls.shape[0]
    if count == 0:
        return np.zeros((sample_count, 2), dtype=float)
    if count == 1:
        return np.repeat(controls, sample_count, axis=0)
    samples = np.zeros((sample_count, 2), dtype=float)
    for sample_idx, u in enumerate(np.linspace(0.0, float(count), sample_count, endpoint=False)):
        seg_idx = int(np.floor(u)) % count
        t = float(u - np.floor(u))
        p0 = controls[(seg_idx - 1) % count]
        p1 = controls[seg_idx % count]
        p2 = controls[(seg_idx + 1) % count]
        p3 = controls[(seg_idx + 2) % count]
        t2 = t * t
        t3 = t2 * t
        samples[sample_idx] = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
    return samples


def _fit_closed_spline(
    representative_xy: np.ndarray,
    control_count: int,
    sample_count: int,
    anchor: str,
    harmonics: int,
) -> tuple[np.ndarray, np.ndarray]:
    rep = np.asarray(representative_xy, dtype=float)
    rep = _center_points(rep)
    rep = _fourier_smooth_closed_path(rep, harmonics=harmonics)
    control_xy = _resample_closed_polyline_by_arclength(rep, control_count)
    spline_xy = _sample_closed_catmull_rom(control_xy, sample_count)
    if anchor:
        anchor_idx = _choose_loop_anchor(spline_xy, anchor)
        if anchor_idx:
            spline_xy = np.roll(spline_xy, -anchor_idx, axis=0)
            control_shift = int(round(anchor_idx * control_xy.shape[0] / max(1, spline_xy.shape[0])))
            control_xy = np.roll(control_xy, -control_shift, axis=0)
    origin = spline_xy[0].copy()
    spline_xy -= origin
    control_xy -= origin
    return (control_xy, spline_xy)


def _build_canonical_loop(
    aligned_laps: np.ndarray,
    harmonics: int,
    anchor: str,
    statistic: str,
    control_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if statistic == 'mean':
        representative_xy = aligned_laps.mean(axis=0)
    else:
        representative_xy = np.median(aligned_laps, axis=0)
    control_xy, spline_xy = _fit_closed_spline(
        representative_xy=np.asarray(representative_xy, dtype=float),
        control_count=control_count,
        sample_count=aligned_laps.shape[1],
        anchor=anchor,
        harmonics=harmonics,
    )
    loop_xy = np.vstack((spline_xy, spline_xy[0]))
    loop_xy -= loop_xy[0]
    return (control_xy, loop_xy)


def _segments_strictly_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return ((o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0)) and ((o3 > 0.0 and o4 < 0.0) or (o3 < 0.0 and o4 > 0.0))


def _count_self_intersections(points_xy: np.ndarray) -> int:
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] < 4:
        return 0
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    count = int(pts.shape[0])
    if count < 4:
        return 0
    intersections = 0
    for i in range(count):
        a = pts[i]
        b = pts[(i + 1) % count]
        for j in range(i + 2, count):
            if i == 0 and j == count - 1:
                continue
            c = pts[j]
            d = pts[(j + 1) % count]
            if _segments_strictly_intersect(a, b, c, d):
                intersections += 1
    return intersections


def _build_representative_lap_loop(
    laps: Sequence[tuple[int, float, float, np.ndarray]],
    aligned_laps: Sequence[np.ndarray],
    alignment_rmses: Sequence[float],
    keep_mask: Sequence[bool],
    harmonics: int,
    anchor: str,
    control_count: int,
    reference_period: float,
) -> tuple[np.ndarray, np.ndarray]:
    best_candidate: Optional[tuple[float, np.ndarray, np.ndarray]] = None
    for (_lap_index, t0, t1, _), aligned_xy, alignment_rmse, keep in zip(laps, aligned_laps, alignment_rmses, keep_mask):
        if not keep:
            continue
        loop_body = _center_points(aligned_xy)
        if anchor:
            anchor_idx = _choose_loop_anchor(loop_body, anchor)
            if anchor_idx:
                loop_body = np.roll(loop_body, -anchor_idx, axis=0)
        smoothed_body = _fourier_smooth_closed_path(loop_body, harmonics=harmonics)
        closure = float(np.linalg.norm(smoothed_body[0] - smoothed_body[-1]))
        self_intersections = _count_self_intersections(smoothed_body)
        duration_penalty = float(abs((t1 - t0) - reference_period))
        # Favor full observed laps that stay closed after smoothing and avoid synthetic self-crossings.
        score = float(alignment_rmse) + 0.4 * closure + 0.04 * duration_penalty + 10.0 * float(self_intersections)
        control_xy = _resample_closed_polyline_by_arclength(smoothed_body, control_count)
        loop_xy = np.vstack((smoothed_body, smoothed_body[0]))
        origin = loop_xy[0].copy()
        loop_xy -= origin
        control_xy -= origin
        candidate = (score, control_xy, loop_xy)
        if best_candidate is None or score < best_candidate[0]:
            best_candidate = candidate
    if best_candidate is None:
        raise ValueError('No representative lap candidates available.')
    return (best_candidate[1], best_candidate[2])


def _project_points_to_closed_path(
    points_xy: np.ndarray,
    path_xy: np.ndarray,
    search_fraction: float,
) -> tuple[np.ndarray, float]:
    pts = np.asarray(points_xy, dtype=float)
    path = np.asarray(path_xy, dtype=float)
    if pts.shape[0] == 0 or path.shape[0] == 0:
        return (np.zeros_like(pts), 0.0)
    window = max(4, int(round(path.shape[0] * search_fraction)))
    projected = np.zeros_like(pts)
    for point_idx, pt in enumerate(pts):
        expected = int(round(point_idx * path.shape[0] / max(1, pts.shape[0]))) % path.shape[0]
        best_idx = expected
        best_dist = float('inf')
        for offset in range(-window, window + 1):
            idx = (expected + offset) % path.shape[0]
            dist = float(np.sum((path[idx] - pt) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        projected[point_idx] = path[best_idx]
    return (projected, _lap_rmse(projected, pts))


def _loop_xy_to_trajectory(loop_xy: np.ndarray, period: float) -> List[TrajectoryPoint]:
    deriv = np.roll(loop_xy, -1, axis=0) - loop_xy
    yaw = np.arctan2(deriv[:, 1], deriv[:, 0])
    out: List[TrajectoryPoint] = []
    for idx, (pt, yi) in enumerate(zip(loop_xy, yaw)):
        t = period * idx / max(1, loop_xy.shape[0] - 1)
        out.append(TrajectoryPoint(t=float(t), x=float(pt[0]), y=float(pt[1]), yaw=float(yi)))
    return out


def _canonicalize_periodic_trajectory(
    traj: Sequence[TrajectoryPoint],
    cfg: Dict,
) -> tuple[List[TrajectoryPoint], Optional[LoopAveragingDebug]]:
    period = float(cfg.get('loop_period_sec', 0.0) or 0.0)
    if period <= 0.0 or len(traj) < 10:
        return (list(traj), None)
    samples_per_lap = max(64, int(cfg.get('loop_samples', 512) or 512))
    min_fraction = float(cfg.get('loop_min_fraction', 0.6) or 0.6)
    harmonics = max(0, int(cfg.get('loop_fourier_harmonics', 3) or 3))
    refine_iterations = max(1, int(cfg.get('loop_refine_iterations', 3) or 3))
    outlier_sigma = float(cfg.get('loop_outlier_sigma', 2.5) or 2.5)
    allow_scale = bool(cfg.get('loop_similarity_align', True))
    phase_search_fraction = float(cfg.get('loop_phase_search_fraction', 0.15) or 0.15)
    anchor_search_fraction = float(cfg.get('loop_anchor_search_fraction', 0.35) or 0.35)
    projection_search_fraction = float(cfg.get('loop_projection_search_fraction', 0.08) or 0.08)
    control_count = max(8, int(cfg.get('loop_spline_control_points', 24) or 24))
    statistic = str(cfg.get('loop_average_statistic', 'median') or 'median').strip().lower()
    if statistic not in {'mean', 'median'}:
        statistic = 'median'
    loop_strategy = str(cfg.get('loop_strategy', 'auto') or 'auto').strip().lower()
    anchor = str(cfg.get('loop_start_anchor', 'bottom_left') or 'bottom_left').strip()
    laps, split_times = _extract_anchor_laps(
        traj,
        period=period,
        samples_per_lap=samples_per_lap,
        min_fraction=min_fraction,
        anchor=anchor,
        search_radius_fraction=anchor_search_fraction,
    )
    if len(laps) < 2:
        laps, split_times = _extract_periodic_laps(
            traj,
            period=period,
            samples_per_lap=samples_per_lap,
            min_fraction=min_fraction,
        )
    if len(laps) < 2:
        return (list(traj), None)
    lap_period = float(np.median([lap_t1 - lap_t0 for _, lap_t0, lap_t1, _ in laps]))

    template_xy = _center_points(laps[0][3])
    for iteration in range(refine_iterations):
        aligned_laps: List[np.ndarray] = []
        rmses: List[float] = []
        for _, _, _, raw_xy in laps:
            _, aligned_xy, rmse, _ = _align_lap_to_template(
                raw_xy,
                template_xy,
                allow_scale=allow_scale,
                phase_search_fraction=phase_search_fraction,
            )
            aligned_laps.append(aligned_xy)
            rmses.append(rmse)
        keep_mask = np.ones(len(aligned_laps), dtype=bool)
        if iteration > 0:
            keep_mask = _robust_keep_mask(rmses, min_keep=2, sigma=outlier_sigma)
        centered_stack = np.stack([_center_points(aligned_xy) for aligned_xy, keep in zip(aligned_laps, keep_mask) if keep], axis=0)
        if statistic == 'mean':
            template_xy = centered_stack.mean(axis=0)
        else:
            template_xy = np.median(centered_stack, axis=0)

    final_aligned_laps: List[np.ndarray] = []
    final_projected_laps: List[np.ndarray] = []
    final_rmses: List[float] = []
    final_projection_rmses: List[float] = []
    final_shifts: List[int] = []
    final_scales: List[float] = []
    for _, _, _, raw_xy in laps:
        phase_shift, aligned_xy, rmse, scale = _align_lap_to_template(
            raw_xy,
            template_xy,
            allow_scale=allow_scale,
            phase_search_fraction=phase_search_fraction,
        )
        final_shifts.append(phase_shift)
        final_aligned_laps.append(_center_points(aligned_xy))
        final_rmses.append(rmse)
        final_scales.append(scale)

    keep_mask = _robust_keep_mask(final_rmses, min_keep=2, sigma=outlier_sigma)
    aligned_stack = np.stack(
        [aligned_xy for aligned_xy, keep in zip(final_aligned_laps, keep_mask) if keep],
        axis=0,
    )
    candidate_loops: List[tuple[str, np.ndarray, np.ndarray]] = []
    if loop_strategy in {'auto', 'average_spline', 'average'}:
        candidate_loops.append(
            (
                'average_spline',
                *_build_canonical_loop(
                    aligned_stack,
                    harmonics=harmonics,
                    anchor=anchor,
                    statistic=statistic,
                    control_count=control_count,
                ),
            )
        )
    if loop_strategy in {'auto', 'representative_lap', 'representative'}:
        try:
            candidate_loops.append(
                (
                    'representative_lap',
                    *_build_representative_lap_loop(
                        laps=laps,
                        aligned_laps=final_aligned_laps,
                        alignment_rmses=final_rmses,
                        keep_mask=[bool(keep) for keep in keep_mask],
                        harmonics=harmonics,
                        anchor=anchor,
                        control_count=control_count,
                        reference_period=lap_period,
                    ),
                )
            )
        except ValueError:
            pass
    if not candidate_loops:
        candidate_loops.append(
            (
                'average_spline',
                *_build_canonical_loop(
                    aligned_stack,
                    harmonics=harmonics,
                    anchor=anchor,
                    statistic=statistic,
                    control_count=control_count,
                ),
            )
        )
    if loop_strategy == 'auto' and len(candidate_loops) > 1:
        best_candidate: Optional[tuple[float, np.ndarray, np.ndarray]] = None
        kept_aligned = [aligned_xy for aligned_xy, keep in zip(final_aligned_laps, keep_mask) if keep]
        for _, candidate_control_xy, candidate_loop_xy in candidate_loops:
            candidate_path = candidate_loop_xy[:-1]
            projection_rmses = [
                _project_points_to_closed_path(
                    aligned_xy,
                    candidate_path,
                    search_fraction=projection_search_fraction,
                )[1]
                for aligned_xy in kept_aligned
            ]
            projection_score = float(np.median(projection_rmses)) if projection_rmses else float('inf')
            intersection_penalty = 10.0 * float(_count_self_intersections(candidate_loop_xy))
            score = projection_score + intersection_penalty
            if best_candidate is None or score < best_candidate[0]:
                best_candidate = (score, candidate_control_xy, candidate_loop_xy)
        if best_candidate is None:
            _, control_xy, loop_xy = candidate_loops[0]
        else:
            _, control_xy, loop_xy = best_candidate
    else:
        _, control_xy, loop_xy = candidate_loops[0]
    spline_path = loop_xy[:-1]
    for aligned_xy in final_aligned_laps:
        projected_xy, projection_rmse = _project_points_to_closed_path(
            aligned_xy,
            spline_path,
            search_fraction=projection_search_fraction,
        )
        final_projected_laps.append(projected_xy)
        final_projection_rmses.append(projection_rmse)
    loop_debug = LoopAveragingDebug(
        period_sec=lap_period,
        samples_per_lap=samples_per_lap,
        laps=[
            LoopLapDebug(
                lap_index=lap_index,
                t0=t0,
                t1=t1,
                raw_xy=raw_xy,
                aligned_xy=aligned_xy,
                projected_xy=projected_xy,
                phase_shift=phase_shift,
                scale=scale,
                alignment_rmse=rmse,
                projection_rmse=projection_rmse,
                kept=bool(keep),
            )
            for (lap_index, t0, t1, raw_xy), aligned_xy, projected_xy, phase_shift, scale, rmse, projection_rmse, keep in zip(
                laps,
                final_aligned_laps,
                final_projected_laps,
                final_shifts,
                final_scales,
                final_rmses,
                final_projection_rmses,
                keep_mask,
            )
        ],
        canonical_xy=loop_xy,
        spline_control_xy=control_xy,
        split_times=split_times,
    )
    return (_loop_xy_to_trajectory(loop_xy, period=lap_period), loop_debug)


class VisionObservationSmoother:

    def __init__(self, window_frames: int=20) -> None:
        self.window_frames = max(1, int(window_frames))
        self.alpha = 2.0 / (self.window_frames + 1.0)
        self._angle_vec: Optional[np.ndarray] = None
        self._bottom_x: Optional[float] = None

    def _update_angle(self, angle_rad: float) -> float:
        if not np.isfinite(angle_rad):
            return float('nan')
        vec = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=float)
        if self._angle_vec is None:
            self._angle_vec = vec
        else:
            self._angle_vec = (1.0 - self.alpha) * self._angle_vec + self.alpha * vec
            norm = float(np.linalg.norm(self._angle_vec))
            if norm > 1e-9:
                self._angle_vec /= norm
        return float(np.arctan2(self._angle_vec[1], self._angle_vec[0]))

    def _update_bottom_x(self, bottom_x: float) -> float:
        if not np.isfinite(bottom_x):
            return float('nan')
        if self._bottom_x is None:
            self._bottom_x = float(bottom_x)
        else:
            self._bottom_x = (1.0 - self.alpha) * self._bottom_x + self.alpha * float(bottom_x)
        return float(self._bottom_x)

    def update(self, obs: TapeObservation) -> TapeObservation:
        return TapeObservation(
            centerline_px=obs.centerline_px,
            angle_rad=self._update_angle(obs.angle_rad),
            bottom_x=self._update_bottom_x(obs.bottom_x),
            shape_hw=obs.shape_hw,
            mask=obs.mask,
            centerline_mask=obs.centerline_mask,
        )


def _line_observation_confidence(obs: TapeObservation, cfg: Dict) -> float:
    points = np.asarray(obs.centerline_px, dtype=float)
    h, w = obs.shape_hw
    if points.shape[0] < int(cfg.get('line_confidence_min_points', 8) or 8):
        return 0.0
    top_cut = h // 2
    bot_cut = h // 10
    roi_end = max(top_cut + 1, h - bot_cut)
    roi_h = max(1, roi_end - top_cut)
    y_span = float(np.clip((points[:, 1].max() - points[:, 1].min()) / roi_h, 0.0, 1.0))
    point_ratio = float(np.clip(points.shape[0] / roi_h, 0.0, 1.0))
    bottom_y = top_cut + 0.72 * roi_h
    bottom_hit = 1.0 if bool(np.any(points[:, 1] >= bottom_y)) else 0.0
    roi_mask = obs.mask[top_cut:roi_end, :]
    area_ratio = float(np.count_nonzero(roi_mask)) / float(max(1, roi_h * w))
    min_area = float(cfg.get('line_confidence_min_area_ratio', 0.0005) or 0.0005)
    max_area = float(cfg.get('line_confidence_max_area_ratio', 0.35) or 0.35)
    if area_ratio < min_area or area_ratio > max_area:
        area_score = 0.0
    elif area_ratio > 0.22:
        area_score = 0.6
    else:
        area_score = 1.0
    angle_score = 1.0 if np.isfinite(obs.angle_rad) else 0.0
    score = area_score * (
        0.4 * point_ratio
        + 0.25 * y_span
        + 0.25 * bottom_hit
        + 0.1 * angle_score
    )
    return float(np.clip(score, 0.0, 1.0))


def _centered_weighted_average(
    values: np.ndarray,
    weights: np.ndarray,
    window_frames: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = values.shape[0]
    if n == 0:
        return values.copy()
    window = max(1, int(window_frames))
    if window <= 1:
        return values.copy()
    radius = window // 2
    out = np.full(n, np.nan, dtype=float)
    for idx in range(n):
        lo = max(0, idx - radius)
        hi = min(n, idx + radius + 1)
        local_values = values[lo:hi]
        local_weights = weights[lo:hi]
        valid = np.isfinite(local_values) & np.isfinite(local_weights) & (local_weights > 0.0)
        if not bool(np.any(valid)):
            continue
        wsum = float(np.sum(local_weights[valid]))
        if wsum <= 1e-12:
            continue
        out[idx] = float(np.sum(local_values[valid] * local_weights[valid]) / wsum)
    return out


def _centered_weighted_angle_average(
    angles: np.ndarray,
    weights: np.ndarray,
    window_frames: int,
) -> np.ndarray:
    angles = np.asarray(angles, dtype=float)
    sin_avg = _centered_weighted_average(np.sin(angles), weights, window_frames)
    cos_avg = _centered_weighted_average(np.cos(angles), weights, window_frames)
    out = np.full(angles.shape[0], np.nan, dtype=float)
    valid = np.isfinite(sin_avg) & np.isfinite(cos_avg)
    norms = np.sqrt(sin_avg[valid] * sin_avg[valid] + cos_avg[valid] * cos_avg[valid])
    valid_idx = np.flatnonzero(valid)
    keep = norms > 1e-9
    out[valid_idx[keep]] = np.arctan2(sin_avg[valid][keep], cos_avg[valid][keep])
    return out


def _mid_angle(a: float, b: float) -> float:
    return _blend_angles(a, b, 0.5)


def _build_offline_tape_line_trajectory(
    records: Sequence[TapeFrameObservation],
    cfg: Dict,
) -> tuple[List[TrajectoryPoint], float]:
    if not records:
        return ([], 0.0)
    times = np.asarray([record.t for record in records], dtype=float)
    dts = np.asarray([record.dt for record in records], dtype=float)
    delta_yaws = np.asarray([record.delta_yaw for record in records], dtype=float)
    angles = np.asarray([record.obs.angle_rad for record in records], dtype=float)
    bottom_xs = np.asarray([record.obs.bottom_x for record in records], dtype=float)
    widths = np.asarray([record.obs.shape_hw[1] for record in records], dtype=float)
    confidences = np.asarray([record.confidence for record in records], dtype=float)
    min_confidence = float(cfg.get('offline_line_min_confidence', 0.15) or 0.15)
    weights = np.where(confidences >= min_confidence, confidences, 0.0)
    valid_ratio = float(np.mean(weights > 0.0)) if weights.size else 0.0

    smoothing_frames = int(
        cfg.get(
            'offline_line_smoothing_frames',
            cfg.get('vision_smoothing_frames', 20),
        ) or 20
    )
    if bool(cfg.get('offline_tape_smoothing', True)):
        smooth_angles = _centered_weighted_angle_average(angles, weights, smoothing_frames)
        smooth_bottom_xs = _centered_weighted_average(bottom_xs, weights, smoothing_frames)
    else:
        smooth_angles = angles.copy()
        smooth_bottom_xs = bottom_xs.copy()

    v_forward = float(cfg.get('forward_speed_mps', 0.25))
    k_yaw = float(cfg.get('vision_yaw_gain', 0.08))
    k_yaw_nonlinear = float(cfg.get('vision_yaw_nonlinear_gain', 0.0) or 0.0)
    yaw_max = float(cfg.get('vision_yaw_max_correction', 0.12))
    k_lat = float(cfg.get('vision_lateral_gain', 0.002))
    use_imu_translation = bool(cfg.get('imu_use_translation', False))

    x, y, yaw = (0.0, 0.0, 0.0)
    traj: List[TrajectoryPoint] = [TrajectoryPoint(t=float(times[0]), x=x, y=y, yaw=yaw)]
    for idx in range(1, len(records)):
        dt_frame = max(1e-6, float(dts[idx]))
        dyaw_imu = float(delta_yaws[idx]) if np.isfinite(delta_yaws[idx]) else 0.0
        yaw_pred = yaw + dyaw_imu
        conf = float(weights[idx])
        conf_scale = float(np.clip(conf, 0.0, 1.0))
        visual_corr = 0.0
        if np.isfinite(smooth_angles[idx]) and conf_scale > 0.0:
            yaw_error = float(smooth_angles[idx])
            yaw_cmd = k_yaw * yaw_error
            if k_yaw_nonlinear != 0.0:
                yaw_cmd += k_yaw_nonlinear * yaw_error * abs(yaw_error)
            visual_corr = _clamp(yaw_cmd, -yaw_max, yaw_max) * dt_frame * conf_scale
        yaw_next = yaw_pred + visual_corr
        yaw_move = _mid_angle(yaw, yaw_next)

        used_imu_translation = False
        if use_imu_translation:
            delta_p = np.asarray(records[idx].delta_p[:2], dtype=float)
            if np.all(np.isfinite(delta_p)):
                c, s = (np.cos(yaw_move), np.sin(yaw_move))
                R = np.array([[c, -s], [s, c]], dtype=float)
                dp_world = R @ delta_p
                x += float(dp_world[0])
                y += float(dp_world[1])
                used_imu_translation = True
        if not used_imu_translation:
            dist = v_forward * dt_frame
            x += float(dist * np.cos(yaw_move))
            y += float(dist * np.sin(yaw_move))

        if np.isfinite(smooth_bottom_xs[idx]) and conf_scale > 0.0:
            err_px = float(smooth_bottom_xs[idx] - widths[idx] * 0.5)
            dy_body = -k_lat * err_px * dt_frame * conf_scale
            x += float(-dy_body * np.sin(yaw_next))
            y += float(dy_body * np.cos(yaw_next))

        yaw = yaw_next
        traj.append(TrajectoryPoint(t=float(times[idx]), x=x, y=y, yaw=yaw))
    return (traj, valid_ratio)


def _write_tape_line_debug_video(
    video_path: str,
    debug_path: str,
    fps: float,
    records: Sequence[TapeFrameObservation],
    traj: Sequence[TrajectoryPoint],
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    writer = None
    frame_i = 0
    try:
        while frame_i < len(records):
            ok, frame = cap.read()
            if not ok:
                break
            record = records[frame_i]
            point = traj[min(frame_i, len(traj) - 1)]
            h, w = record.obs.shape_hw
            if writer is None:
                fourcc = _mp4v_fourcc()
                writer = cv2.VideoWriter(debug_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
            dbg = frame.copy()
            mask = cv2.resize(
                record.obs.mask,
                (0, 0),
                fx=frame.shape[1] / w,
                fy=frame.shape[0] / h,
                interpolation=cv2.INTER_NEAREST,
            )
            mask_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_col[:, :, 1:] = 0
            dbg = cv2.addWeighted(dbg, 1.0, mask_col, 0.35, 0)
            if np.isfinite(record.obs.bottom_x):
                bx = int(round(record.obs.bottom_x * frame.shape[1] / w))
                cv2.line(dbg, (bx, frame.shape[0] - 1), (bx, int(frame.shape[0] * 0.8)), (0, 255, 255), 2)
            cv2.putText(
                dbg,
                f't={point.t:.2f}s x={point.x:.2f} y={point.y:.2f} yaw={point.yaw:.2f} conf={record.confidence:.2f}',
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            writer.write(dbg)
            frame_i += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def _resize_frame_keep_aspect(frame_bgr: np.ndarray, resize_width: int) -> np.ndarray:
    src_h, src_w = frame_bgr.shape[:2]
    if resize_width <= 0 or src_w <= resize_width:
        return frame_bgr
    width = int(resize_width)
    height = max(1, int(round(src_h * (float(width) / float(src_w)))))
    return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)


def _detect_track_features(gray: np.ndarray, cfg: Dict) -> Optional[np.ndarray]:
    h, w = gray.shape[:2]
    border_fraction = float(cfg.get('vio_border_margin_fraction', 0.04) or 0.04)
    border = max(4, int(round(min(h, w) * border_fraction)))
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[border:max(border + 1, h - border), border:max(border + 1, w - border)] = 255
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(cfg.get('vio_max_corners', 800) or 800),
        qualityLevel=float(cfg.get('vio_quality_level', 0.01) or 0.01),
        minDistance=float(cfg.get('vio_min_distance', 8.0) or 8.0),
        blockSize=int(cfg.get('vio_block_size', 7) or 7),
        mask=mask,
    )
    return points


def _track_features_forward_backward(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    prev_pts: Optional[np.ndarray],
    cfg: Dict,
) -> tuple[np.ndarray, np.ndarray]:
    if prev_pts is None or len(prev_pts) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
    win_size = (int(cfg.get('vio_win_size', 21) or 21), int(cfg.get('vio_win_size', 21) or 21))
    max_level = int(cfg.get('vio_max_pyramid_level', 3) or 3)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_pts,
        np.empty_like(prev_pts),
        winSize=win_size,
        maxLevel=max_level,
        criteria=criteria,
    )
    if next_pts is None or status is None:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
    good_prev = prev_pts[status.flatten() == 1].reshape(-1, 2)
    good_next = next_pts[status.flatten() == 1].reshape(-1, 2)
    if good_prev.shape[0] == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
    back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        prev_gray,
        good_next.reshape(-1, 1, 2),
        np.empty_like(good_next.reshape(-1, 1, 2)),
        winSize=win_size,
        maxLevel=max_level,
        criteria=criteria,
    )
    if back_pts is None or back_status is None:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
    back_pts = back_pts.reshape(-1, 2)
    fb_status = back_status.flatten() == 1
    fb_error = np.linalg.norm(back_pts - good_prev, axis=1)
    keep = fb_status & (fb_error <= float(cfg.get('vio_forward_backward_max_error', 1.5) or 1.5))
    return (good_prev[keep], good_next[keep])


def _estimate_normalized_similarity_transform(
    prev_pts: np.ndarray,
    next_pts: np.ndarray,
    shape_hw: tuple[int, int],
    cfg: Dict,
) -> tuple[Optional[np.ndarray], np.ndarray, np.ndarray, float]:
    min_points = max(8, int(cfg.get('vio_min_points_for_motion', 24) or 24))
    if prev_pts.shape[0] < min_points or next_pts.shape[0] < min_points:
        return (
            None,
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            0.0,
        )
    M, inliers = cv2.estimateAffinePartial2D(
        prev_pts,
        next_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg.get('vio_ransac_reproj_threshold_px', 2.5) or 2.5),
        maxIters=int(cfg.get('vio_ransac_max_iters', 2000) or 2000),
        confidence=float(cfg.get('vio_ransac_confidence', 0.99) or 0.99),
    )
    if M is None or inliers is None:
        return (
            None,
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            0.0,
        )
    inlier_mask = inliers.flatten().astype(bool)
    if int(np.sum(inlier_mask)) < min_points:
        return (
            None,
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            float(np.mean(inlier_mask)) if inlier_mask.size else 0.0,
        )
    h, w = shape_hw
    norm_scale = 1.0 / max(h, w)
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    normalize = np.array(
        [
            [norm_scale, 0.0, -norm_scale * cx],
            [0.0, norm_scale, -norm_scale * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    denormalize = np.linalg.inv(normalize)
    H = np.eye(3, dtype=float)
    H[:2, :] = np.asarray(M, dtype=float)
    H_norm = normalize @ H @ denormalize
    linear = H_norm[:2, :2]
    U, _, Vt = np.linalg.svd(linear)
    rotation = U @ Vt
    if float(np.linalg.det(rotation)) < 0.0:
        U[:, -1] *= -1.0
        rotation = U @ Vt
    H_norm[:2, :2] = rotation
    step_norm = float(np.linalg.norm(H_norm[:2, 2]))
    max_step_norm = float(cfg.get('vio_max_step_norm', 0.08) or 0.08)
    if step_norm > max_step_norm > 0.0:
        H_norm[:2, 2] *= max_step_norm / max(step_norm, 1e-9)
    inlier_ratio = float(np.mean(inlier_mask))
    return (H_norm, prev_pts[inlier_mask], next_pts[inlier_mask], inlier_ratio)


def _make_frame_descriptor(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= float(small.mean())
    small /= max(float(small.std()), 1e-6)
    vec = small.reshape(-1)
    vec /= max(float(np.linalg.norm(vec)), 1e-6)
    return vec.astype(np.float32)


def _estimate_loop_period_from_descriptors(
    times: Sequence[float],
    descriptors: Sequence[np.ndarray],
    cfg: Dict,
) -> Optional[float]:
    if len(times) < 16 or len(descriptors) != len(times):
        return None
    ts = np.asarray(times, dtype=float)
    desc = np.asarray(descriptors, dtype=np.float32)
    if ts.size < 2:
        return None
    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 1e-6)]
    if dt.size == 0:
        return None
    median_dt = float(np.median(dt))
    total_time = float(ts[-1] - ts[0])
    min_period = float(cfg.get('auto_loop_min_period_sec', max(10.0, total_time * 0.12)) or max(10.0, total_time * 0.12))
    max_period = float(cfg.get('auto_loop_max_period_sec', min(90.0, total_time * 0.8)) or min(90.0, total_time * 0.8))
    min_lag = max(1, int(round(min_period / median_dt)))
    max_lag = min(desc.shape[0] - 2, int(round(max_period / median_dt)))
    if max_lag <= min_lag:
        return None
    candidate_scores: List[tuple[float, int]] = []
    for lag in range(min_lag, max_lag + 1):
        sims = np.sum(desc[:-lag] * desc[lag:], axis=1)
        if sims.size < 8:
            continue
        trim = max(0, int(round(0.1 * sims.size)))
        if trim > 0 and sims.size > 2 * trim:
            sims = np.sort(sims)[trim:-trim]
        score = float(np.mean(sims))
        candidate_scores.append((score, lag))
    if not candidate_scores:
        return None
    scores = np.array([score for score, _ in candidate_scores], dtype=float)
    lags = np.array([lag for _, lag in candidate_scores], dtype=int)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    baseline = float(np.median(scores))
    min_gain = float(cfg.get('auto_loop_min_score_gain', 0.02) or 0.02)
    min_score = float(cfg.get('auto_loop_min_score', 0.25) or 0.25)
    if best_score < min_score or best_score < baseline + min_gain:
        return None
    lo = max(0, best_idx - 1)
    hi = min(scores.size, best_idx + 2)
    local_scores = scores[lo:hi]
    local_lags = lags[lo:hi].astype(float)
    refined_lag = float(np.sum(local_scores * local_lags) / max(np.sum(local_scores), 1e-9))
    return refined_lag * median_dt


def _resolve_trajectory_mode(cfg: Dict, imu_samples: Optional[Sequence[IMUSample]]) -> str:
    mode = str(cfg.get('trajectory_mode', 'auto') or 'auto').strip().lower()
    if mode in {'tape', 'tape_line', 'legacy', 'line'}:
        return 'tape_line'
    if mode in {'generic_vio', 'vio', 'generic', 'visual_inertial'}:
        return 'generic_vio'
    if mode == 'auto':
        return 'generic_vio' if imu_samples is not None else 'tape_line'
    return 'tape_line'


def _generic_vio_result_is_usable(result: TrajectoryEstimateResult, cfg: Dict) -> bool:
    if not result.raw_traj or len(result.raw_traj) < 8:
        return False
    xy = np.array([[p.x, p.y] for p in result.raw_traj], dtype=float)
    deltas = np.diff(xy, axis=0)
    if deltas.size == 0:
        return False
    total_length = float(np.sum(np.linalg.norm(deltas, axis=1)))
    span = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    if span < 1e-9:
        return False
    length_span_ratio = total_length / span
    max_ratio = float(cfg.get('auto_generic_max_length_span_ratio', 18.0) or 18.0)
    return bool(np.isfinite(length_span_ratio) and length_span_ratio <= max_ratio)


def _tape_line_result_is_usable(result: TrajectoryEstimateResult, cfg: Dict) -> bool:
    if not result.raw_traj or len(result.raw_traj) < 8:
        return False
    min_valid_ratio = float(cfg.get('auto_tape_min_valid_ratio', 0.2) or 0.2)
    if result.line_valid_ratio is None or result.line_valid_ratio < min_valid_ratio:
        return False
    xy = np.array([[p.x, p.y] for p in result.raw_traj], dtype=float)
    span = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    return bool(np.isfinite(span) and span > 1e-9)


def _estimate_generic_vio_trajectory_with_details(
    video_path: str,
    imu_samples: Optional[Sequence[IMUSample]],
    cfg: Dict,
    save_debug_video: Optional[str]=None,
    frame_timestamps: Optional[Sequence[float]]=None,
) -> TrajectoryEstimateResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    max_frames = int(cfg.get('max_frames', 0) or 0)
    resize_width = int(cfg.get('vio_resize_width', 960) or 960)
    redetect_threshold = int(cfg.get('vio_redetect_threshold', 120) or 120)
    imu_rotation_weight = float(cfg.get('vio_imu_rotation_weight', 0.85) or 0.85)
    auto_loop_enabled = bool(cfg.get('auto_loop_period', True))
    descriptor_stride = max(1, int(cfg.get('auto_loop_descriptor_stride', 5) or 5))
    pim = IMUPreintegrationWrapper(params=_build_imu_preintegration_params(cfg))
    pose = np.eye(3, dtype=float)
    prev_gray: Optional[np.ndarray] = None
    prev_pts: Optional[np.ndarray] = None
    frame_descriptors: List[np.ndarray] = []
    frame_descriptor_times: List[float] = []
    traj: List[TrajectoryPoint] = []
    imu_idx = 0
    last_t: Optional[float] = None
    writer = None
    if save_debug_video is not None:
        fourcc = _mp4v_fourcc()
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if max_frames and frame_i > max_frames:
            break
        frame_resized = _resize_frame_keep_aspect(frame, resize_width=resize_width)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        t = _frame_time_at(cap, frame_i, frame_timestamps)
        if frame_i % descriptor_stride == 0:
            frame_descriptors.append(_make_frame_descriptor(gray))
            frame_descriptor_times.append(t)
        if prev_gray is None:
            prev_gray = gray
            prev_pts = _detect_track_features(gray, cfg)
            last_t = t
            traj.append(TrajectoryPoint(t=t, x=0.0, y=0.0, yaw=0.0))
            continue
        if prev_pts is None or len(prev_pts) < redetect_threshold:
            prev_pts = _detect_track_features(prev_gray, cfg)
        tracked_prev, tracked_next = _track_features_forward_backward(prev_gray, gray, prev_pts, cfg)
        H_norm, inlier_prev, inlier_next, inlier_ratio = _estimate_normalized_similarity_transform(
            tracked_prev,
            tracked_next,
            shape_hw=(int(gray.shape[0]), int(gray.shape[1])),
            cfg=cfg,
        )
        dyaw_imu = float('nan')
        if imu_samples is not None and last_t is not None and len(imu_samples) >= 2:
            seg, imu_idx = slice_imu(imu_samples, last_t, t, start_idx=imu_idx)
            res = pim.preintegrate(seg, reset=True)
            dyaw_imu = float(res.delta_yaw)
        if H_norm is not None:
            visual_yaw = float(np.arctan2(H_norm[1, 0], H_norm[0, 0]))
            if np.isfinite(dyaw_imu):
                adaptive_weight = _clamp(imu_rotation_weight + 0.2 * (0.6 - inlier_ratio), 0.0, 1.0)
                fused_yaw = _blend_angles(visual_yaw, dyaw_imu, adaptive_weight)
            else:
                fused_yaw = visual_yaw
            step = np.eye(3, dtype=float)
            c, s = (np.cos(fused_yaw), np.sin(fused_yaw))
            step[:2, :2] = np.array([[c, -s], [s, c]], dtype=float)
            step[:2, 2] = H_norm[:2, 2]
        else:
            fused_yaw = 0.0 if not np.isfinite(dyaw_imu) else dyaw_imu
            step = np.eye(3, dtype=float)
            c, s = (np.cos(fused_yaw), np.sin(fused_yaw))
            step[:2, :2] = np.array([[c, -s], [s, c]], dtype=float)
        pose = pose @ np.linalg.inv(step)
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        traj.append(TrajectoryPoint(t=t, x=float(pose[0, 2]), y=float(pose[1, 2]), yaw=yaw))
        if save_debug_video is not None:
            if writer is None:
                writer = cv2.VideoWriter(save_debug_video, fourcc, fps, (frame_resized.shape[1], frame_resized.shape[0]))
            dbg = frame_resized.copy()
            for p0, p1 in zip(inlier_prev.astype(int), inlier_next.astype(int)):
                cv2.line(dbg, tuple(p0), tuple(p1), (0, 200, 255), 1, cv2.LINE_AA)
                cv2.circle(dbg, tuple(p1), 2, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(
                dbg,
                f't={t:.2f}s x={pose[0, 2]:.3f} y={pose[1, 2]:.3f} yaw={yaw:.2f} inliers={len(inlier_next)}',
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            writer.write(dbg)
        prev_gray = gray
        prev_pts = inlier_next.reshape(-1, 1, 2) if len(inlier_next) >= redetect_threshold else _detect_track_features(gray, cfg)
        last_t = t
    cap.release()
    if writer is not None:
        writer.release()
    raw_traj = list(traj)
    final_traj = raw_traj
    loop_debug = None
    estimated_period = None
    enable_loop_average = bool(cfg.get('loop_average', True))
    if enable_loop_average and raw_traj:
        loop_cfg = dict(cfg)
        loop_period = float(loop_cfg.get('loop_period_sec', 0.0) or 0.0)
        if loop_period <= 0.0 and auto_loop_enabled:
            estimated_period = _estimate_loop_period_from_descriptors(frame_descriptor_times, frame_descriptors, loop_cfg)
            if estimated_period is not None:
                loop_period = estimated_period
        if loop_period > 0.0:
            loop_cfg['loop_period_sec'] = loop_period
            try:
                final_traj, loop_debug = _canonicalize_periodic_trajectory(raw_traj, loop_cfg)
            except Exception:
                final_traj = raw_traj
                loop_debug = None
    return TrajectoryEstimateResult(
        raw_traj=raw_traj,
        final_traj=final_traj,
        loop_debug=loop_debug,
        mode='generic_vio',
        relative_scale=True,
        estimated_loop_period_sec=loop_debug.period_sec if loop_debug is not None else estimated_period,
    )


def _estimate_tape_line_trajectory_with_details(
    video_path: str,
    imu_samples: Optional[Sequence[IMUSample]],
    cfg: Dict,
    save_debug_video: Optional[str]=None,
    frame_timestamps: Optional[Sequence[float]]=None,
) -> TrajectoryEstimateResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    max_frames = int(cfg.get('max_frames', 0) or 0)
    detector = LineDetector(resize_width=int(cfg.get('resize_width', 640)))
    pim = IMUPreintegrationWrapper(params=_build_imu_preintegration_params(cfg))
    auto_loop_enabled = bool(cfg.get('auto_loop_period', True))
    descriptor_stride = max(1, int(cfg.get('auto_loop_descriptor_stride', 5) or 5))
    descriptor_resize_width = int(cfg.get('auto_loop_resize_width', 320) or 320)
    imu_idx = 0
    last_t: Optional[float] = None
    records: List[TapeFrameObservation] = []
    frame_descriptors: List[np.ndarray] = []
    frame_descriptor_times: List[float] = []
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if max_frames and frame_i > max_frames:
            break
        t = _frame_time_at(cap, frame_i, frame_timestamps)
        if frame_i % descriptor_stride == 0:
            descriptor_frame = _resize_frame_keep_aspect(frame, resize_width=descriptor_resize_width)
            descriptor_gray = cv2.cvtColor(descriptor_frame, cv2.COLOR_BGR2GRAY)
            frame_descriptors.append(_make_frame_descriptor(descriptor_gray))
            frame_descriptor_times.append(t)
        obs = detector.process(frame, cfg)
        confidence = _line_observation_confidence(obs, cfg)
        if last_t is None:
            last_t = t
            records.append(
                TapeFrameObservation(
                    t=t,
                    dt=0.0,
                    obs=obs,
                    confidence=confidence,
                    delta_yaw=0.0,
                    delta_p=np.zeros(3, dtype=float),
                )
            )
            continue
        dt_frame = max(1e-6, float(t - last_t))
        delta_yaw = 0.0
        delta_p = np.zeros(3, dtype=float)
        if imu_samples is not None and len(imu_samples) >= 2:
            seg, imu_idx = slice_imu(imu_samples, last_t, t, start_idx=imu_idx)
            res = pim.preintegrate(seg, reset=True)
            delta_yaw = float(res.delta_yaw)
            delta_p = np.asarray(res.delta_p, dtype=float).reshape(3)
        records.append(
            TapeFrameObservation(
                t=t,
                dt=dt_frame,
                obs=obs,
                confidence=confidence,
                delta_yaw=delta_yaw,
                delta_p=delta_p,
            )
        )
        last_t = t
    cap.release()
    raw_traj, line_valid_ratio = _build_offline_tape_line_trajectory(records, cfg)
    if save_debug_video is not None and records:
        _write_tape_line_debug_video(video_path, save_debug_video, fps, records, raw_traj)

    final_traj = raw_traj
    loop_debug = None
    estimated_period = None
    if bool(cfg.get('loop_average', False)):
        loop_cfg = dict(cfg)
        loop_period = float(loop_cfg.get('loop_period_sec', 0.0) or 0.0)
        if loop_period <= 0.0 and auto_loop_enabled:
            estimated_period = _estimate_loop_period_from_descriptors(frame_descriptor_times, frame_descriptors, loop_cfg)
            if estimated_period is not None:
                loop_period = estimated_period
                loop_cfg['loop_period_sec'] = loop_period
        if loop_period > 0.0:
            final_traj, loop_debug = _canonicalize_periodic_trajectory(raw_traj, loop_cfg)
    reported_period = None
    if loop_debug is not None:
        reported_period = loop_debug.period_sec
    return TrajectoryEstimateResult(
        raw_traj=raw_traj,
        final_traj=final_traj,
        loop_debug=loop_debug,
        mode='tape_line',
        relative_scale=False,
        estimated_loop_period_sec=reported_period,
        line_valid_ratio=line_valid_ratio,
    )


def estimate_trajectory_with_details(
    video_path: str,
    imu_samples: Optional[Sequence[IMUSample]],
    cfg: Dict,
    save_debug_video: Optional[str]=None,
    frame_timestamps: Optional[Sequence[float]]=None,
) -> TrajectoryEstimateResult:
    requested_mode = str(cfg.get('trajectory_mode', 'auto') or 'auto').strip().lower()
    if requested_mode == 'auto' and imu_samples is not None:
        if bool(cfg.get('auto_prefer_tape_line', False)):
            try:
                tape_result = _estimate_tape_line_trajectory_with_details(
                    video_path=video_path,
                    imu_samples=imu_samples,
                    cfg=cfg,
                    save_debug_video=save_debug_video,
                    frame_timestamps=frame_timestamps,
                )
                if _tape_line_result_is_usable(tape_result, cfg):
                    return tape_result
            except Exception:
                pass
        try:
            generic_result = _estimate_generic_vio_trajectory_with_details(
                video_path=video_path,
                imu_samples=imu_samples,
                cfg=cfg,
                save_debug_video=save_debug_video,
                frame_timestamps=frame_timestamps,
            )
            if _generic_vio_result_is_usable(generic_result, cfg):
                return generic_result
        except Exception:
            pass
        return _estimate_tape_line_trajectory_with_details(
            video_path=video_path,
            imu_samples=imu_samples,
            cfg=cfg,
            save_debug_video=save_debug_video,
            frame_timestamps=frame_timestamps,
        )
    mode = _resolve_trajectory_mode(cfg, imu_samples)
    if mode == 'generic_vio':
        return _estimate_generic_vio_trajectory_with_details(
            video_path=video_path,
            imu_samples=imu_samples,
            cfg=cfg,
            save_debug_video=save_debug_video,
            frame_timestamps=frame_timestamps,
        )
    return _estimate_tape_line_trajectory_with_details(
        video_path=video_path,
        imu_samples=imu_samples,
        cfg=cfg,
        save_debug_video=save_debug_video,
        frame_timestamps=frame_timestamps,
    )


def estimate_trajectory(
    video_path: str,
    imu_samples: Optional[Sequence[IMUSample]],
    cfg: Dict,
    save_debug_video: Optional[str]=None,
    frame_timestamps: Optional[Sequence[float]]=None,
) -> List[TrajectoryPoint]:
    return estimate_trajectory_with_details(
        video_path=video_path,
        imu_samples=imu_samples,
        cfg=cfg,
        save_debug_video=save_debug_video,
        frame_timestamps=frame_timestamps,
    ).final_traj
