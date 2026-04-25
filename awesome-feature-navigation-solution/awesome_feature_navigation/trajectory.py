from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .imu_io import slice_imu
from .imu_preintegration import IMUSample, IMUPreintegrationWrapper, build_default_params
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


def _clamp(x: float, lo: float, hi: float) -> float:
    """Ограничить x отрезком [lo, hi]."""
    return float(max(lo, min(hi, x)))


def _rigid_align_points(points_xy: np.ndarray, ref_xy: np.ndarray) -> np.ndarray:
    """Жёстко выровнять points_xy к ref_xy (поворот + сдвиг) методом Кабша через SVD."""
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
    """Подобное (similarity) выравнивание points_xy к ref_xy: поворот + сдвиг + масштаб в [scale_limits]."""
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
    """Развернуть последовательность TrajectoryPoint в (timestamps[N], xy[N, 2])."""
    ts = np.array([p.t for p in traj], dtype=float)
    xy = np.array([[p.x, p.y] for p in traj], dtype=float)
    return (ts, xy)


def _center_points(points_xy: np.ndarray) -> np.ndarray:
    """Сдвинуть точки так, чтобы их центр масс оказался в начале координат."""
    pts = np.asarray(points_xy, dtype=float)
    return pts - pts.mean(axis=0, keepdims=True)


def _lap_rmse(points_xy: np.ndarray, ref_xy: np.ndarray) -> float:
    """RMSE между двумя массивами точек одинаковой формы (евклидово расстояние)."""
    diff = np.asarray(points_xy, dtype=float) - np.asarray(ref_xy, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _anchor_scores_from_normalized(norm_xy: np.ndarray, anchor: str) -> np.ndarray:
    """Скор близости каждой точки (в нормализованных координатах [0,1]) к якорному углу/стороне."""
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
    """Скор близости к якорю в исходных координатах (предварительно нормализует bbox в [0,1])."""
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=float)
    min_xy = pts.min(axis=0, keepdims=True)
    span_xy = np.maximum(pts.max(axis=0, keepdims=True) - min_xy, 1e-9)
    norm = (pts - min_xy) / span_xy
    return _anchor_scores_from_normalized(norm, anchor)


def _resample_polyline_by_arclength(points_xy: np.ndarray, sample_count: int) -> np.ndarray:
    """Равномерно (по длине дуги) пересэмплировать ломаную в sample_count точек."""
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
    """То же, что _resample_polyline_by_arclength, но для замкнутого контура (зацикливает первую/последнюю точку)."""
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
    """Подобрать сдвиг фазы и аффинное выравнивание круга к шаблону, минимизируя RMSE."""
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
    """Робастная маска оставленных кругов по робастному z-score (через MAD); гарантирует ≥ min_keep."""
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
    """Индекс точки замкнутого контура, ближайшей к якорю (например, 'bottom_left')."""
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
    """Вырезать круг из траектории по [t0, t1] и пересэмплировать в samples_per_lap точек по дуге."""
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
    """Подобрать моменты времени-границы кругов, перебирая фазовый сдвиг и ища точки, ближайшие к якорю."""
    if period <= 0.0:
        return []
    search_radius = max(period * 0.1, period * search_radius_fraction)
    context_radius = period * 0.6
    min_gap = period * min_gap_fraction
    offset_steps = 64

    def collect_for_targets(targets: np.ndarray) -> tuple[List[tuple[int, float]], List[float]]:
        """Для каждого целевого момента найти лучшую точку-кандидат на границу круга, отфильтровать близкие."""
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
    """Извлечь круги по якорным границам: вернуть [(idx, t0, t1, xy_resampled)] и моменты разбиения."""
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
    """Fallback-нарезка кругов фиксированной длительности period (без якоря)."""
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
    """Сгладить замкнутый контур, оставив только первые `harmonics` гармоник комплексного FFT."""
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
    """Выбрать sample_count точек на замкнутом сплайне Catmull-Rom по control_xy."""
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
    """Подогнать замкнутый сплайн через FFT-сглаживание + Catmull-Rom; вернуть (control_xy, spline_xy)."""
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
    """Усреднить выровненные круги (mean/median) и подогнать замкнутый сплайн как канонический контур."""
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


def _project_points_to_closed_path(
    points_xy: np.ndarray,
    path_xy: np.ndarray,
    search_fraction: float,
) -> tuple[np.ndarray, float]:
    """Спроецировать каждую точку на ближайшую вершину замкнутого пути (с локальным окном поиска)."""
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
    """Превратить XY-замкнутый контур в TrajectoryPoint-список (yaw — направление касательной)."""
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
    """Каноникализировать периодическую траекторию: нарезать на круги, выровнять, усреднить, аппроксимировать сплайном."""
    period = float(cfg.get('loop_period_sec', 0.0) or 0.0)
    if period <= 0.0 or len(traj) < 10:
        return (list(traj), None)
    samples_per_lap = max(64, int(cfg.get('loop_samples', 512) or 512))
    min_fraction = float(cfg.get('loop_min_fraction', 0.6) or 0.6)
    harmonics = max(0, int(cfg.get('loop_fourier_harmonics', 5) or 5))
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
    control_xy, loop_xy = _build_canonical_loop(
        aligned_stack,
        harmonics=harmonics,
        anchor=anchor,
        statistic=statistic,
        control_count=control_count,
    )
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
    """EMA-сглаживание угла ленты и положения её основания между кадрами (стабилизация шумного детектора)."""

    def __init__(self, window_frames: int=20) -> None:
        """Создать сглаживатель; window_frames задаёт эффективное окно EMA (alpha = 2/(N+1))."""
        self.window_frames = max(1, int(window_frames))
        self.alpha = 2.0 / (self.window_frames + 1.0)
        self._angle_vec: Optional[np.ndarray] = None
        self._bottom_x: Optional[float] = None

    def _update_angle(self, angle_rad: float) -> float:
        """EMA-сглаживание угла на единичной окружности (через cos/sin, чтобы не было разрыва на ±π)."""
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
        """EMA-сглаживание X-координаты основания ленты в нижней части кадра."""
        if not np.isfinite(bottom_x):
            return float('nan')
        if self._bottom_x is None:
            self._bottom_x = float(bottom_x)
        else:
            self._bottom_x = (1.0 - self.alpha) * self._bottom_x + self.alpha * float(bottom_x)
        return float(self._bottom_x)

    def update(self, obs: TapeObservation) -> TapeObservation:
        """Применить EMA к angle_rad и bottom_x наблюдения; маски и центральная линия проходят без изменений."""
        return TapeObservation(
            centerline_px=obs.centerline_px,
            angle_rad=self._update_angle(obs.angle_rad),
            bottom_x=self._update_bottom_x(obs.bottom_x),
            shape_hw=obs.shape_hw,
            mask=obs.mask,
            centerline_mask=obs.centerline_mask,
        )


def estimate_trajectory_with_details(
    video_path: str,
    imu_samples: Optional[Sequence[IMUSample]],
    cfg: Dict,
    save_debug_video: Optional[str]=None,
) -> TrajectoryEstimateResult:
    """Главный пайплайн: видео + IMU → 2D-траектория робота с опциональным сглаживанием по периоду круга."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    max_frames = int(cfg.get('max_frames', 0) or 0)
    detector = LineDetector(resize_width=int(cfg.get('resize_width', 640)))
    pim = IMUPreintegrationWrapper(params=build_default_params())
    x, y, yaw = (0.0, 0.0, 0.0)
    v_forward = float(cfg.get('forward_speed_mps', 0.25))
    k_yaw = float(cfg.get('vision_yaw_gain', 0.08))
    yaw_max = float(cfg.get('vision_yaw_max_correction', 0.12))
    k_lat = float(cfg.get('vision_lateral_gain', 0.002))
    k_lat_yaw = float(cfg.get('vision_lat_yaw_gain', 0.0001))
    vision_smoothing_frames = int(cfg.get('vision_smoothing_frames', 20))
    use_imu_translation = bool(cfg.get('imu_use_translation', False))
    vision_smoother = VisionObservationSmoother(window_frames=vision_smoothing_frames)
    imu_idx = 0
    last_t: Optional[float] = None
    traj: List[TrajectoryPoint] = []
    writer = None
    if save_debug_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if max_frames and frame_i > max_frames:
            break
        t = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if last_t is None:
            last_t = t
            obs = vision_smoother.update(detector.process(frame, cfg))
            traj.append(TrajectoryPoint(t=t, x=x, y=y, yaw=yaw))
            continue
        dt_frame = max(1e-6, float(t - last_t))
        obs = vision_smoother.update(detector.process(frame, cfg))
        used_imu_translation = False
        if imu_samples is not None and len(imu_samples) >= 2:
            seg, imu_idx = slice_imu(imu_samples, last_t, t, start_idx=imu_idx)
            res = pim.preintegrate(seg, reset=True)
            yaw += float(res.delta_yaw)
            if use_imu_translation:
                dp_body = res.delta_p[:2].astype(float)
                c, s = (np.cos(yaw), np.sin(yaw))
                R = np.array([[c, -s], [s, c]], dtype=float)
                dp_world = R @ dp_body
                x += float(dp_world[0])
                y += float(dp_world[1])
                used_imu_translation = True
        if not used_imu_translation:
            dist = v_forward * dt_frame
            x += float(dist * np.cos(yaw))
            y += float(dist * np.sin(yaw))
        h, w = obs.shape_hw
        cx_img = w * 0.5
        if np.isfinite(obs.angle_rad):
            corr = _clamp(k_yaw * float(obs.angle_rad), -yaw_max, yaw_max) * dt_frame
            yaw += corr
        if np.isfinite(obs.bottom_x):
            err_px = float(obs.bottom_x - cx_img)
            dy_body = -k_lat * err_px * dt_frame
            x += float(-dy_body * np.sin(yaw))
            y += float(dy_body * np.cos(yaw))
            yaw -= k_lat_yaw * err_px * dt_frame
        traj.append(TrajectoryPoint(t=t, x=x, y=y, yaw=yaw))
        if save_debug_video is not None:
            if writer is None:
                writer = cv2.VideoWriter(save_debug_video, fourcc, fps, (frame.shape[1], frame.shape[0]))
            dbg = frame.copy()
            mask = cv2.resize(
                obs.mask,
                (0, 0),
                fx=frame.shape[1] / w,
                fy=frame.shape[0] / h,
                interpolation=cv2.INTER_NEAREST,
            )
            mask_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_col[:, :, 1:] = 0
            alpha = 0.35
            dbg = cv2.addWeighted(dbg, 1.0, mask_col, alpha, 0)
            if np.isfinite(obs.bottom_x):
                bx = int(round(obs.bottom_x * frame.shape[1] / w))
                cv2.line(dbg, (bx, frame.shape[0] - 1), (bx, int(frame.shape[0] * 0.8)), (0, 255, 255), 2)
            cv2.putText(
                dbg,
                f't={t:.2f}s x={x:.2f} y={y:.2f} yaw={yaw:.2f}',
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            writer.write(dbg)
        last_t = t
    cap.release()
    if writer is not None:
        writer.release()

    raw_traj = list(traj)
    final_traj = raw_traj
    loop_debug = None
    if bool(cfg.get('loop_average', False)):
        final_traj, loop_debug = _canonicalize_periodic_trajectory(raw_traj, cfg)
    return TrajectoryEstimateResult(raw_traj=raw_traj, final_traj=final_traj, loop_debug=loop_debug)


def estimate_trajectory(
    video_path: str,
    imu_samples: Optional[Sequence[IMUSample]],
    cfg: Dict,
    save_debug_video: Optional[str]=None,
) -> List[TrajectoryPoint]:
    """Обёртка над estimate_trajectory_with_details, возвращающая только итоговую траекторию."""
    return estimate_trajectory_with_details(
        video_path=video_path,
        imu_samples=imu_samples,
        cfg=cfg,
        save_debug_video=save_debug_video,
    ).final_traj
