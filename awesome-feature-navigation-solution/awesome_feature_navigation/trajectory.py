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

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

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

def _trajectory_to_arrays(traj: Sequence['TrajectoryPoint']) -> tuple[np.ndarray, np.ndarray]:
    ts = np.array([p.t for p in traj], dtype=float)
    xy = np.array([[p.x, p.y] for p in traj], dtype=float)
    return (ts, xy)

def _choose_loop_anchor(loop_xy: np.ndarray, anchor: str) -> int:
    pts = np.asarray(loop_xy, dtype=float)
    if pts.shape[0] == 0:
        return 0
    min_xy = pts.min(axis=0, keepdims=True)
    span_xy = np.maximum(pts.max(axis=0, keepdims=True) - min_xy, 1e-9)
    norm = (pts - min_xy) / span_xy
    anchor_key = anchor.strip().lower()
    if anchor_key == 'bottom_left':
        score = norm[:, 0] + norm[:, 1]
        return int(np.argmin(score))
    if anchor_key == 'top_left':
        score = norm[:, 0] + (1.0 - norm[:, 1])
        return int(np.argmin(score))
    if anchor_key == 'top_right':
        score = (1.0 - norm[:, 0]) + (1.0 - norm[:, 1])
        return int(np.argmin(score))
    if anchor_key == 'bottom_right':
        score = (1.0 - norm[:, 0]) + norm[:, 1]
        return int(np.argmin(score))
    if anchor_key == 'left':
        return int(np.argmin(norm[:, 0]))
    if anchor_key == 'right':
        return int(np.argmax(norm[:, 0]))
    if anchor_key == 'top':
        return int(np.argmax(norm[:, 1]))
    if anchor_key == 'bottom':
        return int(np.argmin(norm[:, 1]))
    return 0

def _canonicalize_periodic_trajectory(traj: Sequence['TrajectoryPoint'], cfg: Dict) -> List['TrajectoryPoint']:
    period = float(cfg.get('loop_period_sec', 0.0) or 0.0)
    if period <= 0.0 or len(traj) < 10:
        return list(traj)
    samples_per_lap = max(64, int(cfg.get('loop_samples', 512) or 512))
    min_fraction = float(cfg.get('loop_min_fraction', 0.6) or 0.6)
    harmonics = max(0, int(cfg.get('loop_fourier_harmonics', 5) or 5))
    ts, xy = _trajectory_to_arrays(traj)
    total_time = float(ts[-1] - ts[0])
    if total_time < period * min_fraction:
        return list(traj)
    phase = np.linspace(0.0, 1.0, samples_per_lap, endpoint=False)
    laps: List[np.ndarray] = []
    lap_count = int(np.floor(total_time / period)) + 1
    t_start = float(ts[0])
    for lap_idx in range(lap_count):
        lap_t0 = t_start + lap_idx * period
        lap_t1 = min(lap_t0 + period, float(ts[-1]))
        if lap_t1 - lap_t0 < period * min_fraction:
            continue
        query_t = lap_t0 + phase * period
        x = np.interp(query_t, ts, xy[:, 0])
        y = np.interp(query_t, ts, xy[:, 1])
        laps.append(np.column_stack((x, y)))
    if len(laps) < 2:
        return list(traj)
    ref = laps[0]
    aligned = [ref]
    for lap in laps[1:]:
        aligned.append(_rigid_align_points(lap, ref))
    stack = np.stack(aligned, axis=0)
    deltas = np.roll(stack, -1, axis=1) - stack
    mean_delta = deltas.mean(axis=0)
    mean_delta -= mean_delta.mean(axis=0, keepdims=True)
    loop_xy = np.zeros((samples_per_lap, 2), dtype=float)
    for i in range(1, samples_per_lap):
        loop_xy[i] = loop_xy[i - 1] + mean_delta[i - 1]
    loop_xy -= loop_xy.mean(axis=0, keepdims=True)
    if harmonics > 0:
        z = loop_xy[:, 0] + 1j * loop_xy[:, 1]
        coeffs = np.fft.fft(z)
        mask = np.zeros_like(coeffs, dtype=bool)
        mask[:harmonics + 1] = True
        mask[-harmonics:] = True
        coeffs[~mask] = 0.0
        z = np.fft.ifft(coeffs)
        loop_xy = np.column_stack((z.real, z.imag))
    anchor = str(cfg.get('loop_start_anchor', '') or '').strip()
    if anchor:
        anchor_idx = _choose_loop_anchor(loop_xy, anchor)
        loop_xy = np.roll(loop_xy, -anchor_idx, axis=0)
    loop_xy = np.vstack((loop_xy, loop_xy[0]))
    loop_xy -= loop_xy[0]
    deriv = np.roll(loop_xy, -1, axis=0) - loop_xy
    yaw = np.arctan2(deriv[:, 1], deriv[:, 0])
    out: List[TrajectoryPoint] = []
    for idx, (pt, yi) in enumerate(zip(loop_xy, yaw)):
        t = period * idx / max(1, loop_xy.shape[0] - 1)
        out.append(TrajectoryPoint(t=float(t), x=float(pt[0]), y=float(pt[1]), yaw=float(yi)))
    return out

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
            if norm > 1e-09:
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
        return TapeObservation(centerline_px=obs.centerline_px, angle_rad=self._update_angle(obs.angle_rad), bottom_x=self._update_bottom_x(obs.bottom_x), shape_hw=obs.shape_hw, mask=obs.mask, centerline_mask=obs.centerline_mask)

def estimate_trajectory(video_path: str, imu_samples: Optional[Sequence[IMUSample]], cfg: Dict, save_debug_video: Optional[str]=None) -> List[TrajectoryPoint]:
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
        dt_frame = max(1e-06, float(t - last_t))
        obs = vision_smoother.update(detector.process(frame, cfg))
        used_imu_translation = False
        if imu_samples is not None and len(imu_samples) >= 2:
            seg, imu_idx = slice_imu(imu_samples, last_t, t, start_idx=imu_idx)
            res = pim.preintegrate(seg, reset=True)
            dyaw = float(res.delta_yaw)
            yaw += float(dyaw)
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
        traj.append(TrajectoryPoint(t=t, x=x, y=y, yaw=yaw))
        if save_debug_video is not None:
            if writer is None:
                writer = cv2.VideoWriter(save_debug_video, fourcc, fps, (frame.shape[1], frame.shape[0]))
            dbg = frame.copy()
            mask = cv2.resize(obs.mask, (0, 0), fx=frame.shape[1] / w, fy=frame.shape[0] / h, interpolation=cv2.INTER_NEAREST)
            mask_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_col[:, :, 1:] = 0
            alpha = 0.35
            dbg = cv2.addWeighted(dbg, 1.0, mask_col, alpha, 0)
            if np.isfinite(obs.bottom_x):
                bx = int(round(obs.bottom_x * frame.shape[1] / w))
                cv2.line(dbg, (bx, frame.shape[0] - 1), (bx, int(frame.shape[0] * 0.8)), (0, 255, 255), 2)
            cv2.putText(dbg, f't={t:.2f}s x={x:.2f} y={y:.2f} yaw={yaw:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.write(dbg)
        last_t = t
    cap.release()
    if writer is not None:
        writer.release()
    if bool(cfg.get('loop_average', False)):
        return _canonicalize_periodic_trajectory(traj, cfg)
    return traj
