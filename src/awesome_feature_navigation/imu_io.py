from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .imu_preintegration import IMUSample


def _norm(s: str) -> str:
    return ''.join((ch for ch in s.lower() if ch.isalnum() or ch in '_'))

def _pick_col(cols: List[str], candidates: Sequence[str]) -> Optional[str]:
    norm_map = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    for cand in candidates:
        key = _norm(cand)
        for c in cols:
            if key in _norm(c):
                return c
    return None

def _parse_axis_spec(spec: str) -> Tuple[int, float]:
    token = spec.strip().lower()
    sign = 1.0
    if token.startswith('-'):
        sign = -1.0
        token = token[1:]
    elif token.startswith('+'):
        token = token[1:]
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if token not in axis_map:
        raise ValueError(f'Unsupported axis spec: {spec}')
    return (axis_map[token], sign)

def _apply_axis_map(values: np.ndarray, axis_specs: Sequence[str]) -> np.ndarray:
    if len(axis_specs) != 3:
        raise ValueError(f'Axis map must contain 3 entries, got {axis_specs}')
    mapped = np.zeros_like(values, dtype=float)
    for dst_idx, spec in enumerate(axis_specs):
        src_idx, sign = _parse_axis_spec(str(spec))
        mapped[:, dst_idx] = sign * values[:, src_idx]
    return mapped

def _rotation_matrix_from_cfg(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (4, 4):
        return arr[:3, :3]
    if arr.size == 9:
        return arr.reshape(3, 3)
    if arr.size == 16:
        return arr.reshape(4, 4)[:3, :3]
    return None

def _rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-09 or b_norm < 1e-09:
        return np.eye(3)
    a = a / a_norm
    b = b / b_norm
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        v = np.cross(a, axis)
        v = v / float(np.linalg.norm(v))
        K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float)
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float)
    return np.eye(3) + K + (K @ K) * ((1.0 - c) / (s * s))

def load_imu_csv(path: str, time_scale: float=1.0, gyro_scale: float=1.0) -> List[IMUSample]:
    df = pd.read_csv(path)
    cols = list(df.columns)
    t_col = _pick_col(cols, ['t', 'time', 'timestamp', 'sec', 'seconds', 'stamp'])
    if t_col is None:
        raise ValueError(f'Cannot find timestamp column in {cols}')
    ax = _pick_col(cols, ['ax', 'accel_x', 'accelx', 'linear_acceleration_x', 'linearaccelerationx', 'linear_acceleration.x'])
    ay = _pick_col(cols, ['ay', 'accel_y', 'accely', 'linear_acceleration_y', 'linearaccelerationy', 'linear_acceleration.y'])
    az = _pick_col(cols, ['az', 'accel_z', 'accelz', 'linear_acceleration_z', 'linearaccelerationz', 'linear_acceleration.z'])
    gx = _pick_col(cols, ['gx', 'gyro_x', 'gyrox', 'angular_velocity_x', 'angularvelocityx', 'angular_velocity.x'])
    gy = _pick_col(cols, ['gy', 'gyro_y', 'gyroy', 'angular_velocity_y', 'angularvelocityy', 'angular_velocity.y'])
    gz = _pick_col(cols, ['gz', 'gyro_z', 'gyroz', 'angular_velocity_z', 'angularvelocityz', 'angular_velocity.z'])
    missing = [name for name, c in [('ax', ax), ('ay', ay), ('az', az), ('gx', gx), ('gy', gy), ('gz', gz)] if c is None]
    if missing:
        raise ValueError(f'Cannot find IMU columns {missing} in {cols}')
    t = df[t_col].astype(float).to_numpy() * float(time_scale)
    accel = df[[ax, ay, az]].astype(float).to_numpy()
    omega = df[[gx, gy, gz]].astype(float).to_numpy() * float(gyro_scale)
    order = np.argsort(t)
    t = t[order]
    accel = accel[order]
    omega = omega[order]
    samples: List[IMUSample] = []
    for ti, ai, wi in zip(t, accel, omega):
        samples.append(IMUSample(t=float(ti), accel=np.asarray(ai, dtype=float), omega=np.asarray(wi, dtype=float)))
    return samples

def calibrate_imu_samples(samples: Sequence[IMUSample], cfg: Mapping[str, Any]) -> List[IMUSample]:
    if not samples:
        return []
    t = np.array([s.t for s in samples], dtype=float)
    accel = np.array([s.accel for s in samples], dtype=float)
    omega = np.array([s.omega for s in samples], dtype=float)
    accel_axes = cfg.get('imu_accel_axes')
    if isinstance(accel_axes, Sequence) and not isinstance(accel_axes, (str, bytes)):
        accel = _apply_axis_map(accel, [str(item) for item in accel_axes])
    gyro_axes = cfg.get('imu_gyro_axes')
    if isinstance(gyro_axes, Sequence) and not isinstance(gyro_axes, (str, bytes)):
        omega = _apply_axis_map(omega, [str(item) for item in gyro_axes])
    gyro_bias = cfg.get('imu_gyro_bias')
    if isinstance(gyro_bias, Sequence) and not isinstance(gyro_bias, (str, bytes)) and len(gyro_bias) == 3:
        omega = omega - np.asarray([float(item) for item in gyro_bias], dtype=float).reshape(1, 3)
    camera_rotation = _rotation_matrix_from_cfg(cfg.get('imu_camera_rotation'))
    apply_camera_rotation = bool(cfg.get('imu_apply_camchain_rotation', camera_rotation is not None))
    if apply_camera_rotation and camera_rotation is not None:
        accel = (camera_rotation @ accel.T).T
        omega = (camera_rotation @ omega.T).T
    if bool(cfg.get('imu_align_gravity', False)):
        gravity = accel.mean(axis=0)
        R = _rotation_from_a_to_b(gravity, np.array([0.0, 0.0, -1.0], dtype=float))
        accel = (R @ accel.T).T
        omega = (R @ omega.T).T
    yaw_axis = cfg.get('imu_yaw_axis')
    if yaw_axis is not None:
        axis_idx, axis_sign = _parse_axis_spec(str(yaw_axis))
        yaw_bias = float(cfg.get('imu_yaw_bias', 0.0) or 0.0)
        yaw_bias_window_sec = float(cfg.get('imu_yaw_bias_window_sec', 0.0) or 0.0)
        if yaw_bias_window_sec > 0.0:
            t0 = float(t[0])
            mask = t <= t0 + yaw_bias_window_sec
            if bool(np.any(mask)):
                yaw_bias = float(omega[mask, axis_idx].mean())
        yaw_gain = float(cfg.get('imu_yaw_gain', 1.0) or 1.0)
        yaw_rate = axis_sign * yaw_gain * (omega[:, axis_idx] - yaw_bias)
        if bool(cfg.get('imu_yaw_only', True)):
            omega = np.zeros_like(omega)
            omega[:, 2] = yaw_rate
        else:
            omega[:, axis_idx] = yaw_rate
    calibrated: List[IMUSample] = []
    for ti, ai, wi in zip(t, accel, omega):
        calibrated.append(IMUSample(t=float(ti), accel=np.asarray(ai, dtype=float), omega=np.asarray(wi, dtype=float)))
    return calibrated

def shift_imu_samples(samples: Sequence[IMUSample], offset_sec: float) -> List[IMUSample]:
    offset = float(offset_sec)
    return [
        IMUSample(
            t=float(sample.t + offset),
            accel=np.asarray(sample.accel, dtype=float),
            omega=np.asarray(sample.omega, dtype=float),
        )
        for sample in samples
    ]

def slice_imu(samples: Sequence[IMUSample], t0: float, t1: float, start_idx: int=0) -> Tuple[List[IMUSample], int]:
    n = len(samples)
    i = start_idx
    while i < n and samples[i].t < t0:
        i += 1
    j = i
    while j < n and samples[j].t <= t1:
        j += 1
    seg = list(samples[i:j])
    if i > 0 and (len(seg) == 0 or seg[0].t > t0):
        seg.insert(0, samples[i - 1])
    if j < n and (len(seg) == 0 or seg[-1].t < t1):
        seg.append(samples[j])
    return (seg, i)
