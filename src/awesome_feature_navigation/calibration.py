from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml


def _norm(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalnum() or ch in '_')


def _pick_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    norm_map = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    for cand in candidates:
        key = _norm(cand)
        for col in cols:
            if key in _norm(col):
                return col
    return None


def _as_float(value: object, default: float=0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_frame_timestamps_csv(path: str | Path, time_scale: float=1.0) -> List[float]:
    df = pd.read_csv(path)
    cols = list(df.columns)
    t_col = _pick_col(cols, ['timestamp_ns', 'timestamp', 'stamp', 'time', 't', 'sec', 'seconds'])
    if t_col is None:
        raise ValueError(f'Cannot find frame timestamp column in {cols}')
    frame_col = _pick_col(cols, ['frame_idx', 'frame', 'idx', 'index'])
    if frame_col is not None:
        df = df.sort_values(frame_col)
    times = df[t_col].astype(float).to_numpy() * float(time_scale)
    if times.size == 0:
        raise ValueError(f'No frame timestamps in {path}')
    if np.any(~np.isfinite(times)):
        raise ValueError(f'Frame timestamps contain non-finite values: {path}')
    return [float(t) for t in times]


def load_imu_calibration_config(path: str | Path) -> Dict[str, object]:
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    imu = data.get('imu0', data) if isinstance(data, Mapping) else {}
    if not isinstance(imu, Mapping):
        return {}
    updates: Dict[str, object] = {}
    field_map = {
        'accelerometer_noise_density': 'imu_accel_noise_density',
        'gyroscope_noise_density': 'imu_gyro_noise_density',
        'accelerometer_random_walk': 'imu_accel_random_walk',
        'gyroscope_random_walk': 'imu_gyro_random_walk',
        'update_rate': 'imu_update_rate_hz',
        'time_offset': 'imu_time_offset_sec',
    }
    for src, dst in field_map.items():
        if src in imu:
            updates[dst] = _as_float(imu[src])
    return updates


def load_camchain_calibration_config(path: str | Path) -> Dict[str, object]:
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    if not isinstance(data, Mapping):
        return {}
    cam_key = next((key for key in sorted(data.keys()) if str(key).startswith('cam')), None)
    if cam_key is None or not isinstance(data.get(cam_key), Mapping):
        return {}
    cam = data[cam_key]
    updates: Dict[str, object] = {'camchain_camera': str(cam_key)}
    T_value = cam.get('T_cam_imu')
    if T_value is not None:
        T = np.asarray(T_value, dtype=float).reshape(4, 4)
        updates['cam_T_imu'] = T.tolist()
        updates['imu_camera_rotation'] = T[:3, :3].tolist()
        updates['imu_camera_translation_m'] = T[:3, 3].tolist()
    if 'timeshift_cam_imu' in cam:
        updates['imu_timeshift_cam_imu_sec'] = _as_float(cam['timeshift_cam_imu'])
    if 'intrinsics' in cam:
        updates['camera_intrinsics'] = [float(v) for v in cam['intrinsics']]
    if 'distortion_coeffs' in cam:
        updates['camera_distortion_coeffs'] = [float(v) for v in cam['distortion_coeffs']]
    for src, dst in (
        ('camera_model', 'camera_model'),
        ('distortion_model', 'camera_distortion_model'),
        ('resolution', 'camera_resolution'),
        ('rostopic', 'camera_rostopic'),
    ):
        if src in cam:
            updates[dst] = cam[src]
    return updates
