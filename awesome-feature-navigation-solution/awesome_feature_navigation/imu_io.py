from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .imu_preintegration import IMUSample


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch in "_")


def _pick_col(cols: List[str], candidates: Sequence[str]) -> Optional[str]:
    norm_map = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    # fuzzy: contains
    for cand in candidates:
        key = _norm(cand)
        for c in cols:
            if key in _norm(c):
                return c
    return None


def load_imu_csv(path: str, time_scale: float = 1.0) -> List[IMUSample]:
    """Load IMU csv with flexible column names.

    Parameters
    ----------
    time_scale:
        Multiply timestamps by this factor (e.g. 1e-9 if timestamps are in nanoseconds).
    """
    df = pd.read_csv(path)
    cols = list(df.columns)

    t_col = _pick_col(cols, ["t", "time", "timestamp", "sec", "seconds", "stamp"])
    if t_col is None:
        raise ValueError(f"Cannot find timestamp column in {cols}")

    ax = _pick_col(cols, ["ax", "accel_x", "accelx", "linear_acceleration_x", "linearaccelerationx", "linear_acceleration.x"])
    ay = _pick_col(cols, ["ay", "accel_y", "accely", "linear_acceleration_y", "linearaccelerationy", "linear_acceleration.y"])
    az = _pick_col(cols, ["az", "accel_z", "accelz", "linear_acceleration_z", "linearaccelerationz", "linear_acceleration.z"])

    gx = _pick_col(cols, ["gx", "gyro_x", "gyrox", "angular_velocity_x", "angularvelocityx", "angular_velocity.x"])
    gy = _pick_col(cols, ["gy", "gyro_y", "gyroy", "angular_velocity_y", "angularvelocityy", "angular_velocity.y"])
    gz = _pick_col(cols, ["gz", "gyro_z", "gyroz", "angular_velocity_z", "angularvelocityz", "angular_velocity.z"])

    missing = [name for name, c in [("ax", ax), ("ay", ay), ("az", az), ("gx", gx), ("gy", gy), ("gz", gz)] if c is None]
    if missing:
        raise ValueError(f"Cannot find IMU columns {missing} in {cols}")

    t = (df[t_col].astype(float).to_numpy()) * float(time_scale)
    accel = df[[ax, ay, az]].astype(float).to_numpy()
    omega = df[[gx, gy, gz]].astype(float).to_numpy()

    # ensure time-sorted
    order = np.argsort(t)
    t = t[order]
    accel = accel[order]
    omega = omega[order]

    samples: List[IMUSample] = []
    for ti, ai, wi in zip(t, accel, omega):
        samples.append(IMUSample(t=float(ti), accel=np.asarray(ai, dtype=float), omega=np.asarray(wi, dtype=float)))
    return samples


def slice_imu(samples: Sequence[IMUSample], t0: float, t1: float, start_idx: int = 0) -> Tuple[List[IMUSample], int]:
    """Return samples with t in [t0, t1], and next start index."""
    n = len(samples)
    i = start_idx
    while i < n and samples[i].t < t0:
        i += 1
    j = i
    while j < n and samples[j].t <= t1:
        j += 1
    seg = list(samples[i:j])
    # Ensure endpoints for integration stability: include one sample before and after if possible
    if i > 0 and (len(seg) == 0 or seg[0].t > t0):
        seg.insert(0, samples[i - 1])
    if j < n and (len(seg) == 0 or seg[-1].t < t1):
        seg.append(samples[j])
    return seg, i
