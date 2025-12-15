from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .imu_io import slice_imu
from .imu_preintegration import IMUSample, IMUPreintegrationWrapper, build_default_params, rot3_yaw
from .line_detection import LineDetector, TapeObservation


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    yaw: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def estimate_trajectory(
    video_path: str,
    imu_samples: Optional[Sequence[IMUSample]],
    cfg: Dict,
    save_debug_video: Optional[str] = None,
) -> List[TrajectoryPoint]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    max_frames = int(cfg.get("max_frames", 0) or 0)

    detector = LineDetector(resize_width=int(cfg.get("resize_width", 640)))

    # IMU preintegration wrapper
    pim = IMUPreintegrationWrapper(params=build_default_params())

    # State (planar)
    x, y, yaw = 0.0, 0.0, 0.0
    v_forward = float(cfg.get("forward_speed_mps", 0.25))  # used if IMU missing

    # Vision gains
    k_yaw = float(cfg.get("vision_yaw_gain", 0.08))
    yaw_max = float(cfg.get("vision_yaw_max_correction", 0.12))
    k_lat = float(cfg.get("vision_lateral_gain", 0.002))

    # IMU iteration
    imu_idx = 0
    last_t: Optional[float] = None
    traj: List[TrajectoryPoint] = []

    writer = None
    if save_debug_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if max_frames and frame_i > max_frames:
            break

        # Timestamp: prefer CAP_PROP_POS_MSEC after reading (OpenCV gives the time of *current* frame)
        t = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if last_t is None:
            last_t = t
            # still process vision for debug consistency
            obs = detector.process(frame, cfg)
            traj.append(TrajectoryPoint(t=t, x=x, y=y, yaw=yaw))
            continue

        dt_frame = max(1e-6, float(t - last_t))

        obs = detector.process(frame, cfg)

        # 1) Propagation from IMU (if available) else constant speed
        if imu_samples is not None and len(imu_samples) >= 2:
            seg, imu_idx = slice_imu(imu_samples, last_t, t, start_idx=imu_idx)
            res = pim.preintegrate(seg, reset=True)
            dyaw = float(res.delta_yaw)
            # Use delta_p in body_i frame, project to XY
            dp_body = res.delta_p[:2].astype(float)
            # Rotate to world using current yaw (before update)
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]], dtype=float)
            dp_world = R @ dp_body
            x += float(dp_world[0])
            y += float(dp_world[1])
            yaw += float(dyaw)
        else:
            # constant forward motion
            dist = v_forward * dt_frame
            x += float(dist * np.cos(yaw))
            y += float(dist * np.sin(yaw))

        # 2) Vision corrections (heading + lateral)
        h, w = obs.shape_hw
        cx_img = w * 0.5

        if np.isfinite(obs.angle_rad):
            corr = _clamp(k_yaw * float(obs.angle_rad), -yaw_max, yaw_max)
            yaw += corr

        if np.isfinite(obs.bottom_x):
            err_px = float(obs.bottom_x - cx_img)
            # move sideways in body frame to reduce error (x forward, y left)
            dy_body = -k_lat * err_px
            x += float(-dy_body * np.sin(yaw))
            y += float(dy_body * np.cos(yaw))

        traj.append(TrajectoryPoint(t=t, x=x, y=y, yaw=yaw))

        # Debug video overlay
        if save_debug_video is not None:
            if writer is None:
                writer = cv2.VideoWriter(save_debug_video, fourcc, fps, (frame.shape[1], frame.shape[0]))
            dbg = frame.copy()

            # draw tape mask preview in corner
            mask = cv2.resize(obs.mask, (0, 0), fx=frame.shape[1] / w, fy=frame.shape[0] / h, interpolation=cv2.INTER_NEAREST)
            mask_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_col[:, :, 1:] = 0  # red channel only-ish
            alpha = 0.35
            dbg = cv2.addWeighted(dbg, 1.0, mask_col, alpha, 0)

            # bottom_x line
            if np.isfinite(obs.bottom_x):
                bx = int(round(obs.bottom_x * frame.shape[1] / w))
                cv2.line(dbg, (bx, frame.shape[0] - 1), (bx, int(frame.shape[0] * 0.8)), (0, 255, 255), 2)

            # trajectory text
            cv2.putText(dbg, f"t={t:.2f}s x={x:.2f} y={y:.2f} yaw={yaw:.2f}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            writer.write(dbg)

        last_t = t

    cap.release()
    if writer is not None:
        writer.release()
    return traj
