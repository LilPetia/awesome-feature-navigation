from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import yaml

from .imu_io import load_imu_csv
from .trajectory import estimate_trajectory
from .plotting import save_trajectory_csv, save_trajectory_plot


def _load_cfg(path: Optional[str]) -> Dict:
    cfg: Dict = {}
    if path is not None:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Red-tape navigation: video + IMU -> 2D trajectory")
    ap.add_argument("--video", required=True, help="Path to video (mp4)")
    ap.add_argument("--imu", default=None, help="Path to IMU CSV. If omitted, runs in demo mode.")
    ap.add_argument("--imu-time-scale", type=float, default=1.0, help="Multiply IMU timestamps by this factor (e.g. 1e-9 for ns)")
    ap.add_argument("--config", default=None, help="YAML config (thresholds + gains)")
    ap.add_argument("--out", default="trajectory", help="Output prefix (without extension)")
    ap.add_argument("--save-debug", action="store_true", help="Save debug overlay video")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    out_prefix = Path(args.out)

    imu_samples = None
    if args.imu is not None:
        imu_samples = load_imu_csv(args.imu, time_scale=args.imu_time_scale)

    debug_path = None
    if args.save_debug:
        debug_path = str(out_prefix.with_suffix("_debug.mp4"))
        # workaround: pathlib doesn't like with_suffix for non-extension; build manually
        debug_path = str(Path(str(out_prefix) + "_debug.mp4"))

    traj = estimate_trajectory(
        video_path=args.video,
        imu_samples=imu_samples,
        cfg=cfg,
        save_debug_video=debug_path,
    )

    csv_path = str(Path(str(out_prefix) + ".csv"))
    png_path = str(Path(str(out_prefix) + ".png"))
    save_trajectory_csv(traj, csv_path)
    save_trajectory_plot(traj, png_path)

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")
    if debug_path is not None:
        print(f"Saved: {debug_path}")


if __name__ == "__main__":
    main()
