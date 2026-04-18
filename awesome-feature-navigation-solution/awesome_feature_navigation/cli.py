from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional
import yaml
from .imu_io import calibrate_imu_samples, load_imu_csv
from .line_detection import SUPPORTED_COLORS
from .plotting import save_loop_debug_csv, save_loop_debug_plot, save_trajectory_csv, save_trajectory_plot
from .trajectory import estimate_trajectory_with_details

def _load_cfg(path: Optional[str]) -> Dict:
    cfg: Dict = {}
    if path is not None:
        cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    return cfg

def main() -> None:
    ap = argparse.ArgumentParser(description='Red-tape navigation: video + IMU -> 2D trajectory')
    ap.add_argument('--video', required=True, help='Path to video (mp4)')
    ap.add_argument('--imu', default=None, help='Path to IMU CSV. If omitted, runs in demo mode.')
    ap.add_argument('--imu-time-scale', type=float, default=None, help='Multiply IMU timestamps by this factor (e.g. 1e-9 for ns)')
    ap.add_argument('--imu-gyro-scale', type=float, default=None, help='Multiply IMU gyro values by this factor (e.g. 0.0174533 for deg/s -> rad/s)')
    ap.add_argument('--config', default=None, help='YAML config (thresholds + gains)')
    ap.add_argument('--color', choices=SUPPORTED_COLORS, default=None, help='Target tape color preset')
    ap.add_argument('--auto-color', action='store_true', help='Auto-adjust HSV thresholds for the selected color from video brightness')
    ap.add_argument('--imu-use-translation', action='store_true', help='Use IMU preintegrated translation instead of constant forward speed')
    ap.add_argument('--out', default='trajectory', help='Output prefix (without extension)')
    ap.add_argument('--save-debug', action='store_true', help='Save debug overlay video')
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.color is not None:
        cfg['target_color'] = args.color
    if args.auto_color:
        cfg['auto_color_tune'] = True
    if args.imu_gyro_scale is not None:
        cfg['imu_gyro_scale'] = args.imu_gyro_scale
    if args.imu_use_translation:
        cfg['imu_use_translation'] = True
    out_prefix = Path(args.out)
    imu_samples = None
    if args.imu is not None:
        imu_time_scale = float(args.imu_time_scale if args.imu_time_scale is not None else cfg.get('imu_time_scale', 1.0))
        imu_gyro_scale = float(args.imu_gyro_scale if args.imu_gyro_scale is not None else cfg.get('imu_gyro_scale', 1.0))
        imu_samples = load_imu_csv(args.imu, time_scale=imu_time_scale, gyro_scale=imu_gyro_scale)
        imu_samples = calibrate_imu_samples(imu_samples, cfg)
    debug_path = None
    if args.save_debug:
        debug_path = str(Path(str(out_prefix) + '_debug.mp4'))
    result = estimate_trajectory_with_details(
        video_path=args.video,
        imu_samples=imu_samples,
        cfg=cfg,
        save_debug_video=debug_path,
    )
    csv_path = str(Path(str(out_prefix) + '.csv'))
    html_path = str(Path(str(out_prefix) + '.html'))
    save_trajectory_csv(result.final_traj, csv_path)
    save_trajectory_plot(result.final_traj, html_path)
    print(f'Saved: {csv_path}')
    print(f'Saved: {html_path}')
    if result.loop_debug is not None:
        raw_csv_path = str(Path(str(out_prefix) + '_raw.csv'))
        raw_html_path = str(Path(str(out_prefix) + '_raw.html'))
        laps_csv_path = str(Path(str(out_prefix) + '_laps.csv'))
        laps_html_path = str(Path(str(out_prefix) + '_laps.html'))
        save_trajectory_csv(result.raw_traj, raw_csv_path)
        save_trajectory_plot(result.raw_traj, raw_html_path, title='Raw Trajectory Before Loop Averaging')
        save_loop_debug_csv(result.loop_debug, laps_csv_path)
        save_loop_debug_plot(result.loop_debug, laps_html_path)
        print(f'Saved: {raw_csv_path}')
        print(f'Saved: {raw_html_path}')
        print(f'Saved: {laps_csv_path}')
        print(f'Saved: {laps_html_path}')
    if debug_path is not None:
        print(f'Saved: {debug_path}')
if __name__ == '__main__':
    main()
