from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional
import yaml
from .auto_config import apply_auto_video_config
from .calibration import (
    load_camchain_calibration_config,
    load_frame_timestamps_csv,
    load_imu_calibration_config,
)
from .imu_io import calibrate_imu_samples, load_imu_csv, shift_imu_samples
from .line_detection import SUPPORTED_COLORS
from .plotting import save_loop_debug_csv, save_loop_debug_plot, save_trajectory_csv, save_trajectory_plot
from .trajectory import estimate_trajectory_with_details

def _load_cfg(path: Optional[str]) -> Dict:
    cfg: Dict = {}
    if path is not None:
        cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    return cfg

def _resolve_path(value: object, config_path: Optional[Path]) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if config_path is not None:
        from_config = config_path.parent / path
        if from_config.exists():
            return from_config
    return path

def _cfg_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {'0', 'false', 'no', 'off'}
    return bool(value)

def _looks_like_absolute_seconds(value: float) -> bool:
    return abs(float(value)) > 1_000_000.0

def _merge_calibration_cfg(cfg: Dict, path_value: object, config_path: Optional[Path], loader) -> Optional[Path]:
    if path_value is None:
        return None
    path = _resolve_path(path_value, config_path)
    cfg.update(loader(path))
    return path

def _normalize_time_base(frame_timestamps, imu_samples, cfg: Dict):
    if frame_timestamps is None:
        return (None, imu_samples, False)
    frame_timestamps = [float(t) for t in frame_timestamps]
    if not frame_timestamps:
        return (frame_timestamps, imu_samples, False)
    frame_t0 = frame_timestamps[0]
    sync_mode = str(cfg.get('sync_time_base', 'auto') or 'auto').strip().lower()
    synced = False
    if imu_samples is not None and sync_mode != 'none':
        imu_t0 = float(imu_samples[0].t) if imu_samples else 0.0
        should_sync = sync_mode in {'true', 'yes', 'on', 'frame'} or (
            sync_mode == 'auto'
            and _looks_like_absolute_seconds(frame_t0)
            and _looks_like_absolute_seconds(imu_t0)
        )
        if should_sync:
            frame_timestamps = [t - frame_t0 for t in frame_timestamps]
            imu_samples = shift_imu_samples(imu_samples, -frame_t0)
            synced = True
            return (frame_timestamps, imu_samples, synced)
    if _cfg_bool(cfg.get('frame_timestamp_normalize'), True):
        frame_timestamps = [t - frame_t0 for t in frame_timestamps]
    return (frame_timestamps, imu_samples, synced)

def main() -> None:
    ap = argparse.ArgumentParser(description='Colored-line navigation: video + IMU -> 2D trajectory')
    ap.add_argument('--video', required=True, help='Path to video (mp4)')
    ap.add_argument('--imu', default=None, help='Path to IMU CSV. If omitted, runs in demo mode.')
    ap.add_argument('--mode', choices=('auto', 'tape_line', 'generic_vio'), default=None, help='Trajectory estimator: legacy tape-following or generic visual-inertial odometry')
    ap.add_argument('--imu-time-scale', type=float, default=None, help='Multiply IMU timestamps by this factor (e.g. 1e-9 for ns)')
    ap.add_argument('--imu-gyro-scale', type=float, default=None, help='Multiply IMU gyro values by this factor (e.g. 0.0174533 for deg/s -> rad/s)')
    ap.add_argument('--frame-timestamps', default=None, help='CSV with frame_idx,timestamp_ns exported from SVO/ZED')
    ap.add_argument('--frame-time-scale', type=float, default=None, help='Multiply frame timestamps by this factor (default: 1e-9)')
    ap.add_argument('--imu-calibration', default=None, help='Kalibr IMU YAML with noise densities/random walk')
    ap.add_argument('--camchain', default=None, help='Kalibr camchain YAML with T_cam_imu and timeshift_cam_imu')
    ap.add_argument('--config', default=None, help='YAML config (thresholds + gains)')
    ap.add_argument('--color', choices=SUPPORTED_COLORS, default=None, help='Target tape color preset')
    ap.add_argument('--auto-color', action='store_true', help='Auto-adjust HSV thresholds for the selected color from video brightness')
    ap.add_argument('--imu-use-translation', action='store_true', help='Use IMU preintegrated translation instead of constant forward speed')
    ap.add_argument('--no-auto-config', action='store_true', help='Disable automatic line color/HSV detection from the video')
    ap.add_argument('--save-loop-debug', action='store_true', help='Save raw trajectory and loop diagnostics in addition to the final trajectory')
    ap.add_argument('--out', default='trajectory', help='Output prefix (without extension)')
    ap.add_argument('--save-debug', action='store_true', help='Save debug overlay video')
    args = ap.parse_args()
    config_path = Path(args.config).resolve() if args.config is not None else None
    cfg = _load_cfg(args.config)
    if args.mode is not None:
        cfg['trajectory_mode'] = args.mode
    if args.color is not None:
        cfg['target_color'] = args.color
    if args.auto_color:
        cfg['auto_color_tune'] = True
    if args.imu_gyro_scale is not None:
        cfg['imu_gyro_scale'] = args.imu_gyro_scale
    if args.imu_use_translation:
        cfg['imu_use_translation'] = True
    if args.frame_timestamps is not None:
        cfg['frame_timestamps'] = args.frame_timestamps
    if args.frame_time_scale is not None:
        cfg['frame_timestamp_time_scale'] = args.frame_time_scale
    if args.imu_calibration is not None:
        cfg['imu_calibration'] = args.imu_calibration
    if args.camchain is not None:
        cfg['camchain_calibration'] = args.camchain
    if args.no_auto_config:
        cfg['auto_video_config'] = False
    imu_calibration_path = _merge_calibration_cfg(
        cfg,
        cfg.get('imu_calibration'),
        config_path,
        load_imu_calibration_config,
    )
    camchain_path = _merge_calibration_cfg(
        cfg,
        cfg.get('camchain_calibration'),
        config_path,
        load_camchain_calibration_config,
    )
    if camchain_path is not None:
        cfg.setdefault('imu_apply_camchain_rotation', True)
        cfg.setdefault('imu_apply_camchain_timeshift', True)
        cfg.setdefault('imu_align_gravity', True)
        cfg.setdefault('imu_yaw_axis', 'z')
        cfg.setdefault('imu_yaw_only', True)
        cfg.setdefault('imu_yaw_bias_window_sec', 1.0)
    cfg, auto_line = apply_auto_video_config(args.video, cfg)
    out_prefix = Path(args.out)
    frame_timestamps = None
    frame_timestamp_path = None
    if cfg.get('frame_timestamps') is not None:
        frame_timestamp_path = _resolve_path(cfg['frame_timestamps'], config_path)
        frame_time_scale = float(cfg.get('frame_timestamp_time_scale', 1.0e-9))
        frame_timestamps = load_frame_timestamps_csv(frame_timestamp_path, time_scale=frame_time_scale)
    imu_samples = None
    if args.imu is not None:
        imu_time_scale = float(args.imu_time_scale if args.imu_time_scale is not None else cfg.get('imu_time_scale', 1.0))
        imu_gyro_scale = float(args.imu_gyro_scale if args.imu_gyro_scale is not None else cfg.get('imu_gyro_scale', 1.0))
        imu_samples = load_imu_csv(args.imu, time_scale=imu_time_scale, gyro_scale=imu_gyro_scale)
    frame_timestamps, imu_samples, synced_time_base = _normalize_time_base(frame_timestamps, imu_samples, cfg)
    if imu_samples is not None:
        if (
            _cfg_bool(cfg.get('imu_apply_camchain_timeshift'), False)
            and cfg.get('imu_timeshift_cam_imu_sec') is not None
        ):
            # Kalibr convention: t_imu = t_cam + timeshift_cam_imu.
            # Shift IMU samples onto the camera clock before slicing by frame time.
            imu_samples = shift_imu_samples(imu_samples, -float(cfg['imu_timeshift_cam_imu_sec']))
        imu_samples = calibrate_imu_samples(imu_samples, cfg)
    debug_path = None
    if args.save_debug:
        debug_path = str(Path(str(out_prefix) + '_debug.mp4'))
    result = estimate_trajectory_with_details(
        video_path=args.video,
        imu_samples=imu_samples,
        cfg=cfg,
        save_debug_video=debug_path,
        frame_timestamps=frame_timestamps,
    )
    axis_unit = 'relative units' if result.relative_scale else 'meters'
    final_title = 'Estimated Trajectory (relative units)' if result.relative_scale else 'Estimated Trajectory (Interactive)'
    csv_path = str(Path(str(out_prefix) + '.csv'))
    html_path = str(Path(str(out_prefix) + '.html'))
    save_trajectory_csv(result.final_traj, csv_path)
    save_trajectory_plot(result.final_traj, html_path, title=final_title, axis_unit=axis_unit)
    print(f'Saved: {csv_path}')
    print(f'Saved: {html_path}')
    print(f'Mode: {result.mode}')
    if frame_timestamp_path is not None:
        print(f'Frame timestamps: {frame_timestamp_path} ({len(frame_timestamps or [])} frames)')
    if synced_time_base:
        print('Time sync: normalized video and IMU to the first frame timestamp')
    if imu_calibration_path is not None:
        print(f'IMU calibration: {imu_calibration_path}')
    if camchain_path is not None:
        print(f'Camchain calibration: {camchain_path}')
    if auto_line is not None:
        print(
            'Auto line: '
            f'color={auto_line.target_color}, '
            f'valid={auto_line.valid_ratio:.2f}, '
            f'score={auto_line.mean_score:.2f}'
        )
    if result.estimated_loop_period_sec is not None:
        print(f'Estimated loop period: {result.estimated_loop_period_sec:.2f}s')
    if result.loop_debug is not None and args.save_loop_debug:
        raw_csv_path = str(Path(str(out_prefix) + '_raw.csv'))
        raw_html_path = str(Path(str(out_prefix) + '_raw.html'))
        laps_csv_path = str(Path(str(out_prefix) + '_laps.csv'))
        laps_html_path = str(Path(str(out_prefix) + '_laps.html'))
        save_trajectory_csv(result.raw_traj, raw_csv_path)
        save_trajectory_plot(result.raw_traj, raw_html_path, title='Raw Trajectory Before Loop Averaging', axis_unit=axis_unit)
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
