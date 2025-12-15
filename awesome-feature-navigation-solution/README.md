# awesome-feature-navigation (full solution)

This repository produces a **2D trajectory** (x, y over time) from:
1) a video where the robot follows a **red tape** on the floor, and
2) IMU samples (accelerometer + gyroscope).

It includes:
- **Line/tape extraction** -> 1-pixel centerline + line direction + lateral offset.
- **IMU preintegration** (GTSAM) between camera frames.
- A pragmatic **planar fusion** loop to output `(t, x, y)` and a trajectory plot.

> Notes
> - The fusion here is intentionally lightweight (complementary-style). It will not match a full VIO/INS, but is robust enough for coursework/demos and gives a clean trajectory plot.
> - If you want a proper factor-graph optimizer (Pose/Vel/Bias with IMU factors + vision factors), the code is structured so you can swap the fusion module.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Run

### With IMU CSV

```bash
afn-run --video path/to/video.mp4 --imu path/to/imu.csv --out trajectory
```

Outputs:
- `trajectory.csv`
- `trajectory.png`
- optional debug video overlay `trajectory_debug.mp4` (if `--save-debug`)

### Without IMU (demo mode)

```bash
afn-run --video path/to/video.mp4 --out trajectory
```

In this mode we assume a constant forward speed and use the tape direction as heading feedback.

## IMU CSV format

The loader is flexible, but it expects at least:
- timestamp column: one of `t`, `time`, `timestamp`, `sec`, `seconds` (seconds)
- accel columns: `ax, ay, az` (m/s^2) or variations (`accel_x`, `linear_acceleration.x`, etc.)
- gyro columns: `gx, gy, gz` (rad/s) or variations (`gyro_z`, `angular_velocity.z`, etc.)

You can also pass `--imu-time-scale 1e-9` if your timestamps are in nanoseconds.

## Parameters

`--config` points to a YAML with thresholds and gains. See `examples/config.yaml`.
