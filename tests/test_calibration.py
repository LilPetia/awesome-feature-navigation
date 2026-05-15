from pathlib import Path

import numpy as np
import pytest

from awesome_feature_navigation.calibration import (
    load_camchain_calibration_config,
    load_frame_timestamps_csv,
    load_imu_calibration_config,
)


def test_load_imu_calibration_config_maps_kalibr_fields(tmp_path: Path) -> None:
    path = tmp_path / 'imu.yaml'
    path.write_text(
        '\n'.join(
            [
                'imu0:',
                '  accelerometer_noise_density: 0.01',
                '  gyroscope_noise_density: 0.02',
                '  accelerometer_random_walk: 0.03',
                '  gyroscope_random_walk: 0.04',
                '  update_rate: 200.0',
                '  time_offset: -0.1',
            ]
        ),
        encoding='utf-8',
    )

    cfg = load_imu_calibration_config(path)

    assert cfg == {
        'imu_accel_noise_density': 0.01,
        'imu_gyro_noise_density': 0.02,
        'imu_accel_random_walk': 0.03,
        'imu_gyro_random_walk': 0.04,
        'imu_update_rate_hz': 200.0,
        'imu_time_offset_sec': -0.1,
    }


def test_load_camchain_calibration_config_extracts_transform_and_camera_metadata(tmp_path: Path) -> None:
    path = tmp_path / 'camchain.yaml'
    path.write_text(
        '\n'.join(
            [
                'cam0:',
                '  T_cam_imu:',
                '    - [1.0, 0.0, 0.0, 0.1]',
                '    - [0.0, 1.0, 0.0, 0.2]',
                '    - [0.0, 0.0, 1.0, 0.3]',
                '    - [0.0, 0.0, 0.0, 1.0]',
                '  timeshift_cam_imu: -0.013',
                '  intrinsics: [400.0, 401.0, 320.0, 240.0]',
                '  distortion_coeffs: [0.1, 0.2, 0.3, 0.4]',
                '  camera_model: pinhole',
                '  distortion_model: radtan',
                '  resolution: [640, 480]',
                '  rostopic: /cam0',
            ]
        ),
        encoding='utf-8',
    )

    cfg = load_camchain_calibration_config(path)

    assert cfg['camchain_camera'] == 'cam0'
    np.testing.assert_allclose(
        np.asarray(cfg['imu_camera_rotation'], dtype=float),
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    assert cfg['imu_camera_translation_m'] == pytest.approx([0.1, 0.2, 0.3])
    assert cfg['imu_timeshift_cam_imu_sec'] == pytest.approx(-0.013)
    assert cfg['camera_intrinsics'] == pytest.approx([400.0, 401.0, 320.0, 240.0])
    assert cfg['camera_distortion_coeffs'] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert cfg['camera_model'] == 'pinhole'
    assert cfg['camera_distortion_model'] == 'radtan'
    assert cfg['camera_resolution'] == [640, 480]
    assert cfg['camera_rostopic'] == '/cam0'


def test_load_frame_timestamps_csv_sorts_by_frame_and_scales_time(tmp_path: Path) -> None:
    path = tmp_path / 'timestamps.csv'
    path.write_text(
        '\n'.join(
            [
                'frame_idx,timestamp_ns',
                '2,3000000000',
                '0,1000000000',
                '1,2000000000',
            ]
        ),
        encoding='utf-8',
    )

    assert load_frame_timestamps_csv(path, time_scale=1.0e-9) == [1.0, 2.0, 3.0]
