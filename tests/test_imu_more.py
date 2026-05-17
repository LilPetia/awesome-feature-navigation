from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import awesome_feature_navigation.imu_preintegration as imu_preintegration
from awesome_feature_navigation.imu_io import (
    _apply_axis_map,
    _parse_axis_spec,
    _pick_col,
    _rotation_from_a_to_b,
    _rotation_matrix_from_cfg,
    calibrate_imu_samples,
    load_imu_csv,
    shift_imu_samples,
    slice_imu,
)
from awesome_feature_navigation.imu_preintegration import (
    IMUPreintegrationWrapper,
    IMUSample,
    MinimalRot3,
    _skew,
    _so3_exp,
    build_default_params,
    rot3_yaw,
)


def _sample(t: float, accel: list[float] | None=None, omega: list[float] | None=None) -> IMUSample:
    return IMUSample(
        t=t,
        accel=np.asarray(accel if accel is not None else [0.0, 0.0, -1.0], dtype=float),
        omega=np.asarray(omega if omega is not None else [0.0, 0.0, 0.0], dtype=float),
    )


def test_load_imu_csv_sorts_columns_and_applies_scales(tmp_path: Path) -> None:
    path = tmp_path / 'imu.csv'
    path.write_text(
        '\n'.join(
            [
                'timestamp,linear_acceleration.x,linear_acceleration.y,linear_acceleration.z,angular_velocity.x,angular_velocity.y,angular_velocity.z',
                '2,1,2,3,10,20,30',
                '1,4,5,6,40,50,60',
            ]
        ),
        encoding='utf-8',
    )

    samples = load_imu_csv(str(path), time_scale=0.5, gyro_scale=0.1)

    assert [sample.t for sample in samples] == [0.5, 1.0]
    np.testing.assert_allclose(samples[0].accel, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(samples[0].omega, [4.0, 5.0, 6.0])


def test_load_imu_csv_reports_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / 'bad.csv'
    path.write_text('time,ax\n0,1\n', encoding='utf-8')

    with pytest.raises(ValueError, match='Cannot find IMU columns'):
        load_imu_csv(str(path))

    no_time = tmp_path / 'no_time.csv'
    no_time.write_text('ax,ay,az,gx,gy,gz\n1,2,3,4,5,6\n', encoding='utf-8')
    with pytest.raises(ValueError, match='timestamp column'):
        load_imu_csv(str(no_time))


def test_imu_axis_mapping_rotation_and_yaw_calibration() -> None:
    samples = [
        _sample(0.0, accel=[1.0, 2.0, -1.0], omega=[0.1, 0.2, 0.3]),
        _sample(1.0, accel=[1.0, 2.0, -1.0], omega=[0.2, 0.4, 0.6]),
    ]

    calibrated = calibrate_imu_samples(
        samples,
        {
            'imu_accel_axes': ['y', '-x', 'z'],
            'imu_gyro_axes': ['z', 'y', '-x'],
            'imu_gyro_bias': [0.05, 0.05, 0.05],
            'imu_camera_rotation': np.eye(3).tolist(),
            'imu_apply_camchain_rotation': True,
            'imu_yaw_axis': '-x',
            'imu_yaw_gain': 2.0,
            'imu_yaw_bias_window_sec': 0.1,
            'imu_yaw_only': True,
        },
    )

    np.testing.assert_allclose(calibrated[0].accel, [2.0, -1.0, -1.0])
    np.testing.assert_allclose(calibrated[0].omega, [0.0, 0.0, 0.0])
    assert calibrated[1].omega[2] == pytest.approx(-0.6)


def test_imu_rotation_helpers_handle_edge_cases() -> None:
    assert _pick_col(['sensor_linear_acceleration_x'], ['linear_acceleration_x']) == 'sensor_linear_acceleration_x'
    assert _parse_axis_spec('+z') == (2, 1.0)
    with pytest.raises(ValueError):
        _parse_axis_spec('q')
    values = np.array([[1.0, 2.0, 3.0]], dtype=float)
    np.testing.assert_allclose(_apply_axis_map(values, ['z', '-y', 'x']), [[3.0, -2.0, 1.0]])
    with pytest.raises(ValueError, match='Axis map'):
        _apply_axis_map(values, ['x', 'y'])
    assert _rotation_matrix_from_cfg(None) is None
    assert _rotation_matrix_from_cfg('bad') is None
    flat_rotation = _rotation_matrix_from_cfg(np.arange(9.0))
    flat_transform = _rotation_matrix_from_cfg(np.arange(16.0))
    assert flat_rotation is not None
    assert flat_transform is not None
    np.testing.assert_allclose(flat_rotation, np.arange(9.0).reshape(3, 3))
    np.testing.assert_allclose(flat_transform, np.arange(16.0).reshape(4, 4)[:3, :3])
    rotation = _rotation_matrix_from_cfg(np.eye(4).tolist())
    assert rotation is not None
    assert rotation.shape == (3, 3)
    assert _rotation_matrix_from_cfg([1.0, 0.0]) is None
    np.testing.assert_allclose(_rotation_from_a_to_b(np.zeros(3), np.ones(3)), np.eye(3))
    np.testing.assert_allclose(_rotation_from_a_to_b(np.ones(3), np.ones(3)), np.eye(3))
    np.testing.assert_allclose(
        _rotation_from_a_to_b(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        @ np.array([1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _rotation_from_a_to_b(
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([-1.0, 0.0, 0.0], dtype=float),
        ) @ np.array([1.0, 0.0, 0.0], dtype=float),
        [-1.0, 0.0, 0.0],
    )


def test_imu_calibration_empty_gravity_alignment_and_non_yaw_only() -> None:
    assert calibrate_imu_samples([], {}) == []

    gravity_samples = [
        _sample(0.0, accel=[0.0, 1.0, 0.0], omega=[0.0, 0.0, 0.0]),
        _sample(1.0, accel=[0.0, 1.0, 0.0], omega=[0.0, 0.0, 0.0]),
    ]
    aligned = calibrate_imu_samples(gravity_samples, {'imu_align_gravity': True})
    np.testing.assert_allclose(np.mean([sample.accel for sample in aligned], axis=0), [0.0, 0.0, -1.0], atol=1e-6)

    yaw_samples = [_sample(0.0, omega=[0.1, 0.2, 0.3]), _sample(1.0, omega=[0.2, 0.4, 0.6])]
    calibrated = calibrate_imu_samples(
        yaw_samples,
        {
            'imu_yaw_axis': 'y',
            'imu_yaw_gain': 2.0,
            'imu_yaw_bias': 0.1,
            'imu_yaw_only': False,
        },
    )

    assert calibrated[0].omega[0] == pytest.approx(0.1)
    assert calibrated[0].omega[1] == pytest.approx(0.2)
    assert calibrated[0].omega[2] == pytest.approx(0.3)
    assert calibrated[1].omega[1] == pytest.approx(0.6)


def test_shift_and_slice_imu_samples_include_boundary_samples() -> None:
    samples = [_sample(0.0), _sample(1.0), _sample(2.0), _sample(3.0)]

    shifted = shift_imu_samples(samples, 0.25)
    segment, next_idx = slice_imu(samples, 0.5, 2.5, start_idx=0)

    assert [sample.t for sample in shifted] == [0.25, 1.25, 2.25, 3.25]
    assert [sample.t for sample in segment] == [0.0, 1.0, 2.0, 3.0]
    assert next_idx == 1


def test_minimal_preintegration_integrates_rotation_and_translation() -> None:
    samples = [
        _sample(0.0, accel=[1.0, 0.0, 0.0], omega=[0.0, 0.0, np.pi / 2.0]),
        _sample(1.0, accel=[1.0, 0.0, 0.0], omega=[0.0, 0.0, 0.0]),
    ]
    wrapper = IMUPreintegrationWrapper(params=None)

    result = wrapper.preintegrate(samples)

    assert build_default_params() is None
    assert result.delta_t == pytest.approx(1.0)
    np.testing.assert_allclose(result.delta_v, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.delta_p, [0.5, 0.0, 0.0])
    assert result.delta_yaw == pytest.approx(np.pi / 2.0)


def test_preintegration_handles_short_negative_and_accumulated_sequences() -> None:
    wrapper = IMUPreintegrationWrapper(params=None)
    short = wrapper.preintegrate([_sample(0.0)])
    assert short.delta_t == pytest.approx(0.0)

    first = wrapper.preintegrate(
        [
            _sample(0.0, accel=[1.0, 0.0, 0.0]),
            _sample(-1.0, accel=[1.0, 0.0, 0.0]),
            _sample(0.0, accel=[1.0, 0.0, 0.0]),
        ]
    )
    second = wrapper.preintegrate(
        [_sample(0.0, accel=[1.0, 0.0, 0.0]), _sample(1.0, accel=[1.0, 0.0, 0.0])],
        reset=False,
    )

    assert first.delta_t == pytest.approx(1.0)
    assert second.delta_t == pytest.approx(2.0)


def test_gtsam_preintegration_path_with_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeParams:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def setAccelerometerCovariance(self, value: np.ndarray) -> None:
            self.calls.append('accel')

        def setGyroscopeCovariance(self, value: np.ndarray) -> None:
            self.calls.append('gyro')

        def setIntegrationCovariance(self, value: np.ndarray) -> None:
            self.calls.append('integration')

        def setBiasAccCovariance(self, value: np.ndarray) -> None:
            self.calls.append('bias_acc')

        def setBiasOmegaCovariance(self, value: np.ndarray) -> None:
            self.calls.append('bias_gyro')

        def setBiasAccOmegaInt(self, value: np.ndarray) -> None:
            self.calls.append('bias_cross')

    class FakePreintegrationParams:
        @staticmethod
        def MakeSharedU(gravity: float) -> FakeParams:
            params = FakeParams()
            params.calls.append(f'gravity={gravity}')
            return params

    class FakeBias:
        pass

    class FakeImuBias:
        @staticmethod
        def ConstantBias() -> FakeBias:
            return FakeBias()

    class FakeRot3:
        def __init__(self, yaw: float=0.25) -> None:
            self.yaw = yaw

        def rpy(self) -> tuple[float, float, float]:
            return (0.0, 0.0, self.yaw)

    class FakePIM:
        def __init__(self, params: FakeParams, bias: FakeBias) -> None:
            self.params = params
            self.bias = bias
            self.measurements: list[tuple[np.ndarray, np.ndarray, float]] = []

        def integrateMeasurement(self, accel: np.ndarray, omega: np.ndarray, dt: float) -> None:
            self.measurements.append((accel, omega, dt))

        def deltaRij(self) -> FakeRot3:
            return FakeRot3(0.25)

        def deltaTij(self) -> float:
            return float(sum(dt for _, _, dt in self.measurements))

        def deltaVij(self) -> np.ndarray:
            return np.array([1.0, 2.0, 3.0], dtype=float)

        def deltaPij(self) -> np.ndarray:
            return np.array([0.1, 0.2, 0.3], dtype=float)

        def preintMeasCov(self) -> np.ndarray:
            return np.eye(9)

    class FakeGtsam:
        PreintegrationParams = FakePreintegrationParams
        imuBias = FakeImuBias
        PreintegratedImuMeasurements = FakePIM
        Rot3 = FakeRot3

    monkeypatch.setattr(imu_preintegration, '_HAS_GTSAM', True)
    monkeypatch.setattr(imu_preintegration, 'gtsam', FakeGtsam)

    params = build_default_params(gravity_mps2=9.7)
    assert isinstance(params, FakeParams)
    assert 'gravity=9.7' in params.calls
    assert 'bias_cross' in params.calls

    wrapper = IMUPreintegrationWrapper(params=params)
    wrapper.reset(bias=FakeBias())
    short = wrapper.preintegrate([_sample(0.0)])
    result = wrapper.preintegrate(
        [
            _sample(0.0, accel=[1.0, 0.0, 0.0], omega=[0.1, 0.0, 0.0]),
            _sample(0.5, accel=[0.0, 1.0, 0.0], omega=[0.0, 0.2, 0.0]),
            _sample(0.25, accel=[0.0, 0.0, 1.0], omega=[0.0, 0.0, 0.3]),
            _sample(1.0, accel=[0.0, 0.0, 1.0], omega=[0.0, 0.0, 0.3]),
        ]
    )

    assert isinstance(short.delta_R, FakeRot3)
    assert result.delta_t == pytest.approx(1.25)
    assert result.delta_yaw == pytest.approx(0.25)
    np.testing.assert_allclose(result.delta_v, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result.delta_p, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(result.covariance, np.eye(9))
    assert rot3_yaw(FakeRot3(0.4)) == pytest.approx(0.4)


def test_minimal_rot3_and_so3_helpers() -> None:
    R = _so3_exp(np.array([0.0, 0.0, np.pi / 2.0], dtype=float))
    rot = MinimalRot3(R)

    np.testing.assert_allclose(_skew(np.array([1.0, 2.0, 3.0])), [[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
    np.testing.assert_allclose(_so3_exp(np.zeros(3)), np.eye(3))
    assert rot3_yaw(rot) == pytest.approx(np.pi / 2.0)
    assert MinimalRot3.identity().rpy() == pytest.approx((0.0, 0.0, 0.0))

    gimbal = MinimalRot3(np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=float))
    assert gimbal.rpy()[0] == pytest.approx(0.0)

    class MatrixRot:
        def matrix(self) -> np.ndarray:
            return R

    assert rot3_yaw(MatrixRot()) == pytest.approx(np.pi / 2.0)
