import numpy as np
import pytest

from awesome_feature_navigation.imu_io import shift_imu_samples, slice_imu
from awesome_feature_navigation.imu_preintegration import IMUSample


def _sample(t: float) -> IMUSample:
    return IMUSample(
        t=t,
        accel=np.array([1.0, 2.0, 3.0], dtype=float),
        omega=np.array([0.1, 0.2, 0.3], dtype=float),
    )


def test_shift_imu_samples_offsets_time_without_mutating_vectors() -> None:
    samples = [_sample(10.0), _sample(11.0)]

    shifted = shift_imu_samples(samples, -10.0)

    assert [sample.t for sample in shifted] == [0.0, 1.0]
    assert shifted[0].accel.tolist() == [1.0, 2.0, 3.0]
    assert shifted[0].omega.tolist() == [0.1, 0.2, 0.3]


def test_slice_imu_includes_neighbor_samples_for_integration_boundaries() -> None:
    samples = [_sample(0.0), _sample(1.0), _sample(2.0), _sample(3.0)]

    segment, next_idx = slice_imu(samples, 1.5, 2.5)

    assert [sample.t for sample in segment] == [1.0, 2.0, 3.0]
    assert next_idx == 2


def test_slice_imu_returns_empty_segment_when_no_samples_are_available() -> None:
    segment, next_idx = slice_imu([], 0.0, 1.0)

    assert segment == []
    assert next_idx == 0


def test_shift_imu_samples_preserves_numeric_arrays() -> None:
    sample = _sample(1.0)

    shifted = shift_imu_samples([sample], 2.0)[0]

    assert shifted.t == pytest.approx(3.0)
    np.testing.assert_allclose(shifted.accel, sample.accel)
    np.testing.assert_allclose(shifted.omega, sample.omega)
