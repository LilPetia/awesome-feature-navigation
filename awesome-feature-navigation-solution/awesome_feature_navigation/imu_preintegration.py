from dataclasses import dataclass
from typing import Sequence

import numpy as np
import gtsam


@dataclass
class IMUSample:
    t: float
    accel: np.ndarray
    omega: np.ndarray


@dataclass
class IMUPreintegrationResult:
    delta_p: np.ndarray
    delta_yaw: float


def build_default_params(
    gravity_mps2: float = 9.81,
    accel_noise_sigma: float = 0.2,
    gyro_noise_sigma: float = 0.02,
) -> gtsam.PreintegrationParams:
    """Собрать стандартные PreintegrationParams (гравитация вдоль -Z, шумы IMU)."""
    I3 = np.eye(3)
    params = gtsam.PreintegrationParams.MakeSharedU(gravity_mps2)
    params.setAccelerometerCovariance(accel_noise_sigma ** 2 * I3)
    params.setGyroscopeCovariance(gyro_noise_sigma ** 2 * I3)
    params.setIntegrationCovariance(1e-8 * I3)
    return params


class IMUPreintegrationWrapper:
    """Обёртка над gtsam.PreintegratedImuMeasurements: накопление IMU-измерений между ключевыми кадрами."""

    def __init__(
        self,
        params: gtsam.PreintegrationParams,
        initial_bias: gtsam.imuBias.ConstantBias | None = None,
    ):
        self._bias = initial_bias if initial_bias is not None else gtsam.imuBias.ConstantBias()
        self._pim = gtsam.PreintegratedImuMeasurements(params, self._bias)

    def reset(self, bias: gtsam.imuBias.ConstantBias | None = None) -> None:
        """Сбросить накопленные дельты; опционально обновить bias."""
        if bias is not None:
            self._bias = bias
            self._pim.resetIntegrationAndSetBias(bias)
        else:
            self._pim.resetIntegration()

    def preintegrate(
        self,
        samples: Sequence[IMUSample],
        reset: bool = True,
    ) -> IMUPreintegrationResult:
        """Преинтегрировать IMU-сэмплы между двумя ключевыми кадрами и вернуть Δp и Δyaw."""
        if len(samples) < 2:
            raise ValueError("Need at least two IMU samples to preintegrate a segment")

        if reset:
            self.reset()

        for s_i, s_j in zip(samples, samples[1:]):
            dt = float(s_j.t - s_i.t)
            if dt <= 0.0:
                raise ValueError("Non positive dt!!!")
            self._pim.integrateMeasurement(s_i.accel, s_i.omega, dt)

        return IMUPreintegrationResult(
            delta_p=np.asarray(self._pim.deltaPij(), dtype=float),
            delta_yaw=float(self._pim.deltaRij().yaw()),
        )
