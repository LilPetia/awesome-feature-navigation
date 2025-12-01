from dataclasses import dataclass
from typing import Sequence
import numpy as np
import gtsam

@dataclass
class IMUSample:
    """
    Одно измерение IMU.

    t      : время (сек)
    accel  : ускорение / specific force (акселерометр), shape (3,)
    omega  : угловая скорость (гироскоп), shape (3,), в body-фрейме
    """

    t: float
    accel: np.ndarray
    omega: np.ndarray

@dataclass
class IMUPreintegrationResult:
    """
    Результат преинтеграции IMU между двумя ключевыми кадрами.

    delta_R : относительный поворот R_k^T * R_{k+1} (gtsam.Rot3)
    delta_v : изменение скорости в body_k-фрейме (np.ndarray, shape (3,))
    delta_p : смещение / изменение позиции в body_k-фрейме (np.ndarray, shape (3,))
    delta_t : суммарное время интегрирования (сек)
    covariance : ковариация [delta_p, delta_v, delta_theta] (9x9 np.ndarray)
    pim : исходный gtsam.PreintegratedImuMeasurements, чтобы можно было
          прямо передать его в gtsam.ImuFactor / CombinedImuFactor.
    """

    delta_R: gtsam.Rot3
    delta_v: np.ndarray
    delta_p: np.ndarray
    delta_t: float
    covariance: np.ndarray
    pim: gtsam.PreintegratedImuMeasurements


class IMUPreintegrationWrapper:
    """
    Обёртка над gtsam.PreintegratedImuMeasurements
    """
    
    def __init__(self, params: gtsam.PreintegrationParams, initial_bias: gtsam.imuBias.ConstantBias):
        self._params = params
        self._bias = initial_bias
        self._pim: gtsam.PreintegratedImuMeasurements

        self.reset()

    def reset(self, bias: gtsam.imuBias.ConstantBias | None = None) -> None:
        if bias is not None:
            self._bias = bias

        self._pim = gtsam.PreintegratedImuMeasurements(self._params, self._bias)

    @property
    def bias(self) -> gtsam.imuBias.ConstantBias:
        """Текущий bias IMU."""
        return self._bias

    @property
    def params(self) -> gtsam.PreintegrationParams:
        """Параметры преинтеграции (PreintegrationParams)."""
        return self._params

    @property
    def pim(self) -> gtsam.PreintegratedImuMeasurements:
        """Внутренний объект GTSAM (можно напрямую использовать в ImuFactor)."""
        return self._pim
    
    def integrate_segment(
        self,
        samples: Sequence[IMUSample],
        reset_before: bool = True,
    ) -> IMUPreintegrationResult:
        """
        Преинтегрировать набор IMU-измерений между двумя ключевыми кадрами.

        Ожидается, что samples отсортирован по времени и содержит >= 2 сэмплов.

        reset_before :
            если True, перед интеграцией вызывается reset(...) с текущим bias.
            Если False, предынитегрирование будет накапливаться в уже имеющиеся дельты
        """
        if len(samples) < 2:
            raise ValueError("Need at least two IMU samples to preintegrate a segment")

        if reset_before:
            self.reset()

        for i in range(len(samples) - 1):
            s_i = samples[i]
            s_j = samples[i + 1]

            dt = float(s_j.t - s_i.t)
            if dt <= 0.0:
                raise ValueError("Non positive dt!!!")

            accel = np.asarray(s_i.accel, dtype=float).reshape(3)
            omega = np.asarray(s_i.omega, dtype=float).reshape(3)

            self._pim.integrateMeasurement(accel, omega, dt)

        delta_R: gtsam.Rot3 = self._pim.deltaRij()
        delta_v = np.array(self._pim.deltaVij(), dtype=float).reshape(3)
        delta_p = np.array(self._pim.deltaPij(), dtype=float).reshape(3)
        delta_t: float = float(self._pim.deltaTij())
        cov = np.array(self._pim.preintMeasCov(), dtype=float)

        return IMUPreintegrationResult(
            delta_R=delta_R,
            delta_v=delta_v,
            delta_p=delta_p,
            delta_t=delta_t,
            covariance=cov,
            pim=self._pim,
        )

if __name__ == "__main__":
    # Минимальный пример использования

    # 1. Параметры преинтеграции (гравитация вдоль -Z)
    params = gtsam.PreintegrationParams.MakeSharedU(9.81)

    I3 = np.eye(3)
    params.setAccelerometerCovariance(1e-4 * I3)
    params.setGyroscopeCovariance(1e-4 * I3)
    params.setIntegrationCovariance(1e-6 * I3)

    # 2. Начальные bias'ы (нулевые)
    accel_bias = np.zeros(3)
    gyro_bias = np.zeros(3)
    bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)

    # 3. Синтетические IMU-сэмплы:
    #    IMU неподвижен, измеряет только гравитацию along +Z в body-фрейме
    dt = 0.01
    N = 101  # 1 секунда, 100 интервалов
    samples: list[IMUSample] = []
    for i in range(N):
        t = i * dt
        accel = np.array([0.0, 0.0, 9.81])  # specific force: -g (в world) → +g в body z
        omega = np.zeros(3)
        samples.append(IMUSample(t=t, accel=accel, omega=omega))

    # 4. Преинтеграция
    preintegration_wrapper = IMUPreintegrationWrapper(params=params, initial_bias=bias)

    result = preintegration_wrapper.Preintegrate(samples)

    # 5. Вывод результатов
    print("Δt:", result.delta_t)
    print("Δv:", result.delta_v)
    print("Δp:", result.delta_p)
    print("ΔR (matrix):")
    print(result.delta_R.matrix())
    print("covariance shape:", result.covariance.shape)

