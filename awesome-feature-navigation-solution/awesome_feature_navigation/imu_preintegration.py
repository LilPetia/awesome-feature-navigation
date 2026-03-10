from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
try:
    import gtsam
    _HAS_GTSAM = True
except Exception:
    gtsam = None
    _HAS_GTSAM = False

class MinimalRot3:

    def __init__(self, R: Optional[np.ndarray]=None) -> None:
        self._R = np.eye(3) if R is None else np.asarray(R, dtype=float).reshape(3, 3)

    def matrix(self) -> np.ndarray:
        return self._R.copy()

    def rpy(self) -> tuple[float, float, float]:
        R = self._R
        pitch = float(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))
        if abs(np.cos(pitch)) < 1e-08:
            roll = 0.0
            yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
        else:
            roll = float(np.arctan2(R[2, 1], R[2, 2]))
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return (roll, pitch, yaw)

    @staticmethod
    def identity() -> 'MinimalRot3':
        return MinimalRot3(np.eye(3))
Rot3Like = 'gtsam.Rot3' if _HAS_GTSAM else MinimalRot3

@dataclass(frozen=True)
class IMUSample:
    t: float
    accel: np.ndarray
    omega: np.ndarray

@dataclass(frozen=True)
class PreintegrationResult:
    delta_t: float
    delta_R: object
    delta_v: np.ndarray
    delta_p: np.ndarray
    covariance: np.ndarray
    delta_yaw: float

def build_default_params(gravity_mps2: float=9.81, accel_noise_sigma: float=0.2, gyro_noise_sigma: float=0.02, accel_bias_rw_sigma: float=0.0005, gyro_bias_rw_sigma: float=0.0002):
    if not _HAS_GTSAM:
        return None
    params = gtsam.PreintegrationParams.MakeSharedU(gravity_mps2)
    params.setAccelerometerCovariance(accel_noise_sigma ** 2 * np.eye(3))
    params.setGyroscopeCovariance(gyro_noise_sigma ** 2 * np.eye(3))
    params.setIntegrationCovariance(1e-08 * np.eye(3))
    params.setBiasAccCovariance(accel_bias_rw_sigma ** 2 * np.eye(3))
    params.setBiasOmegaCovariance(gyro_bias_rw_sigma ** 2 * np.eye(3))
    params.setBiasAccOmegaInt(1e-08 * np.eye(6))
    return params

def _skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=float)

def _so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + _skew(w)
    K = _skew(w / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

class IMUPreintegrationWrapper:

    def __init__(self, params=None, initial_bias=None) -> None:
        self._use_gtsam = _HAS_GTSAM and params is not None
        self._params = params
        self._bias = initial_bias
        self.reset()

    def reset(self, bias=None) -> None:
        if self._use_gtsam:
            if bias is not None:
                self._bias = bias
            if self._bias is None:
                self._bias = gtsam.imuBias.ConstantBias()
            self._pim = gtsam.PreintegratedImuMeasurements(self._params, self._bias)
        else:
            self._R = np.eye(3)
            self._v = np.zeros(3)
            self._p = np.zeros(3)
            self._dt = 0.0

    def preintegrate(self, samples: Sequence[IMUSample], reset: bool=True) -> PreintegrationResult:
        if len(samples) < 2:
            return PreintegrationResult(delta_t=0.0, delta_R=gtsam.Rot3() if _HAS_GTSAM else MinimalRot3.identity(), delta_v=np.zeros(3), delta_p=np.zeros(3), covariance=np.zeros((9, 9)), delta_yaw=0.0)
        if reset:
            self.reset()
        if self._use_gtsam:
            for s0, s1 in zip(samples[:-1], samples[1:]):
                dt = float(s1.t - s0.t)
                if dt <= 0:
                    continue
                self._pim.integrateMeasurement(np.asarray(s0.accel, dtype=float).reshape(3), np.asarray(s0.omega, dtype=float).reshape(3), dt)
            pim = self._pim
            rot = pim.deltaRij()
            dyaw = rot3_yaw(rot)
            return PreintegrationResult(delta_t=float(pim.deltaTij()), delta_R=rot, delta_v=np.array(pim.deltaVij(), dtype=float).reshape(3), delta_p=np.array(pim.deltaPij(), dtype=float).reshape(3), covariance=np.array(pim.preintMeasCov(), dtype=float), delta_yaw=dyaw)
        for s0, s1 in zip(samples[:-1], samples[1:]):
            dt = float(s1.t - s0.t)
            if dt <= 0:
                continue
            omega = np.asarray(s0.omega, dtype=float).reshape(3)
            accel = np.asarray(s0.accel, dtype=float).reshape(3)
            dR = _so3_exp(omega * dt)
            a0 = self._R @ accel
            self._p += self._v * dt + 0.5 * a0 * dt * dt
            self._v += a0 * dt
            self._R = self._R @ dR
            self._dt += dt
        rot = MinimalRot3(self._R)
        _, _, dyaw = rot.rpy()
        return PreintegrationResult(delta_t=float(self._dt), delta_R=rot, delta_v=self._v.copy(), delta_p=self._p.copy(), covariance=np.zeros((9, 9)), delta_yaw=float(dyaw))

def rot3_yaw(rot: object) -> float:
    if _HAS_GTSAM and hasattr(rot, 'rpy'):
        r, p, y = rot.rpy()
        return float(y)
    if isinstance(rot, MinimalRot3):
        return float(rot.rpy()[2])
    R = np.asarray(rot.matrix(), dtype=float)
    return float(np.arctan2(R[1, 0], R[0, 0]))
