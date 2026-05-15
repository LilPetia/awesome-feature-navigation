import numpy as np
import pytest

from awesome_feature_navigation.line_detection import TapeObservation
from awesome_feature_navigation.trajectory import (
    TapeFrameObservation,
    _build_offline_tape_line_trajectory,
    _line_observation_confidence,
)


def _observation(angle_rad: float, bottom_x: float=50.0) -> TapeObservation:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50:90, 48:53] = 255
    centerline_mask = np.zeros_like(mask)
    centerline_mask[50:90, 50] = 255
    centerline = np.column_stack(
        (
            np.full(40, bottom_x, dtype=np.float32),
            np.arange(50, 90, dtype=np.float32),
        )
    )
    return TapeObservation(
        centerline_px=centerline,
        angle_rad=angle_rad,
        bottom_x=bottom_x,
        shape_hw=(100, 100),
        mask=mask,
        centerline_mask=centerline_mask,
    )


def _record(t: float, angle_rad: float, confidence: float) -> TapeFrameObservation:
    return TapeFrameObservation(
        t=t,
        dt=0.0 if t == 0.0 else 1.0,
        obs=_observation(angle_rad=angle_rad),
        confidence=confidence,
        delta_yaw=0.0,
        delta_p=np.zeros(3, dtype=float),
    )


def test_line_observation_confidence_is_positive_for_clean_centerline() -> None:
    confidence = _line_observation_confidence(_observation(angle_rad=0.0), {})

    assert confidence > 0.5


def test_offline_tape_line_smoothing_ignores_low_confidence_angle_outlier() -> None:
    records = [
        _record(0.0, 0.0, 1.0),
        _record(1.0, 0.0, 1.0),
        _record(2.0, 1.5, 0.0),
        _record(3.0, 0.0, 1.0),
        _record(4.0, 0.0, 1.0),
    ]
    cfg = {
        'forward_speed_mps': 1.0,
        'vision_yaw_gain': 1.0,
        'vision_yaw_max_correction': 1.0,
        'vision_lateral_gain': 0.0,
        'offline_tape_smoothing': True,
        'offline_line_smoothing_frames': 5,
        'offline_line_min_confidence': 0.1,
    }

    trajectory, valid_ratio = _build_offline_tape_line_trajectory(records, cfg)

    assert valid_ratio == pytest.approx(0.8)
    assert trajectory[-1].x == pytest.approx(4.0)
    assert trajectory[-1].y == pytest.approx(0.0, abs=1e-9)
    assert trajectory[-1].yaw == pytest.approx(0.0, abs=1e-9)
