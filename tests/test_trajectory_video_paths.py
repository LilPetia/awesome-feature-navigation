from __future__ import annotations

import cv2
import numpy as np
import pytest

import awesome_feature_navigation.trajectory as tr
from awesome_feature_navigation.line_detection import TapeObservation
from awesome_feature_navigation.trajectory import (
    IMUSample,
    TapeFrameObservation,
    TrajectoryEstimateResult,
    TrajectoryPoint,
)


class _FakeVideoCapture:
    def __init__(self, frames: list[np.ndarray], fps: float=10.0) -> None:
        self.frames = frames
        self.fps = fps
        self.index = 0
        self.opened = True

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_POS_MSEC:
            return float(self.index * 100.0)
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return (False, None)
        frame = self.frames[self.index]
        self.index += 1
        return (True, frame.copy())

    def release(self) -> None:
        self.opened = False


class _FakeWriter:
    writes = 0
    released = False

    def __init__(self, path: str, fourcc: int, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size

    def write(self, frame: np.ndarray) -> None:
        _FakeWriter.writes += 1

    def release(self) -> None:
        _FakeWriter.released = True


class _FakeLineDetector:
    def __init__(self, resize_width: int) -> None:
        self.resize_width = resize_width

    def process(self, frame: np.ndarray, cfg: dict[str, object]) -> TapeObservation:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[frame.shape[0] // 2:, frame.shape[1] // 2 - 1:frame.shape[1] // 2 + 2] = 255
        points = np.column_stack(
            (
                np.full(20, frame.shape[1] / 2.0, dtype=float),
                np.linspace(frame.shape[0] / 2.0, frame.shape[0] - 2.0, 20),
            )
        )
        return TapeObservation(
            centerline_px=points.astype(np.float32),
            angle_rad=0.0,
            bottom_x=frame.shape[1] / 2.0,
            shape_hw=(int(mask.shape[0]), int(mask.shape[1])),
            mask=mask,
            centerline_mask=mask.copy(),
        )


def _frames(count: int=4) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for idx in range(count):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + idx, 20 + idx), 5, (255, 255, 255), -1)
        frames.append(frame)
    return frames


def test_write_tape_line_debug_video_uses_capture_and_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _frames(2)
    monkeypatch.setattr(tr.cv2, 'VideoCapture', lambda path: _FakeVideoCapture(frames))
    monkeypatch.setattr(tr.cv2, 'VideoWriter', _FakeWriter)
    _FakeWriter.writes = 0
    _FakeWriter.released = False
    records = [
        TapeFrameObservation(0.0, 0.0, _FakeLineDetector(80).process(frames[0], {}), 1.0, 0.0, np.zeros(3)),
        TapeFrameObservation(1.0, 1.0, _FakeLineDetector(80).process(frames[1], {}), 1.0, 0.0, np.zeros(3)),
    ]
    traj = [TrajectoryPoint(0.0, 0.0, 0.0, 0.0), TrajectoryPoint(1.0, 1.0, 0.0, 0.0)]

    tr._write_tape_line_debug_video('video.mp4', 'debug.mp4', 10.0, records, traj)

    assert _FakeWriter.writes == 2
    assert _FakeWriter.released is True


def test_tape_line_estimator_runs_on_synthetic_video(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _frames(4)
    monkeypatch.setattr(tr.cv2, 'VideoCapture', lambda path: _FakeVideoCapture(frames))
    monkeypatch.setattr(tr, 'LineDetector', _FakeLineDetector)
    monkeypatch.setattr(tr, '_write_tape_line_debug_video', lambda *args: None)

    result = tr._estimate_tape_line_trajectory_with_details(
        video_path='video.mp4',
        imu_samples=None,
        cfg={
            'resize_width': 80,
            'auto_loop_period': False,
            'loop_average': False,
            'offline_tape_smoothing': False,
            'offline_adaptive_confidence': False,
            'offline_line_min_confidence': 0.1,
            'vision_lateral_gain': 0.0,
            'vision_yaw_gain': 0.0,
            'forward_speed_mps': 1.0,
            'auto_loop_descriptor_stride': 1,
        },
        save_debug_video='debug.mp4',
        frame_timestamps=[0.0, 1.0, 2.0, 3.0],
    )

    assert result.mode == 'tape_line'
    assert len(result.raw_traj) == 4
    assert result.tape_diagnostics is not None
    assert result.line_valid_ratio is not None


def test_generic_vio_estimator_runs_on_synthetic_video(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _frames(4)
    monkeypatch.setattr(tr.cv2, 'VideoCapture', lambda path: _FakeVideoCapture(frames))
    monkeypatch.setattr(tr.cv2, 'VideoWriter', _FakeWriter)
    monkeypatch.setattr(
        tr,
        '_detect_track_features',
        lambda gray, cfg: np.array([[[10.0, 10.0]], [[30.0, 10.0]], [[10.0, 30.0]], [[30.0, 30.0]]], dtype=np.float32),
    )
    monkeypatch.setattr(
        tr,
        '_track_features_forward_backward',
        lambda prev_gray, gray, prev_pts, cfg: (
            prev_pts.reshape(-1, 2),
            prev_pts.reshape(-1, 2) + np.array([1.0, 0.0], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        tr,
        '_estimate_normalized_similarity_transform',
        lambda prev_pts, next_pts, shape_hw, cfg: (
            np.array([[1.0, 0.0, 0.01], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
            prev_pts,
            next_pts,
            1.0,
        ),
    )
    imu = [
        IMUSample(t=0.0, accel=np.zeros(3), omega=np.zeros(3)),
        IMUSample(t=3.0, accel=np.zeros(3), omega=np.zeros(3)),
    ]

    result = tr._estimate_generic_vio_trajectory_with_details(
        video_path='video.mp4',
        imu_samples=imu,
        cfg={
            'max_frames': 0,
            'vio_resize_width': 80,
            'vio_redetect_threshold': 2,
            'loop_average': False,
            'auto_loop_descriptor_stride': 1,
        },
        save_debug_video='debug.mp4',
        frame_timestamps=[0.0, 1.0, 2.0, 3.0],
    )

    assert result.mode == 'generic_vio'
    assert result.relative_scale is True
    assert len(result.raw_traj) == 4


def test_estimate_trajectory_dispatches_between_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    generic = TrajectoryEstimateResult(
        raw_traj=[TrajectoryPoint(float(idx), float(idx), 0.0, 0.0) for idx in range(10)],
        smoothed_traj=[],
        final_traj=[TrajectoryPoint(0.0, 0.0, 0.0, 0.0)],
        loop_debug=None,
        mode='generic_vio',
        relative_scale=True,
    )
    tape = TrajectoryEstimateResult(
        raw_traj=[TrajectoryPoint(float(idx), float(idx), 0.0, 0.0) for idx in range(10)],
        smoothed_traj=[],
        final_traj=[TrajectoryPoint(0.0, 1.0, 0.0, 0.0)],
        loop_debug=None,
        mode='tape_line',
        line_valid_ratio=0.9,
    )

    monkeypatch.setattr(tr, '_estimate_generic_vio_trajectory_with_details', lambda **kwargs: generic)
    monkeypatch.setattr(tr, '_estimate_tape_line_trajectory_with_details', lambda **kwargs: tape)

    imu = [IMUSample(t=0.0, accel=np.zeros(3), omega=np.zeros(3))]
    auto_result = tr.estimate_trajectory_with_details('video.mp4', imu, {'trajectory_mode': 'auto'})
    tape_result = tr.estimate_trajectory_with_details('video.mp4', None, {'trajectory_mode': 'tape_line'})
    final = tr.estimate_trajectory('video.mp4', None, {'trajectory_mode': 'tape_line'})

    assert auto_result.mode == 'generic_vio'
    assert tape_result.mode == 'tape_line'
    assert final[0].x == pytest.approx(1.0)
