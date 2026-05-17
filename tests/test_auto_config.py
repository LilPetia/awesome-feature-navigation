from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

import awesome_feature_navigation.auto_config as auto_config
from awesome_feature_navigation.auto_config import (
    AutoLineConfig,
    _build_ranges_from_pixels,
    _clip_hsv,
    _collect_line_hsv_pixels,
    _observation_score,
    _preset_ranges,
    _resize_keep_aspect,
    _sample_frames,
    _score_color,
    apply_auto_video_config,
    infer_line_config_from_video,
)
from awesome_feature_navigation.line_detection import TapeObservation


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], frame_count: int) -> None:
        self.frames = frames
        self.frame_count = frame_count
        self.index = 0
        self.opened = True
        self.set_indices: list[int] = []

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.frame_count)
        return 0.0

    def set(self, prop: int, value: float) -> None:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.index = int(value)
            self.set_indices.append(self.index)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return (False, None)
        frame = self.frames[self.index]
        self.index += 1
        return (True, frame)

    def release(self) -> None:
        self.opened = False


def _frame(color_bgr: tuple[int, int, int]=(255, 0, 0)) -> np.ndarray:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[10:18, 14:17] = color_bgr
    return frame


def _observation(mask: np.ndarray, points: np.ndarray) -> TapeObservation:
    shape_hw = (int(mask.shape[0]), int(mask.shape[1]))
    return TapeObservation(
        centerline_px=points.astype(np.float32),
        angle_rad=0.0,
        bottom_x=15.0,
        shape_hw=shape_hw,
        mask=mask,
        centerline_mask=mask.copy(),
    )


def test_sample_frames_uses_video_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [_frame() for _ in range(5)]
    capture = _FakeCapture(frames, frame_count=5)
    monkeypatch.setattr(auto_config.cv2, 'VideoCapture', lambda path: capture)

    sampled = _sample_frames('video.mp4', {'auto_config_samples': 4})

    assert len(sampled) == 4
    assert capture.set_indices == [0, 1, 2, 4]
    assert not capture.opened


def test_sample_frames_uses_stride_when_frame_count_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [_frame() for _ in range(6)]
    capture = _FakeCapture(frames, frame_count=0)
    monkeypatch.setattr(auto_config.cv2, 'VideoCapture', lambda path: capture)

    sampled = _sample_frames('stream.mp4', {'auto_config_samples': 4, 'auto_config_sample_stride': 2})

    assert len(sampled) == 3
    assert capture.set_indices == []
    assert not capture.opened


def test_sample_frames_raises_when_video_cannot_open(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _FakeCapture([], frame_count=0)
    capture.opened = False
    monkeypatch.setattr(auto_config.cv2, 'VideoCapture', lambda path: capture)

    with pytest.raises(RuntimeError, match='Cannot open video'):
        _sample_frames('missing.mp4', {})


def test_auto_config_range_helpers_build_color_specific_ranges() -> None:
    blue_pixels = np.tile(np.array([[110, 180, 200]], dtype=np.uint8), (60, 1))
    white_pixels = np.tile(np.array([[10, 20, 240]], dtype=np.uint8), (60, 1))
    red_pixels = np.vstack(
        (
            np.tile(np.array([[3, 180, 200]], dtype=np.uint8), (40, 1)),
            np.tile(np.array([[176, 190, 210]], dtype=np.uint8), (40, 1)),
        )
    )

    assert _preset_ranges('blue') == [[95, 100, 60, 130, 255, 255]]
    assert _clip_hsv([-10, -2, 99, 200, 300, 301]) == [0, 0, 99, 180, 255, 255]
    assert _build_ranges_from_pixels('blue', blue_pixels)[0][0] <= 110
    assert _build_ranges_from_pixels('white', white_pixels)[0][2] >= 80
    assert len(_build_ranges_from_pixels('red', red_pixels)) == 2
    assert _build_ranges_from_pixels('blue', np.zeros((0, 3), dtype=np.uint8)) == _preset_ranges('blue')


def test_observation_score_rejects_short_centerline_and_scores_clean_line() -> None:
    mask = np.zeros((100, 80), dtype=np.uint8)
    mask[50:90, 39:42] = 255
    points = np.column_stack((np.full(40, 40.0), np.arange(50.0, 90.0)))

    shape_hw = (int(mask.shape[0]), int(mask.shape[1]))

    assert _observation_score(mask, points[:3], 0.0, shape_hw) == 0.0
    assert _observation_score(mask, points, 0.0, shape_hw) > 1.0


def test_observation_score_penalizes_implausible_mask_area() -> None:
    points = np.column_stack((np.full(40, 50.0), np.arange(50.0, 90.0)))
    shape_hw = (100, 100)
    normal_mask = np.zeros(shape_hw, dtype=np.uint8)
    normal_mask[50:90, 49:52] = 255
    wide_mask = np.zeros(shape_hw, dtype=np.uint8)
    wide_mask[50:90, 0:25] = 255
    full_mask = np.zeros(shape_hw, dtype=np.uint8)
    full_mask[50:90, :] = 255
    tiny_mask = np.zeros(shape_hw, dtype=np.uint8)
    tiny_mask[50, 50] = 255

    normal = _observation_score(normal_mask, points, 0.0, shape_hw)
    wide = _observation_score(wide_mask, points, 0.0, shape_hw)
    full = _observation_score(full_mask, points, 0.0, shape_hw)
    tiny = _observation_score(tiny_mask, points, float('nan'), shape_hw)

    assert 0.0 < full < wide < normal
    assert 0.0 < tiny < normal


def test_score_color_uses_detector_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    mask = np.zeros((100, 80), dtype=np.uint8)
    mask[50:90, 39:42] = 255
    points = np.column_stack((np.full(40, 40.0), np.arange(50.0, 90.0)))

    class FakeDetector:
        def __init__(self, resize_width: int) -> None:
            assert resize_width == 40

        def process(self, frame: np.ndarray, cfg: dict[str, object]) -> TapeObservation:
            assert cfg['target_color'] == 'blue'
            return _observation(mask, points)

    monkeypatch.setattr(auto_config, 'LineDetector', FakeDetector)

    mean_score, valid_ratio = _score_color([_frame(), _frame()], 'blue', {'resize_width': 40, 'auto_config_valid_score': 0.5})

    assert mean_score > 1.0
    assert valid_ratio == 1.0


def test_score_color_returns_zero_without_frames() -> None:
    assert _score_color([], 'blue', {}) == (0.0, 0.0)


def test_collect_line_hsv_pixels_samples_valid_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame((255, 0, 0))
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[10:18, 14:17] = 255
    points = np.column_stack((np.full(8, 15.0), np.arange(10.0, 18.0)))

    class FakeDetector:
        def __init__(self, resize_width: int) -> None:
            assert resize_width == 30

        def process(self, frame: np.ndarray, cfg: dict[str, object]) -> TapeObservation:
            return _observation(mask, points)

    monkeypatch.setattr(auto_config, 'LineDetector', FakeDetector)

    pixels = _collect_line_hsv_pixels([frame], 'blue', {'resize_width': 30, 'auto_config_valid_score': 0.1})

    assert pixels.shape[1] == 3
    assert pixels.shape[0] > 0


def test_build_ranges_handles_sparse_red_groups_and_narrow_hues() -> None:
    red_pixels = np.vstack(
        (
            np.tile(np.array([[5, 180, 200]], dtype=np.uint8), (60, 1)),
            np.tile(np.array([[176, 180, 200]], dtype=np.uint8), (10, 1)),
        )
    )
    narrow_pixels = np.tile(np.array([[100, 180, 200]], dtype=np.uint8), (60, 1))

    red_ranges = _build_ranges_from_pixels('red', red_pixels)
    narrow_range = _build_ranges_from_pixels('green', narrow_pixels)[0]

    assert len(red_ranges) == 1
    assert narrow_range[3] - narrow_range[0] >= 6


def test_build_ranges_expands_narrow_hue_when_percentiles_are_too_close(monkeypatch: pytest.MonkeyPatch) -> None:
    pixels = np.tile(np.array([[100, 180, 200]], dtype=np.uint8), (60, 1))
    calls = 0

    def fake_percentile(values: np.ndarray, percentile: float) -> float:
        nonlocal calls
        calls += 1
        if calls == 1:
            return 180.0
        if calls == 2:
            return 200.0
        if calls == 3:
            return 100.0
        return 90.0

    monkeypatch.setattr(auto_config.np, 'percentile', fake_percentile)

    hue_range = _build_ranges_from_pixels('blue', pixels)[0]

    assert hue_range[3] - hue_range[0] == 6


def test_collect_line_hsv_pixels_rejects_low_scores_and_shape_mismatches(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame((255, 0, 0))
    valid_mask = np.zeros((20, 30), dtype=np.uint8)
    valid_mask[10:18, 14:17] = 255
    points = np.column_stack((np.full(8, 15.0), np.arange(10.0, 18.0)))

    class LowScoreDetector:
        def __init__(self, resize_width: int) -> None:
            pass

        def process(self, frame: np.ndarray, cfg: dict[str, object]) -> TapeObservation:
            return _observation(valid_mask, points[:3])

    monkeypatch.setattr(auto_config, 'LineDetector', LowScoreDetector)
    assert _collect_line_hsv_pixels([frame], 'blue', {'resize_width': 30}).shape == (0, 3)

    class MismatchedMaskDetector:
        def __init__(self, resize_width: int) -> None:
            pass

        def process(self, frame: np.ndarray, cfg: dict[str, object]) -> TapeObservation:
            return _observation(valid_mask, points)

    monkeypatch.setattr(auto_config, 'LineDetector', MismatchedMaskDetector)
    pixels = _collect_line_hsv_pixels(
        [frame],
        'blue',
        {'resize_width': 15, 'auto_config_valid_score': 0.1},
    )

    assert pixels.shape == (0, 3)


def test_collect_line_hsv_pixels_downsamples_large_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[100:180, 100:200] = [255, 0, 0]
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[100:180, 100:200] = 255
    points = np.column_stack((np.full(80, 150.0), np.arange(100.0, 180.0)))

    class LargeMaskDetector:
        def __init__(self, resize_width: int) -> None:
            assert resize_width == 300

        def process(self, frame: np.ndarray, cfg: dict[str, object]) -> TapeObservation:
            return _observation(mask, points)

    monkeypatch.setattr(auto_config, 'LineDetector', LargeMaskDetector)

    pixels = _collect_line_hsv_pixels(
        [frame],
        'blue',
        {'resize_width': 300},
    )

    assert 0 < pixels.shape[0] < np.count_nonzero(mask)


def test_infer_line_config_selects_best_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auto_config, '_sample_frames', lambda video_path, cfg: [_frame(), _frame()])
    monkeypatch.setattr(
        auto_config,
        '_score_color',
        lambda frames, color, cfg: (2.0, 0.75) if color == 'green' else (0.2, 0.0),
    )
    monkeypatch.setattr(
        auto_config,
        '_collect_line_hsv_pixels',
        lambda frames, color, cfg: np.tile(np.array([[55, 180, 200]], dtype=np.uint8), (60, 1)),
    )

    line_cfg = infer_line_config_from_video('video.mp4', {'auto_config_color_candidates': ['blue', 'green']})

    assert line_cfg is not None
    assert line_cfg.target_color == 'green'
    assert line_cfg.sample_count == 2


def test_infer_line_config_accepts_single_candidate_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auto_config, '_sample_frames', lambda video_path, cfg: [_frame(), _frame()])
    monkeypatch.setattr(auto_config, '_score_color', lambda frames, color, cfg: (2.0, 1.0))
    monkeypatch.setattr(
        auto_config,
        '_collect_line_hsv_pixels',
        lambda frames, color, cfg: np.tile(np.array([[110, 180, 200]], dtype=np.uint8), (60, 1)),
    )

    line_cfg = infer_line_config_from_video('video.mp4', {'auto_config_color_candidates': 'blue'})

    assert line_cfg is not None
    assert line_cfg.target_color == 'blue'


def test_infer_line_config_returns_none_for_empty_or_weak_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auto_config, '_sample_frames', lambda video_path, cfg: [])
    assert infer_line_config_from_video('video.mp4', {}) is None

    monkeypatch.setattr(auto_config, '_sample_frames', lambda video_path, cfg: [_frame(), _frame()])
    monkeypatch.setattr(auto_config, '_score_color', lambda frames, color, cfg: (0.01, 0.0))

    line_cfg = infer_line_config_from_video(
        'video.mp4',
        {'auto_config_color_candidates': ['unknown']},
    )

    assert line_cfg is None


def test_apply_auto_video_config_respects_disable_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    assert apply_auto_video_config('video.mp4', {'auto_video_config': False}) == ({'auto_video_config': False}, None)
    assert apply_auto_video_config('video.mp4', {'auto_detect_line': False}) == ({'auto_detect_line': False}, None)

    def fake_infer(video_path: str, cfg: dict[str, Any]) -> AutoLineConfig:
        return AutoLineConfig(
            target_color='yellow',
            hsv_ranges=[[18, 90, 90, 38, 255, 255]],
            valid_ratio=0.8,
            mean_score=2.0,
            sample_count=3,
        )

    monkeypatch.setattr(auto_config, 'infer_line_config_from_video', fake_infer)

    cfg, line_cfg = apply_auto_video_config('video.mp4', {'auto_color_tune_after_detect': True})

    assert line_cfg is not None
    assert cfg['target_color'] == 'yellow'
    assert cfg['auto_color_tune'] is True
    assert cfg['_auto_line_valid_ratio'] == pytest.approx(0.8)


def test_apply_auto_video_config_keeps_input_when_detection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auto_config, 'infer_line_config_from_video', lambda video_path, cfg: None)

    cfg, line_cfg = apply_auto_video_config('video.mp4', {'target_color': 'blue'})

    assert cfg == {'target_color': 'blue'}
    assert line_cfg is None


def test_resize_keep_aspect_preserves_small_frames() -> None:
    frame = _frame()

    assert _resize_keep_aspect(frame, 0) is frame
    resized = _resize_keep_aspect(frame, 15)
    assert resized.shape[:2] == (10, 15)
