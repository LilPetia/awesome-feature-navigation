from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .line_detection import COLOR_PRESETS, LineDetector, SUPPORTED_COLORS


@dataclass(frozen=True)
class AutoLineConfig:
    target_color: str
    hsv_ranges: List[List[int]]
    valid_ratio: float
    mean_score: float
    sample_count: int


def _preset_ranges(color: str) -> List[List[int]]:
    return [list(low) + list(high) for low, high in COLOR_PRESETS[color]]


def _resize_keep_aspect(frame_bgr: np.ndarray, resize_width: int) -> np.ndarray:
    src_h, src_w = frame_bgr.shape[:2]
    if resize_width <= 0 or src_w == resize_width:
        return frame_bgr
    width = int(resize_width)
    height = max(1, int(round(src_h * (float(width) / float(src_w)))))
    return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)


def _sample_frames(video_path: str, cfg: Dict) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    max_samples = max(4, int(cfg.get('auto_config_samples', 36) or 36))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: List[np.ndarray] = []
    if frame_count > 0:
        indices = np.linspace(0, max(0, frame_count - 1), max_samples, dtype=int)
        for idx in np.unique(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    else:
        stride = max(1, int(cfg.get('auto_config_sample_stride', 30) or 30))
        frame_i = 0
        while len(frames) < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_i % stride == 0:
                frames.append(frame)
            frame_i += 1
    cap.release()
    return frames


def _observation_score(mask: np.ndarray, points_xy: np.ndarray, angle_rad: float, shape_hw: tuple[int, int]) -> float:
    if points_xy.shape[0] < 8:
        return 0.0
    h, w = shape_hw
    top_cut = h // 2
    bot_cut = h // 10
    roi_end = max(top_cut + 1, h - bot_cut)
    roi_h = max(1, roi_end - top_cut)
    pts = points_xy.astype(float)
    y_span = float(np.clip((pts[:, 1].max() - pts[:, 1].min()) / roi_h, 0.0, 1.0))
    point_ratio = float(np.clip(points_xy.shape[0] / roi_h, 0.0, 1.0))
    bottom_y = top_cut + 0.72 * roi_h
    bottom_hit = 1.0 if bool(np.any(pts[:, 1] >= bottom_y)) else 0.0
    area_ratio = float(np.count_nonzero(mask[top_cut:roi_end, :])) / float(max(1, roi_h * w))
    if area_ratio < 0.0005 or area_ratio > 0.35:
        area_score = 0.25
    elif area_ratio > 0.22:
        area_score = 0.6
    else:
        area_score = 1.0
    angle_score = 1.0 if np.isfinite(angle_rad) else 0.0
    return area_score * (2.0 * point_ratio + y_span + 0.6 * bottom_hit + 0.3 * angle_score)


def _score_color(frames: Sequence[np.ndarray], color: str, cfg: Dict) -> tuple[float, float]:
    resize_width = int(cfg.get('resize_width', cfg.get('auto_config_resize_width', 640)) or 640)
    detector = LineDetector(resize_width=resize_width)
    detect_cfg = dict(cfg)
    detect_cfg['target_color'] = color
    detect_cfg['hsv_ranges'] = _preset_ranges(color)
    detect_cfg['auto_color_tune'] = True
    scores: List[float] = []
    for frame in frames:
        obs = detector.process(frame, detect_cfg)
        scores.append(_observation_score(obs.mask, obs.centerline_px, obs.angle_rad, obs.shape_hw))
    if not scores:
        return (0.0, 0.0)
    score_arr = np.asarray(scores, dtype=float)
    valid_ratio = float(np.mean(score_arr >= float(cfg.get('auto_config_valid_score', 1.0) or 1.0)))
    mean_score = float(np.mean(score_arr))
    return (mean_score, valid_ratio)


def _clip_hsv(values: Sequence[float]) -> List[int]:
    h_low, s_low, v_low, h_high, s_high, v_high = values
    return [
        int(np.clip(round(h_low), 0, 180)),
        int(np.clip(round(s_low), 0, 255)),
        int(np.clip(round(v_low), 0, 255)),
        int(np.clip(round(h_high), 0, 180)),
        int(np.clip(round(s_high), 0, 255)),
        int(np.clip(round(v_high), 0, 255)),
    ]


def _build_ranges_from_pixels(color: str, hsv_pixels: np.ndarray) -> List[List[int]]:
    if hsv_pixels.shape[0] < 50:
        return _preset_ranges(color)
    h = hsv_pixels[:, 0].astype(float)
    s = hsv_pixels[:, 1].astype(float)
    v = hsv_pixels[:, 2].astype(float)
    if color == 'white':
        v_low = float(np.percentile(v, 12)) - 25.0
        s_high = float(np.percentile(s, 90)) + 25.0
        return [_clip_hsv([0, 0, v_low, 180, s_high, 255])]
    s_low = max(20.0, float(np.percentile(s, 12)) - 35.0)
    v_low = max(20.0, float(np.percentile(v, 12)) - 35.0)
    if color == 'red':
        ranges: List[List[int]] = []
        for group_mask in (h <= 30.0, h >= 145.0):
            if int(np.sum(group_mask)) < 25:
                continue
            hg = h[group_mask]
            h_low = float(np.percentile(hg, 4)) - 5.0
            h_high = float(np.percentile(hg, 96)) + 5.0
            ranges.append(_clip_hsv([h_low, s_low, v_low, h_high, 255, 255]))
        return ranges or _preset_ranges(color)
    h_low = float(np.percentile(h, 4)) - 5.0
    h_high = float(np.percentile(h, 96)) + 5.0
    if h_high - h_low < 6.0:
        mid = 0.5 * (h_low + h_high)
        h_low = mid - 3.0
        h_high = mid + 3.0
    return [_clip_hsv([h_low, s_low, v_low, h_high, 255, 255])]


def _collect_line_hsv_pixels(frames: Sequence[np.ndarray], color: str, cfg: Dict) -> np.ndarray:
    resize_width = int(cfg.get('resize_width', cfg.get('auto_config_resize_width', 640)) or 640)
    detector = LineDetector(resize_width=resize_width)
    detect_cfg = dict(cfg)
    detect_cfg['target_color'] = color
    detect_cfg['hsv_ranges'] = _preset_ranges(color)
    detect_cfg['auto_color_tune'] = True
    pixels: List[np.ndarray] = []
    for frame in frames:
        resized = _resize_keep_aspect(frame, resize_width)
        hsv_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        obs = detector.process(frame, detect_cfg)
        score = _observation_score(obs.mask, obs.centerline_px, obs.angle_rad, obs.shape_hw)
        if score < float(cfg.get('auto_config_valid_score', 1.0) or 1.0):
            continue
        mask = obs.mask > 0
        if hsv_frame.shape[:2] != mask.shape:
            continue
        frame_pixels = hsv_frame[mask]
        if frame_pixels.shape[0] > 3000:
            stride = max(1, frame_pixels.shape[0] // 3000)
            frame_pixels = frame_pixels[::stride]
        if frame_pixels.shape[0] > 0:
            pixels.append(frame_pixels)
    if not pixels:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(pixels).astype(np.uint8)


def infer_line_config_from_video(video_path: str, cfg: Dict) -> Optional[AutoLineConfig]:
    frames = _sample_frames(video_path, cfg)
    if not frames:
        return None
    candidates = cfg.get('auto_config_color_candidates', SUPPORTED_COLORS)
    if isinstance(candidates, (str, bytes)):
        candidates = [candidates]
    colors = [str(color).strip().lower() for color in candidates if str(color).strip().lower() in SUPPORTED_COLORS]
    if not colors:
        colors = list(SUPPORTED_COLORS)
    best_color: Optional[str] = None
    best_mean = 0.0
    best_valid = 0.0
    best_rank = -1.0
    for color in colors:
        mean_score, valid_ratio = _score_color(frames, color, cfg)
        rank = mean_score + 1.5 * valid_ratio
        if rank > best_rank:
            best_rank = rank
            best_color = color
            best_mean = mean_score
            best_valid = valid_ratio
    min_valid = float(cfg.get('auto_config_min_valid_ratio', 0.18) or 0.18)
    min_score = float(cfg.get('auto_config_min_mean_score', 0.35) or 0.35)
    if best_color is None or best_valid < min_valid or best_mean < min_score:
        return None
    hsv_pixels = _collect_line_hsv_pixels(frames, best_color, cfg)
    ranges = _build_ranges_from_pixels(best_color, hsv_pixels)
    return AutoLineConfig(
        target_color=best_color,
        hsv_ranges=ranges,
        valid_ratio=best_valid,
        mean_score=best_mean,
        sample_count=len(frames),
    )


def apply_auto_video_config(video_path: str, cfg: Dict) -> tuple[Dict, Optional[AutoLineConfig]]:
    if not bool(cfg.get('auto_video_config', True)):
        return (dict(cfg), None)
    out = dict(cfg)
    if not bool(out.get('auto_detect_line', True)):
        return (out, None)
    line_cfg = infer_line_config_from_video(video_path, out)
    if line_cfg is None:
        return (out, None)
    out['target_color'] = line_cfg.target_color
    out['hsv_ranges'] = line_cfg.hsv_ranges
    out['auto_color_tune'] = bool(out.get('auto_color_tune_after_detect', False))
    out['_auto_line_color'] = line_cfg.target_color
    out['_auto_line_valid_ratio'] = line_cfg.valid_ratio
    out['_auto_line_mean_score'] = line_cfg.mean_score
    return (out, line_cfg)
