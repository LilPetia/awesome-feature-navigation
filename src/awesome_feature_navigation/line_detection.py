from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
SUPPORTED_COLORS: Tuple[str, ...] = ('red', 'blue', 'green', 'yellow', 'white')
COLOR_PRESETS = {'red': [([0, 110, 70], [10, 255, 255]), ([170, 110, 70], [180, 255, 255])], 'blue': [([95, 100, 60], [130, 255, 255])], 'green': [([40, 70, 60], [90, 255, 255])], 'yellow': [([18, 90, 90], [38, 255, 255])], 'white': [([0, 0, 180], [180, 70, 255])]}

@dataclass(frozen=True)
class TapeObservation:
    centerline_px: np.ndarray
    angle_rad: float
    bottom_x: float
    shape_hw: Tuple[int, int]
    mask: np.ndarray
    centerline_mask: np.ndarray

def _fit_direction(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return float('nan')
    pts = points_xy.astype(np.float64)
    mean = pts.mean(axis=0, keepdims=True)
    z = pts - mean
    cov = z.T @ z / max(1, pts.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    angle = float(np.arctan2(v[0], -v[1]))
    if v[1] > 0:
        angle = float(np.arctan2(-v[0], v[1]))
    return angle


def _vector_to_angle(vec_xy: np.ndarray) -> float:
    vec = np.asarray(vec_xy, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return float('nan')
    return float(np.arctan2(vec[0], -vec[1]))


def _fit_local_direction(points_xy: np.ndarray, fraction: float, min_points: int) -> float:
    if points_xy.shape[0] < 2:
        return float('nan')
    pts = points_xy.astype(np.float64)
    clamped_fraction = float(np.clip(fraction, 0.05, 1.0))
    count = max(int(min_points), int(round(pts.shape[0] * clamped_fraction)))
    count = min(count, pts.shape[0])
    if count < 2:
        return float('nan')
    segment = pts[-count:]
    deltas = np.diff(segment, axis=0)
    valid = np.sqrt(np.sum(deltas * deltas, axis=1)) > 1e-9
    if not np.any(valid):
        return _vector_to_angle(segment[0] - segment[-1])
    # The robot follows the tangent near the bottom of the frame, so use the
    # local segment direction instead of the full visible curve.
    forward_vec = -deltas[valid].mean(axis=0)
    return _vector_to_angle(forward_vec)


def _blend_angles(primary: float, secondary: float, secondary_weight: float) -> float:
    if not np.isfinite(primary):
        return float(secondary)
    if not np.isfinite(secondary):
        return float(primary)
    weight = float(np.clip(secondary_weight, 0.0, 1.0))
    primary_vec = np.array([np.cos(primary), np.sin(primary)], dtype=np.float64)
    secondary_vec = np.array([np.cos(secondary), np.sin(secondary)], dtype=np.float64)
    mixed = (1.0 - weight) * primary_vec + weight * secondary_vec
    norm = float(np.linalg.norm(mixed))
    if norm < 1e-9:
        return float(primary)
    mixed /= norm
    return float(np.arctan2(mixed[1], mixed[0]))


def _measure_bottom_x(points_xy: np.ndarray, fraction: float, min_points: int, statistic: str) -> float:
    if points_xy.shape[0] == 0:
        return float('nan')
    pts = points_xy.astype(np.float64)
    clamped_fraction = float(np.clip(fraction, 0.0, 1.0))
    if clamped_fraction <= 0.0:
        return float(pts[-1, 0])
    count = max(int(min_points), int(round(pts.shape[0] * clamped_fraction)))
    count = min(count, pts.shape[0])
    window_x = pts[-count:, 0]
    if statistic == 'mean':
        return float(np.mean(window_x))
    return float(np.median(window_x))

def _clip_hsv_triplet(values: Sequence[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=int).reshape(3)
    arr[0] = int(np.clip(arr[0], 0, 180))
    arr[1:] = np.clip(arr[1:], 0, 255)
    return arr.astype(np.uint8)

def _normalize_hsv_ranges(ranges: Iterable[Sequence[int]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    normalized: List[Tuple[np.ndarray, np.ndarray]] = []
    for item in ranges:
        values = np.asarray(item, dtype=int).flatten()
        if values.size != 6:
            continue
        raw_low = _clip_hsv_triplet(values[:3])
        raw_high = _clip_hsv_triplet(values[3:])
        low = np.minimum(raw_low, raw_high)
        high = np.maximum(raw_low, raw_high)
        normalized.append((low, high))
    return normalized

def resolve_target_color(cfg: dict) -> str:
    color = str(cfg.get('target_color', 'blue')).strip().lower()
    if color not in SUPPORTED_COLORS:
        return 'blue'
    return color

def resolve_hsv_ranges(cfg: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    explicit = cfg.get('hsv_ranges')
    if explicit:
        normalized = _normalize_hsv_ranges(explicit)
        if normalized:
            return normalized
    color = resolve_target_color(cfg)
    if color == 'red':
        legacy_ranges = []
        for idx in (1, 2):
            low = cfg.get(f'hsv_red{idx}_low')
            high = cfg.get(f'hsv_red{idx}_high')
            if low is not None and high is not None:
                legacy_ranges.append(list(low) + list(high))
        if legacy_ranges:
            normalized = _normalize_hsv_ranges(legacy_ranges)
            if normalized:
                return normalized
    preset = COLOR_PRESETS[color]
    return _normalize_hsv_ranges([list(low) + list(high) for low, high in preset])

class LineDetector:

    def __init__(self, resize_width: int=640):
        self.resize_width = resize_width
        self.min_pixels = 10
        self.prev_centerline: Optional[np.ndarray] = None

    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_label = 1 + int(areas.argmax())
            return (labels == max_label).astype('uint8') * 255
        return (mask > 0).astype('uint8') * 255

    def _calculate_centerline_dt(self, mask: np.ndarray, start_y: int, end_y: int) -> Tuple[np.ndarray, np.ndarray]:
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        roi_map = np.zeros_like(dist_map)
        roi_map[start_y:end_y, :] = dist_map[start_y:end_y, :]
        max_indices = np.argmax(roi_map, axis=1)
        max_values = np.max(roi_map, axis=1)
        centerline_img = np.zeros_like(mask, dtype=np.uint8)
        points_list = []
        for y in range(start_y, end_y):
            if max_values[y] > 0:
                cx = int(max_indices[y])
                centerline_img[y, cx] = 255
                points_list.append([cx, y])
        if points_list:
            pts_array = np.array(points_list, dtype=np.float32)
        else:
            pts_array = np.zeros((0, 2), dtype=np.float32)
        return (centerline_img, pts_array)

    def _smooth_centerline_points(self, points_xy: np.ndarray, smooth_window: int, width: int) -> np.ndarray:
        if points_xy.shape[0] < 3:
            return points_xy
        window = max(1, int(smooth_window))
        if window % 2 == 0:
            window += 1
        if window <= 1:
            return points_xy
        xs = points_xy[:, 0].astype(np.float32)
        ys = points_xy[:, 1].astype(np.float32)
        radius = window // 2
        padded = np.pad(xs, (radius, radius), mode='edge')
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smoothed_xs = np.convolve(padded, kernel, mode='valid')
        smoothed_xs = np.clip(smoothed_xs, 0, max(0, width - 1))
        return np.column_stack((smoothed_xs, ys)).astype(np.float32)

    def _render_centerline(self, mask_shape: Tuple[int, int], points_xy: np.ndarray) -> np.ndarray:
        centerline_img = np.zeros(mask_shape, dtype=np.uint8)
        if points_xy.shape[0] == 0:
            return centerline_img
        pts = np.round(points_xy).astype(np.int32)
        if pts.shape[0] == 1:
            x, y = pts[0]
            centerline_img[y, x] = 255
            return centerline_img
        cv2.polylines(centerline_img, [pts.reshape(-1, 1, 2)], isClosed=False, color=255, thickness=1, lineType=cv2.LINE_AA)
        return centerline_img

    def _build_mask(self, hsv_frame: np.ndarray, ranges: Sequence[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        for low, high in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, low, high))
        return mask

    def _auto_tune_ranges(self, hsv_frame: np.ndarray, ranges: Sequence[Tuple[np.ndarray, np.ndarray]], color: str, start_y: int, end_y: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        hsv_roi = hsv_frame[start_y:end_y, :, :]
        if hsv_roi.size == 0:
            return list(ranges)
        tuned: List[Tuple[np.ndarray, np.ndarray]] = []
        for low, high in ranges:
            h = hsv_roi[:, :, 0]
            s = hsv_roi[:, :, 1]
            v = hsv_roi[:, :, 2]
            relaxed_low = low.astype(int).copy()
            relaxed_high = high.astype(int).copy()
            if color == 'white':
                candidate = (s <= max(int(high[1]) + 50, 120)) & (v >= max(80, int(low[2]) - 60))
            else:
                relaxed_low[1] = max(20, relaxed_low[1] - 70)
                relaxed_low[2] = max(20, relaxed_low[2] - 60)
                relaxed_low[0] = max(0, relaxed_low[0] - 5)
                relaxed_high[0] = min(180, relaxed_high[0] + 5)
                candidate = (h >= relaxed_low[0]) & (h <= relaxed_high[0]) & (s >= relaxed_low[1]) & (v >= relaxed_low[2])
            if int(candidate.sum()) < 80:
                tuned.append((low.copy(), high.copy()))
                continue
            cand_h = h[candidate]
            cand_s = s[candidate]
            cand_v = v[candidate]
            tuned_low = low.astype(int).copy()
            tuned_high = high.astype(int).copy()
            if color == 'white':
                tuned_low[2] = int(np.clip(np.percentile(cand_v, 25) - 15, 80, 255))
                tuned_high[1] = int(np.clip(np.percentile(cand_s, 85) + 15, 10, 255))
            else:
                tuned_low[0] = int(np.clip(np.percentile(cand_h, 5) - 4, low[0], high[0]))
                tuned_high[0] = int(np.clip(np.percentile(cand_h, 95) + 4, tuned_low[0], high[0]))
                tuned_low[1] = int(np.clip(np.percentile(cand_s, 20) - 20, 20, 255))
                tuned_low[2] = int(np.clip(np.percentile(cand_v, 20) - 20, 20, 255))
            tuned.append((_clip_hsv_triplet(tuned_low), _clip_hsv_triplet(tuned_high)))
        return tuned

    def process(self, frame_bgr: np.ndarray, cfg: dict) -> TapeObservation:
        src_h, src_w = frame_bgr.shape[:2]
        if self.resize_width > 0 and src_w != self.resize_width:
            width = int(self.resize_width)
            height = max(1, int(round(src_h * (float(width) / float(src_w)))))
            frame_resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
        else:
            height, width = src_h, src_w
            frame_resized = frame_bgr
        hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        color = resolve_target_color(cfg)
        ranges = resolve_hsv_ranges(cfg)
        h, w = hsv_frame.shape[:2]
        top_cut = h // 2
        bot_cut = h // 10
        if bool(cfg.get('auto_color_tune', False)):
            ranges = self._auto_tune_ranges(hsv_frame, ranges, color, top_cut, h - bot_cut)
        mask = self._build_mask(hsv_frame, ranges)
        mask = cv2.medianBlur(mask, 5)
        clean_mask = self._keep_largest_component(mask)
        clean_mask[:top_cut, :] = 0
        if bot_cut > 0:
            clean_mask[h - bot_cut:, :] = 0
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, close_kernel)
        _, points_xy = self._calculate_centerline_dt(clean_mask, top_cut, h - bot_cut)
        points_xy = self._smooth_centerline_points(points_xy, smooth_window=int(cfg.get('centerline_smooth_window', 11)), width=w)
        centerline_img = self._render_centerline(clean_mask.shape, points_xy)
        if points_xy.shape[0] < self.min_pixels and self.prev_centerline is not None:
            centerline_img = self.prev_centerline
            points_xy = np.zeros((0, 2), dtype=np.float32)
        elif points_xy.shape[0] >= self.min_pixels:
            self.prev_centerline = centerline_img.copy()
        if points_xy.shape[0] >= 2:
            global_angle = _fit_direction(points_xy)
            local_angle = _fit_local_direction(
                points_xy,
                fraction=float(cfg.get('line_angle_window_fraction', 0.35) or 0.35),
                min_points=int(cfg.get('line_angle_window_min_points', 8) or 8),
            )
            angle_mode = str(cfg.get('line_angle_mode', 'pca') or 'pca').strip().lower()
            if angle_mode == 'bottom_segment':
                angle = local_angle
            elif angle_mode == 'blend':
                angle = _blend_angles(
                    primary=global_angle,
                    secondary=local_angle,
                    secondary_weight=float(cfg.get('line_angle_local_weight', 0.75) or 0.75),
                )
            else:
                angle = global_angle
        else:
            angle = float('nan')
        if points_xy.shape[0] > 0:
            bottom_x = _measure_bottom_x(
                points_xy,
                fraction=float(cfg.get('line_bottom_window_fraction', 0.0) or 0.0),
                min_points=int(cfg.get('line_bottom_window_min_points', 6) or 6),
                statistic=str(cfg.get('line_bottom_window_statistic', 'median') or 'median').strip().lower(),
            )
        else:
            bottom_x = float('nan')
        return TapeObservation(centerline_px=points_xy, angle_rad=angle, bottom_x=bottom_x, shape_hw=(h, w), mask=clean_mask, centerline_mask=centerline_img)
