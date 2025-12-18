from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class TapeObservation:
    centerline_px: np.ndarray
    angle_rad: float
    bottom_x: float
    shape_hw: Tuple[int, int]
    mask: np.ndarray


def _fit_direction(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return float("nan")
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


class LineDetector:
    def __init__(self, resize_width: int = 640):
        self.min_pixels = 10
        self.prev_centerline: Optional[np.ndarray] = None

    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_label = 1 + int(areas.argmax())
            return (labels == max_label).astype("uint8") * 255
        else:
            return (mask > 0).astype("uint8") * 255

    def _calculate_centerline_dt(self, mask: np.ndarray, start_y: int, end_y: int) -> Tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape

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

        pts_array = np.array(points_list, dtype=np.float32) if points_list else np.zeros((0, 2), dtype=np.float32)

        return centerline_img, pts_array

    def process(self, frame_bgr: np.ndarray, cfg: dict) -> TapeObservation:
        scale_percent = 0.8
        width = int(frame_bgr.shape[1] * scale_percent)
        height = int(frame_bgr.shape[0] * scale_percent)
        frame_resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)

        lb = int(cfg.get("LB", 0))
        lg = int(cfg.get("LG", 0))
        lr = int(cfg.get("LR", 200))
        hb = int(cfg.get("HB", 255))
        hg = int(cfg.get("HG", 255))
        hr = int(cfg.get("HR", 255))

        mask = cv2.inRange(frame_resized, (lb, lg, lr), (hb, hg, hr))
        mask = cv2.medianBlur(mask, 5)

        clean_mask = self._keep_largest_component(mask)

        h, w = clean_mask.shape
        top_cut = h // 2
        bot_cut = h // 10

        clean_mask[:top_cut, :] = 0
        if bot_cut > 0:
            clean_mask[h - bot_cut:, :] = 0

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, close_kernel)

        centerline_img, points_xy = self._calculate_centerline_dt(clean_mask, top_cut, h - bot_cut)

        if points_xy.shape[0] < self.min_pixels and self.prev_centerline is not None:
            centerline_img = self.prev_centerline
            points_xy = np.zeros((0, 2), dtype=np.float32)
        elif points_xy.shape[0] >= self.min_pixels:
            self.prev_centerline = centerline_img.copy()

        if points_xy.shape[0] >= 2:
            angle = _fit_direction(points_xy)
        else:
            angle = float("nan")

        if points_xy.shape[0] > 0:
            bottom_x = float(points_xy[-1][0])
        else:
            bottom_x = float("nan")

        return TapeObservation(
            centerline_px=points_xy,
            angle_rad=angle,
            bottom_x=bottom_x,
            shape_hw=(h, w),
            mask=centerline_img
        )