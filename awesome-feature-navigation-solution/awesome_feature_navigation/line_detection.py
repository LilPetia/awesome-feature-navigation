from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class TapeObservation:
    """Vision measurement extracted from one frame."""

    # (N, 2) integer pixels in image coordinates (x, y)
    centerline_px: np.ndarray

    # tape direction angle in image coordinates (radians).
    # 0 means tape goes straight "up" in the image (negative y direction).
    angle_rad: float

    # x coordinate of tape near the bottom of the image (pixels), NaN if unavailable
    bottom_x: float

    # resized image shape (h, w)
    shape_hw: Tuple[int, int]

    # binary mask of tape (uint8 0/255), resized
    mask: np.ndarray


def _as_hsv_ranges(cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r1l = np.array(cfg.get("hsv_red1_low", [0, 80, 80]), dtype=np.uint8)
    r1h = np.array(cfg.get("hsv_red1_high", [10, 255, 255]), dtype=np.uint8)
    r2l = np.array(cfg.get("hsv_red2_low", [160, 80, 80]), dtype=np.uint8)
    r2h = np.array(cfg.get("hsv_red2_high", [180, 255, 255]), dtype=np.uint8)
    return r1l, r1h, r2l, r2h


def _zs_thinning(bin_img: np.ndarray, max_iter: int = 64) -> np.ndarray:
    """Zhang-Suen thinning. Input is binary 0/255 or 0/1. Returns 0/255."""
    img = (bin_img > 0).astype(np.uint8)
    h, w = img.shape
    if h < 3 or w < 3:
        return (img * 255).astype(np.uint8)

    def neighbors(y: int, x: int) -> list[int]:
        # P2..P9 around P1 (y,x), clockwise starting at north
        p2 = img[y - 1, x]
        p3 = img[y - 1, x + 1]
        p4 = img[y, x + 1]
        p5 = img[y + 1, x + 1]
        p6 = img[y + 1, x]
        p7 = img[y + 1, x - 1]
        p8 = img[y, x - 1]
        p9 = img[y - 1, x - 1]
        return [p2, p3, p4, p5, p6, p7, p8, p9]

    for _ in range(max_iter):
        changed = False
        to_del = []

        # step 1
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] == 0:
                    continue
                nb = neighbors(y, x)
                n = sum(nb)
                if n < 2 or n > 6:
                    continue
                transitions = sum((nb[i] == 0 and nb[(i + 1) % 8] == 1) for i in range(8))
                if transitions != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = nb
                if p2 * p4 * p6 != 0:
                    continue
                if p4 * p6 * p8 != 0:
                    continue
                to_del.append((y, x))

        if to_del:
            for y, x in to_del:
                img[y, x] = 0
            changed = True

        to_del = []

        # step 2
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] == 0:
                    continue
                nb = neighbors(y, x)
                n = sum(nb)
                if n < 2 or n > 6:
                    continue
                transitions = sum((nb[i] == 0 and nb[(i + 1) % 8] == 1) for i in range(8))
                if transitions != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = nb
                if p2 * p4 * p8 != 0:
                    continue
                if p2 * p6 * p8 != 0:
                    continue
                to_del.append((y, x))

        if to_del:
            for y, x in to_del:
                img[y, x] = 0
            changed = True

        if not changed:
            break

    return (img * 255).astype(np.uint8)


def _fit_direction(points_xy: np.ndarray) -> float:
    """Fit principal direction via PCA and return angle relative to image up (radians)."""
    if points_xy.shape[0] < 2:
        return float("nan")
    pts = points_xy.astype(np.float64)
    mean = pts.mean(axis=0, keepdims=True)
    z = pts - mean
    cov = z.T @ z / max(1, pts.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]  # principal axis
    # v is in (x, y). We want angle relative to "up" direction (0,-1).
    # Angle is positive when tape leans to the right in the image.
    # atan2 for vector relative to up: angle = atan2(vx, -vy)
    angle = float(np.arctan2(v[0], -v[1]))
    # Resolve sign ambiguity (v and -v represent same line). Make it point "up" (negative y) if possible.
    if v[1] > 0:
        angle = float(np.arctan2(-v[0], v[1]))
    return angle


class LineDetector:
    """Extracts a 1-pixel centerline of the red tape, plus orientation and lateral offset."""

    def __init__(self, resize_width: int = 640):
        self.resize_width = resize_width
        self._prev_centers: Optional[np.ndarray] = None
        self._prev_centerline: Optional[np.ndarray] = None

    def process(self, frame_bgr: np.ndarray, cfg: dict) -> TapeObservation:
        # Resize keeping aspect ratio (makes parameters stable across videos)
        h0, w0 = frame_bgr.shape[:2]
        scale = self.resize_width / float(w0)
        h = int(round(h0 * scale))
        w = int(round(w0 * scale))
        frame = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)

        # Blur -> HSV -> red mask
        blur_k = int(cfg.get("blur_ksize", 5))
        if blur_k % 2 == 0:
            blur_k += 1
        blurred = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        r1l, r1h, r2l, r2h = _as_hsv_ranges(cfg)
        m1 = cv2.inRange(hsv, r1l, r1h)
        m2 = cv2.inRange(hsv, r2l, r2h)
        mask = cv2.bitwise_or(m1, m2)

        # Morphology to fill gaps and remove noise
        open_k = int(cfg.get("open_ksize", 5))
        close_k = int(cfg.get("close_ksize", 7))
        open_k = max(1, open_k | 1)
        close_k = max(1, close_k | 1)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # Keep the largest connected component to reduce false positives
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            best = 1 + int(np.argmax(areas))
            mask = np.where(labels == best, 255, 0).astype(np.uint8)

        # Thinning to 1-pixel skeleton
        centerline = _zs_thinning(mask)

        # Build per-row center points (helps provide stable bottom_x even if skeleton breaks)
        alpha = float(cfg.get("ema_alpha", 0.85))
        min_row_pixels = int(cfg.get("min_row_pixels", 8))
        current_centers = np.full((h,), -1.0, dtype=np.float32)

        # We'll compute centers from the *mask* (more robust), but return the skeleton as centerline
        points = []
        for y in range(h):
            xs = np.where(mask[y] > 0)[0]
            if xs.size < min_row_pixels:
                continue
            cx = float(xs.mean())
            if self._prev_centers is not None and self._prev_centers[y] >= 0:
                cx = alpha * float(self._prev_centers[y]) + (1.0 - alpha) * cx
            current_centers[y] = cx
            points.append((cx, float(y)))

        if len(points) >= 2:
            pts = np.array(points, dtype=np.float32)
            angle = _fit_direction(pts)
        else:
            pts = np.zeros((0, 2), dtype=np.float32)
            angle = float("nan")

        # bottom x from lower 10% of rows
        if len(points) > 0:
            y0 = int(h * 0.9)
            bx_candidates = current_centers[y0:][current_centers[y0:] >= 0]
            bottom_x = float(np.median(bx_candidates)) if bx_candidates.size else float("nan")
        else:
            bottom_x = float("nan")

        # Fallback: if skeleton is too sparse, keep previous (for smoother debug/heading)
        if np.count_nonzero(centerline) < 20 and self._prev_centerline is not None:
            centerline = self._prev_centerline.copy()

        if pts.shape[0] > 0:
            self._prev_centerline = centerline.copy()
            self._prev_centers = current_centers.copy()

        return TapeObservation(
            centerline_px=pts.astype(np.int32) if pts.size else pts,
            angle_rad=angle,
            bottom_x=bottom_x,
            shape_hw=(h, w),
            mask=mask,
        )
