from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import cv2
import yaml
import numpy as np


def _nothing(x: int) -> None:
    return


def _load_cfg(path: str) -> Dict:
    p = Path(path)
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


def _save_cfg(path: str, cfg: Dict) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune HSV thresholds for red tape")
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)

    # Trackbars for two red ranges in HSV
    win = "tuner"
    cv2.namedWindow(win)

    def make_tb(name: str, init: int) -> None:
        cv2.createTrackbar(name, win, int(init), 255, _nothing)

    # defaults
    defaults = {
        "r1_h_low": 0, "r1_h_high": 10,
        "r1_s_low": 80, "r1_s_high": 255,
        "r1_v_low": 80, "r1_v_high": 255,
        "r2_h_low": 160, "r2_h_high": 180,
        "r2_s_low": 80, "r2_s_high": 255,
        "r2_v_low": 80, "r2_v_high": 255,
    }

    for k, v in defaults.items():
        make_tb(k, int(cfg.get(k, v)))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        vals = {k: cv2.getTrackbarPos(k, win) for k in defaults.keys()}

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        r1l = np.array([vals["r1_h_low"], vals["r1_s_low"], vals["r1_v_low"]], dtype=np.uint8)
        r1h = np.array([vals["r1_h_high"], vals["r1_s_high"], vals["r1_v_high"]], dtype=np.uint8)
        r2l = np.array([vals["r2_h_low"], vals["r2_s_low"], vals["r2_v_low"]], dtype=np.uint8)
        r2h = np.array([vals["r2_h_high"], vals["r2_s_high"], vals["r2_v_high"]], dtype=np.uint8)

        m1 = cv2.inRange(hsv, r1l, r1h)
        m2 = cv2.inRange(hsv, r2l, r2h)
        mask = cv2.bitwise_or(m1, m2)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        vis = cv2.addWeighted(frame, 1.0, mask_bgr, 0.4, 0.0)
        cv2.imshow(win, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            # save in the config format used by the pipeline
            out_cfg = dict(cfg)
            out_cfg.update({
                "hsv_red1_low":  [vals["r1_h_low"], vals["r1_s_low"], vals["r1_v_low"]],
                "hsv_red1_high": [vals["r1_h_high"], vals["r1_s_high"], vals["r1_v_high"]],
                "hsv_red2_low":  [vals["r2_h_low"], vals["r2_s_low"], vals["r2_v_low"]],
                "hsv_red2_high": [vals["r2_h_high"], vals["r2_s_high"], vals["r2_v_high"]],
            })
            # also keep raw trackbars if you want
            out_cfg.update(vals)
            _save_cfg(args.config, out_cfg)
            print(f"Saved config -> {args.config}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
