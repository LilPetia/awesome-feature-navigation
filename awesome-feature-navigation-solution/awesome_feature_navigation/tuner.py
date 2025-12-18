from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import cv2
import yaml
import numpy as np

from .line_detection import LineDetector


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    detector = LineDetector()

    win = "Tuner"
    cv2.namedWindow(win)

    def make_tb(name: str, default: int) -> None:
        val = int(cfg.get(name, default))
        cv2.createTrackbar(name, win, val, 255, _nothing)

    make_tb("LB", 0)
    make_tb("LG", 0)
    make_tb("LR", 200)
    make_tb("HB", 255)
    make_tb("HG", 255)
    make_tb("HR", 255)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        current_cfg = {
            "LB": cv2.getTrackbarPos("LB", win),
            "LG": cv2.getTrackbarPos("LG", win),
            "LR": cv2.getTrackbarPos("LR", win),
            "HB": cv2.getTrackbarPos("HB", win),
            "HG": cv2.getTrackbarPos("HG", win),
            "HR": cv2.getTrackbarPos("HR", win),
        }

        obs = detector.process(frame, current_cfg)

        scale = 0.8
        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        line_bgr = cv2.cvtColor(obs.mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([frame_resized, line_bgr])

        cv2.imshow(win, combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cfg.update(current_cfg)
            _save_cfg(args.config, cfg)
            print(f"Saved config -> {args.config}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()