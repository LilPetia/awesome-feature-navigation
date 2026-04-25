from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import yaml
from .line_detection import COLOR_PRESETS, LineDetector, SUPPORTED_COLORS, resolve_hsv_ranges, resolve_target_color
COLOR_KEYS = {ord('1'): 'red', ord('2'): 'blue', ord('3'): 'green', ord('4'): 'yellow', ord('5'): 'white'}

def _nothing(x: int) -> None:
    """Заглушка-callback для cv2.createTrackbar."""
    return

def _load_cfg(path: str) -> Dict:
    """Загрузить YAML-конфиг или вернуть {} если файл отсутствует."""
    p = Path(path)
    if p.exists():
        return yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    return {}

def _save_cfg(path: str, cfg: Dict) -> None:
    """Сохранить cfg в YAML-файл (порядок ключей сохраняется)."""
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

def _preset_ranges(color: str) -> List[List[int]]:
    """Получить дефолтные HSV-диапазоны для заданного цвета (как плоские 6-числовые списки)."""
    preset = COLOR_PRESETS[color]
    return [list(low) + list(high) for low, high in preset]

def _set_trackbars(win: str, color_name: str, ranges: List[List[int]], auto_color: bool) -> None:
    """Выставить положения всех трекбаров окна в соответствии с цветом и HSV-диапазонами."""
    ranges = list(ranges)
    if len(ranges) < 2:
        ranges.append([0, 0, 0, 0, 0, 0])
    cv2.setTrackbarPos('Color', win, SUPPORTED_COLORS.index(color_name))
    cv2.setTrackbarPos('Auto', win, 1 if auto_color else 0)
    cv2.setTrackbarPos('Range2', win, 1 if any(ranges[1]) else 0)
    for idx, values in enumerate(ranges[:2], start=1):
        names = ('H low', 'S low', 'V low', 'H high', 'S high', 'V high')
        for name, value in zip(names, values):
            cv2.setTrackbarPos(f'R{idx} {name}', win, int(value))

def _read_trackbars(win: str) -> Dict:
    """Прочитать состояние всех трекбаров и собрать cfg-словарь (target_color, auto, hsv_ranges)."""
    color_idx = cv2.getTrackbarPos('Color', win)
    color_name = SUPPORTED_COLORS[min(color_idx, len(SUPPORTED_COLORS) - 1)]
    auto_color = bool(cv2.getTrackbarPos('Auto', win))
    use_range2 = bool(cv2.getTrackbarPos('Range2', win))
    ranges: List[List[int]] = []
    for idx in (1, 2):
        values = [cv2.getTrackbarPos(f'R{idx} H low', win), cv2.getTrackbarPos(f'R{idx} S low', win), cv2.getTrackbarPos(f'R{idx} V low', win), cv2.getTrackbarPos(f'R{idx} H high', win), cv2.getTrackbarPos(f'R{idx} S high', win), cv2.getTrackbarPos(f'R{idx} V high', win)]
        if idx == 1 or use_range2:
            ranges.append(values)
    return {'target_color': color_name, 'auto_color_tune': auto_color, 'hsv_ranges': ranges}

def main() -> None:
    """Точка входа CLI `afn-tune`: интерактивный подбор HSV-порогов под видео в окне OpenCV."""
    ap = argparse.ArgumentParser(description='Interactive HSV tuner for tape detection')
    ap.add_argument('--video', required=True)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--color', choices=SUPPORTED_COLORS, default=None)
    ap.add_argument('--auto-color', action='store_true')
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.color is not None:
        cfg['target_color'] = args.color
    if args.auto_color:
        cfg['auto_color_tune'] = True
    detector = LineDetector()
    color_name = resolve_target_color(cfg)
    initial_ranges = [list(low) + list(high) for low, high in resolve_hsv_ranges(cfg)]
    auto_color = bool(cfg.get('auto_color_tune', False))
    win = 'AFN HSV Tuner'
    cv2.namedWindow(win)
    cv2.createTrackbar('Color', win, SUPPORTED_COLORS.index(color_name), len(SUPPORTED_COLORS) - 1, _nothing)
    cv2.createTrackbar('Auto', win, 1 if auto_color else 0, 1, _nothing)
    cv2.createTrackbar('Range2', win, 1 if len(initial_ranges) > 1 else 0, 1, _nothing)
    for idx in (1, 2):
        defaults = initial_ranges[idx - 1] if idx <= len(initial_ranges) else [0, 0, 0, 0, 0, 0]
        names = ('H low', 'S low', 'V low', 'H high', 'S high', 'V high')
        max_values = (180, 255, 255, 180, 255, 255)
        for name, default, max_value in zip(names, defaults, max_values):
            cv2.createTrackbar(f'R{idx} {name}', win, int(default), int(max_value), _nothing)
    _set_trackbars(win, color_name, initial_ranges, auto_color)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')
    print('Keys: q=quit, s=save, 1=red, 2=blue, 3=green, 4=yellow, 5=white, r=reset preset')
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        current_cfg = _read_trackbars(win)
        obs = detector.process(frame, current_cfg)
        scale = 0.8
        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        mask_bgr = cv2.cvtColor(obs.mask, cv2.COLOR_GRAY2BGR)
        centerline_bgr = cv2.cvtColor(obs.centerline_mask, cv2.COLOR_GRAY2BGR)
        info = f"color={current_cfg['target_color']} auto={('on' if current_cfg['auto_color_tune'] else 'off')} angle={obs.angle_rad:.3f} bottom_x={obs.bottom_x:.1f}"
        cv2.putText(frame_resized, info, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        combined = np.hstack([frame_resized, mask_bgr, centerline_bgr])
        cv2.imshow(win, combined)
        key = cv2.waitKey(30) & 255
        if key == ord('q'):
            break
        if key == ord('s'):
            cfg.update(current_cfg)
            _save_cfg(args.config, cfg)
            print(f'Saved config -> {args.config}')
        if key == ord('r'):
            color_name = current_cfg['target_color']
            _set_trackbars(win, color_name, _preset_ranges(color_name), current_cfg['auto_color_tune'])
        if key in COLOR_KEYS:
            color_name = COLOR_KEYS[key]
            _set_trackbars(win, color_name, _preset_ranges(color_name), current_cfg['auto_color_tune'])
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
