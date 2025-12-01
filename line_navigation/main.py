import cv2
import numpy as np
import os

from classes.config_tuner import ConfigTuner
from classes.line_detector import LineDetector


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    # Ищем конфиг в корне проекта
    tuner = ConfigTuner(config_path="line_config.yaml")
    detector = LineDetector()

    print("=== Line Detection Pipeline ===")
    print("Управление: [s] - Сохранить конфиг, [q] - Выход")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        thresholds = tuner.get_values()

        # Получаем обработанные кадры
        resized_frame, raw_mask, result_line = detector.process(frame, thresholds)

        # Визуализация
        raw_mask_bgr = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
        result_line_bgr = cv2.cvtColor(result_line, cv2.COLOR_GRAY2BGR)

        # --- ИЗМЕНЕНИЕ 2: Цвет линии ---
        # Мы просто удалили код, который красил линию в зеленый.
        # Теперь result_line_bgr останется черно-белым.

        # Склейка окон
        combined_view = np.hstack([resized_frame, raw_mask_bgr, result_line_bgr])

        cv2.imshow("Pipeline View", combined_view)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            tuner.save_config()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_FILE = os.path.join("resources", "Left_cam.mp4")

    if not os.path.exists(VIDEO_FILE):
        print(f"ОШИБКА: Файл {VIDEO_FILE} не найден.")
    else:
        main(VIDEO_FILE)
