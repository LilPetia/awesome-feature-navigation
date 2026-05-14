# awesome-feature-navigation

Минимальная `uv`-библиотека для построения 2D-траектории по правой камере ZED и IMU.

Ветка содержит только то, что нужно для текущего solve:

- Python-пакет в `src/awesome_feature_navigation`;
- конфиг запуска `configs/right_camera.yaml`;
- данные в `data/`: `Right_cam.mp4`, `imu_data.csv`, `timestamps2.csv`, `imu-imu_calibration.yaml`, `camchain-imucam-imu_calibration.yaml`;
- описание алгоритма в `docs/algorithm.md`.

## Установка

```bash
uv sync
```

## Запуск

```bash
uv run afn-run \
  --video data/Right_cam.mp4 \
  --imu data/imu_data.csv \
  --config configs/right_camera.yaml \
  --out outputs/right_trajectory
```

Результаты:

- `outputs/right_trajectory.csv`;
- `outputs/right_trajectory.html`.

Для debug-видео:

```bash
uv run afn-run \
  --video data/Right_cam.mp4 \
  --imu data/imu_data.csv \
  --config configs/right_camera.yaml \
  --save-debug \
  --out outputs/right_trajectory
```

Тогда дополнительно появится `outputs/right_trajectory_debug.mp4`.

## Основные модули

- `cli.py` - точка входа `afn-run`;
- `calibration.py` - чтение timestamps и YAML-калибровок;
- `imu_io.py` - чтение и подготовка IMU;
- `imu_preintegration.py` - IMU preintegration;
- `trajectory.py` - построение траектории;
- `line_detection.py` - детекция цветной линии для fallback-режима;
- `auto_config.py` - автоопределение цвета/HSV;
- `plotting.py` - сохранение CSV/HTML.

## Что не включено

В ветке намеренно нет `Left_cam.mp4`, `imu_fixed.csv`, HSV-тюнера, скриптов сбора calibration images, старых рабочих markdown-файлов, IDE-файлов и уже сгенерированных trajectory outputs.
