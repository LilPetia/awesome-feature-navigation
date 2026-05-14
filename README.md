# Awesome Feature Navigation

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-2f80ed)
![OpenCV](https://img.shields.io/badge/vision-OpenCV-green)
![IMU](https://img.shields.io/badge/sensors-camera%20%2B%20IMU-orange)

`awesome-feature-navigation` - это Python-библиотека и CLI-инструмент для построения 2D-траектории робота по видео с правой камеры ZED и данным IMU.

Пайплайн использует:

- видео `Right_cam.mp4`;
- IMU CSV с акселерометром и гироскопом;
- timestamps кадров из SVO/ZED;
- Kalibr IMU calibration;
- Kalibr camera-IMU camchain;
- OpenCV optical flow и fallback-режим по цветной линии.

## Быстрый Старт

Установить зависимости:

```bash
uv sync
```

Запустить построение траектории:

```bash
uv run afn-run \
  --video data/Right_cam.mp4 \
  --imu data/imu_data.csv \
  --config configs/right_camera.yaml \
  --out outputs/right_trajectory
```

После запуска появятся:

```text
outputs/right_trajectory.csv
outputs/right_trajectory.html
```

HTML-файл можно открыть в браузере и посмотреть интерактивный график траектории.

## Подготовка Данных

Ожидаемая структура входных файлов:

```text
data/
  Right_cam.mp4
  imu_data.csv
  timestamps2.csv
  imu-imu_calibration.yaml
  camchain-imucam-imu_calibration.yaml
```

Видео может не храниться в Git из-за размера. Если `data/Right_cam.mp4` отсутствует, положи его в эту папку вручную.

`imu_data.csv` лучше использовать в исходном виде, с абсолютными timestamps. Не нужно заранее переводить его в `imu_fixed.csv`, если используется `timestamps2.csv`: библиотека сама синхронизирует видео и IMU по абсолютному времени.

## Команды

Основная команда:

```bash
uv run afn-run --video VIDEO --imu IMU --config CONFIG --out OUTPUT_PREFIX
```

Пример с debug-видео:

```bash
uv run afn-run \
  --video data/Right_cam.mp4 \
  --imu data/imu_data.csv \
  --config configs/right_camera.yaml \
  --save-debug \
  --out outputs/right_trajectory
```

Будут созданы:

```text
outputs/right_trajectory.csv
outputs/right_trajectory.html
outputs/right_trajectory_debug.mp4
```

Справка по CLI:

```bash
uv run afn-run --help
```

## Основные Флаги

`--video` - путь к видео с правой камеры.

`--imu` - путь к IMU CSV.

`--config` - YAML-конфиг пайплайна.

`--out` - префикс выходных файлов без расширения.

`--save-debug` - сохранить debug overlay video.

`--mode` - режим построения траектории: `auto`, `generic_vio`, `tape_line`.

`--frame-timestamps` - CSV с `frame_idx,timestamp_ns`, если нужно переопределить путь из конфига.

`--imu-calibration` - YAML с шумами IMU.

`--camchain` - YAML с `T_cam_imu` и `timeshift_cam_imu`.

## Конфигурация

Основной конфиг лежит в:

```text
configs/right_camera.yaml
```

В нем уже прописаны:

- пути к timestamps и calibration YAML;
- масштабы времени и гироскопа;
- применение `T_cam_imu`;
- Kalibr time shift;
- gravity alignment;
- параметры поиска синей линии (`target_color: blue`);
- режим `trajectory_mode: auto`.

Обычно менять CLI-команду не нужно. Если меняется видео или IMU, достаточно подставить новые пути через `--video` и `--imu`.

## Как Работает Пайплайн

1. `afn-run` читает YAML-конфиг и аргументы командной строки.
2. Загружает timestamps кадров из `timestamps2.csv`.
3. Загружает IMU CSV и переводит время из наносекунд в секунды.
4. Синхронизирует видео и IMU по абсолютному времени.
5. Применяет Kalibr `timeshift_cam_imu`.
6. Поворачивает accel/gyro из IMU frame в camera frame через `T_cam_imu`.
7. Строит траекторию в режиме `generic_vio`.
8. Если `generic_vio` выглядит нестабильно, `auto` может перейти в `tape_line`.
9. Сохраняет результат в CSV и HTML.

Подробное описание алгоритма:

```text
docs/algorithm.md
```

## Структура Проекта

```text
src/awesome_feature_navigation/
  cli.py                 # CLI entry point
  calibration.py         # timestamps and YAML calibration loaders
  imu_io.py              # IMU CSV loading and calibration
  imu_preintegration.py  # IMU integration between frames
  trajectory.py          # trajectory estimation
  line_detection.py      # colored line detection fallback
  auto_config.py         # automatic HSV/color configuration
  plotting.py            # CSV and HTML outputs
```

## Разработка

Проверить импорт и синтаксис:

```bash
uv run python -m compileall src
```

Запустить CLI из исходников:

```bash
uv run afn-run --help
```

Локальные результаты сохраняй в `outputs/`. Эта папка предназначена для generated-файлов и не должна превращаться в источник данных.

## Частые Проблемы

Если команда не находит видео, проверь путь:

```bash
ls data/Right_cam.mp4
```

Если траектория строится в неверном масштабе, не включай `--imu-use-translation` без отдельной проверки: двойное интегрирование IMU быстро накапливает ошибку.

Если цветная линия не находится, попробуй явно указать цвет:

```bash
uv run afn-run \
  --video data/Right_cam.mp4 \
  --imu data/imu_data.csv \
  --config configs/right_camera.yaml \
  --color blue \
  --auto-color \
  --out outputs/right_trajectory
```
