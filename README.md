# Awesome Feature Navigation

[![CI](https://github.com/LilPetia/awesome-feature-navigation/actions/workflows/ci.yml/badge.svg?branch=clean-uv-solve)](https://github.com/LilPetia/awesome-feature-navigation/actions/workflows/ci.yml?query=branch%3Aclean-uv-solve)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FLilPetia%2Fawesome-feature-navigation%2Fclean-uv-solve%2Fbadges%2Fcoverage.json&cacheSeconds=300)](https://github.com/LilPetia/awesome-feature-navigation/actions/workflows/ci.yml?query=branch%3Aclean-uv-solve)
[![Python](https://img.shields.io/badge/python-3.10--3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-2f80ed)](https://docs.astral.sh/uv/)
[![OpenCV](https://img.shields.io/badge/vision-OpenCV-green)](https://opencv.org/)
[![IMU](https://img.shields.io/badge/sensors-camera%20%2B%20IMU-orange)](docs/algorithm.md)
[![Version](https://img.shields.io/badge/version-0.1.0-2ea44f)](https://github.com/LilPetia/awesome-feature-navigation)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

`awesome-feature-navigation` - это Python-библиотека и CLI-инструмент для построения 2D-траектории робота по видео с камеры ZED и данным IMU.

Пайплайн использует:

- видео `Right_cam.mp4`, а для двухкамерного сценария также `Left_cam.mp4`;
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
  Left_cam.mp4
  imu_data.csv
  timestamps2.csv
  imu-imu_calibration.yaml
  camchain-imucam-imu_calibration.yaml
```

Видео может не храниться в Git из-за размера. Если `data/Right_cam.mp4` или `data/Left_cam.mp4` отсутствует, положи его в эту папку вручную.

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

Пример с ручными границами кругов (пропускает авто-сегментацию и усредняет ровно по указанным интервалам — полезно, если робот делал «грязный» круг или долго стоял на старте):

```bash
uv run afn-run \
  --video data/Right_cam.mp4 \
  --imu data/imu_data.csv \
  --config configs/right_camera.yaml \
  --lap-bounds "0:11,0:53,1:44" \
  --save-loop-debug \
  --out outputs/manual/right
```

Для `Right_cam.mp4` эти границы уже внесены в основной `configs/right_camera.yaml`: первые два круга размечены как `forward`, третий проезд после разворота - как `reverse`. При построении representative lap reverse-круг разворачивается по порядку точек через `loop_normalize_reverse_laps: true` и учитывается вместе с forward-кругами. `loop_start_anchor: manual_start` оставляет старт канонического круга около ручной границы, а не переносит его в автоматический `bottom_left` anchor. Полный путь с исходным обратным проездом остается в `_smoothed.csv/html` при `--save-loop-debug`. Отдельный `configs/right_camera_2laps.yaml` оставлен как вариант, который жестко обрезает запись после второго круга через `max_frames: 3120`:

```bash
uv run afn-run \
  --video data/Right_cam.mp4 \
  --imu data/imu_data.csv \
  --config configs/right_camera_2laps.yaml \
  --save-loop-debug \
  --out outputs/twolaps/right
```

## Основные Флаги

`--video` - путь к видео с правой камеры.

`--imu` - путь к IMU CSV.

`--config` - YAML-конфиг пайплайна.

`--out` - префикс выходных файлов без расширения.

`--save-debug` - сохранить debug overlay video.

`--save-loop-debug` - сохранить дополнительные `raw`, `smoothed`, diagnostics и, если включен `loop_average`, loop/lap debug outputs.

`--lap-bounds` - явные границы кругов через запятую в секундах или формате `M:S`/`H:M:S` (например, `"0:11,0:53,1:44"`). Если задан, авто-сегментация кругов пропускается и канонический круг усредняется ровно по указанным интервалам.

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
- параметры поиска цветной линии;
- режим `trajectory_mode: auto` с приоритетом line-based `tape_line`;
- офлайн-сглаживание наблюдений линии по времени;
- адаптивный confidence threshold;
- variable-speed correction для offline-режима;
- замкнутую финальную representative-lap траекторию, если по видео надежно найден период круга.

Обычно менять CLI-команду не нужно. Если меняется видео или IMU, достаточно подставить новые пути через `--video` и `--imu`.

## Как Работает Пайплайн

1. `afn-run` читает YAML-конфиг и аргументы командной строки.
2. Загружает timestamps кадров из `timestamps2.csv`.
3. Загружает IMU CSV и переводит время из наносекунд в секунды.
4. Синхронизирует видео и IMU по абсолютному времени.
5. Применяет Kalibr `timeshift_cam_imu`.
6. Поворачивает accel/gyro из IMU frame в camera frame через `T_cam_imu`.
7. В режиме `auto` сначала пробует line-based `tape_line`, если линия найдена достаточно надежно.
8. Если `tape_line` непригоден, использует fallback `generic_vio`.
9. Для `tape_line` строит `raw_traj`, затем `smoothed_traj` через offline smoothing, adaptive confidence и variable-speed correction.
10. Если найден надежный повтор петли, строит замкнутую `final_traj` как representative lap. Полный многокруговой путь остается доступен в `_smoothed.csv/html` при `--save-loop-debug`.
11. Сохраняет финальный результат в CSV и HTML.

Подробное описание алгоритма:

```text
docs/algorithm.md
```

## Двухкамерное Усреднение

Для двух камер ZED сначала независимо строятся две траектории: одна по `Left_cam.mp4`, вторая по `Right_cam.mp4`. Это одна и та же физическая поездка робота, снятая двумя разными "глазами", поэтому после построения 2D-траекторий их можно объединить командой `afn-fuse`.

Усреднять нужно именно траектории после построения, а не сырые пиксели: левая и правая камеры видят одну сцену из разных оптических центров, поэтому pixel motion и видимая centerline отличаются. После перевода каждого видео в 2D-путь эти различия становятся ошибками оценки, которые можно уменьшать выравниванием и точечным средним.

Пример полного запуска:

```bash
uv run afn-run \
  --video data/Left_cam.mp4 \
  --imu data/imu_data.csv \
  --config configs/right_camera.yaml \
  --out outputs/left_trajectory

uv run afn-run \
  --video data/Right_cam.mp4 \
  --imu data/imu_data.csv \
  --config configs/right_camera.yaml \
  --out outputs/right_trajectory

uv run afn-fuse \
  --left outputs/left_trajectory.csv \
  --right outputs/right_trajectory.csv \
  --out outputs/fused_trajectory \
  --save-aligned-debug
```

`afn-fuse` приводит LEFT к системе RIGHT: ресэмплирует обе замкнутые петли, проверяет обратное направление LEFT, выравнивает фазу, применяет rigid Procrustes alignment без scale-подгонки и сохраняет среднюю траекторию в `outputs/fused_trajectory.csv/html`. При `--save-aligned-debug` дополнительно сохраняются `outputs/fused_trajectory_left_aligned.*` и `outputs/fused_trajectory_right_resampled.*`.

## Структура Проекта

```text
src/awesome_feature_navigation/
  __init__.py            # package metadata
  cli.py                 # CLI entry point
  calibration.py         # timestamps and YAML calibration loaders
  imu_io.py              # IMU CSV loading and calibration
  imu_preintegration.py  # IMU integration between frames
  trajectory.py          # trajectory estimation
  line_detection.py      # colored line detection fallback
  auto_config.py         # automatic HSV/color configuration
  plotting.py            # CSV and HTML outputs
  fusion.py              # LEFT/RIGHT trajectory alignment and averaging
  fusion_cli.py          # afn-fuse CLI entry point
```

## Разработка

Установить dev-зависимости:

```bash
make sync-dev
```

Запустить все проверки как в CI:

```bash
make
```

То же самое явно:

```bash
make check
```

Отдельные проверки:

```bash
make lint     # ruff: стиль, импорты, часть потенциальных ошибок, аннотации
make type     # mypy: статическая проверка типов
make test     # pytest + coverage, минимум 95%
make compile  # compileall для src и tests
make fix      # ruff auto-fix для безопасных исправлений
```

Что проверяют тесты:

- `tests/test_calibration.py` проверяет чтение Kalibr IMU YAML, camchain YAML, frame timestamps CSV, сортировку кадров, обработку пустых/битых calibration-файлов и защиту от нечисловых timestamps.
- `tests/test_imu_io.py` и `tests/test_imu_more.py` проверяют загрузку IMU CSV, подбор колонок, scale времени/гироскопа, remap осей, поворот в camera frame, gravity alignment, yaw-only calibration, сдвиг и нарезку IMU по временным границам, fallback preintegration и fake-GTSAM ветку.
- `tests/test_line_detection.py` и `tests/test_line_detection_more.py` проверяют выбор цвета линии, нормализацию HSV-диапазонов, legacy red-конфиги, геометрию centerline, оценку локального/глобального угла, bottom-x, auto-tune HSV и fallback на предыдущую centerline при плохом кадре.
- `tests/test_auto_config.py` проверяет сэмплирование кадров видео, scoring кандидатов цвета, построение HSV-диапазонов по пикселям линии, отбрасывание слабых наблюдений, downsample больших масок и применение/отключение auto video config.
- `tests/test_offline_tape_line.py` проверяет confidence наблюдений линии, offline smoothing по времени, adaptive confidence threshold, разделение `raw_traj`/`smoothed_traj`/`final_traj`, работу с reverse-кругами, loop canonicalization, soft closure и output flip.
- `tests/test_trajectory_helpers.py` и `tests/test_trajectory_edge_cases.py` проверяют чистую математику траектории: Procrustes alignment, similarity alignment, resampling по длине дуги, anchor-search, manual/periodic lap extraction, Fourier smoothing, spline, representative lap, quality gate, speed scale solver, output transforms и отказ от плохой канонизации.
- `tests/test_trajectory_video_paths.py` проверяет video-dependent ветки через fake `VideoCapture`/`VideoWriter`: debug-video для `tape_line`, синтетический `tape_line`, синтетический `generic_vio` и dispatch между режимами.
- `tests/test_cli_helpers.py` проверяет парсинг CLI, `--lap-bounds`, bool/path/time normalization, merge calibration config, wiring основного `cli.main()`, debug outputs, diagnostics и loop debug saves.
- `tests/test_fusion.py` проверяет загрузку trajectory CSV, выравнивание LEFT к RIGHT, поиск reverse-направления, phase alignment, сохранение fused/debug outputs через `afn-fuse` и ошибки на плохих входах.
- `tests/test_plotting.py` проверяет CSV/HTML export для траектории, diagnostics и loop debug, повышенное качество Plotly PNG export, отсутствие `kaleido` и предупреждения на пустых входах.

Их общая задача - зафиксировать контракты вокруг данных, калибровок, времени, IMU, line detection, auto-config, offline trajectory math, video IO, CLI и plotting. Тесты используют синтетические данные и fake-объекты, чтобы проверять общий алгоритм.

Проверить только импорт и синтаксис без остальных проверок:

```bash
uv run python -m compileall src
```

Запустить CLI из исходников:

```bash
uv run afn-run --help
```
