# Awesome Feature Navigation

[![CI](https://github.com/LilPetia/awesome-feature-navigation/actions/workflows/ci.yml/badge.svg)](https://github.com/LilPetia/awesome-feature-navigation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10--3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Codecov](https://codecov.io/gh/LilPetia/awesome-feature-navigation/branch/main/graph/badge.svg)](https://codecov.io/gh/LilPetia/awesome-feature-navigation)
[![Version](https://img.shields.io/badge/version-0.1.0-2ea44f)](https://github.com/LilPetia/awesome-feature-navigation)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

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

Альтернатива: жёстко обрезать видео по кадрам через отдельный конфиг с `max_frames`. Готовый пример `configs/right_camera_2laps.yaml` (`max_frames: 3120`, ≈ 1:44 при 30 fps) исключает всё после третьего круга:

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
make test     # pytest + coverage
make compile  # compileall для src и tests
make fix      # ruff auto-fix для безопасных исправлений
```

Что строго проверяют тесты:

`tests/test_calibration.py`

- `test_load_imu_calibration_config_maps_kalibr_fields` создает временный Kalibr IMU YAML с секцией `imu0` и проверяет, что поля шумов, random walk, update rate и time offset переносятся в внутренние ключи конфига без потери численных значений.
- `test_load_camchain_calibration_config_extracts_transform_and_camera_metadata` создает временный camchain YAML и проверяет, что `T_cam_imu` читается как 4x4 transform, верхний левый 3x3 блок становится `imu_camera_rotation`, последний столбец становится `imu_camera_translation_m`, а `timeshift_cam_imu`, intrinsics, distortion, model, resolution и rostopic сохраняются в ожидаемых ключах.
- `test_load_frame_timestamps_csv_sorts_by_frame_and_scales_time` создает CSV с кадрами в неправильном порядке и проверяет, что timestamps сначала сортируются по `frame_idx`, потом масштабируются через `time_scale`. Это защищает синхронизацию видео и IMU от ошибки порядка кадров.

`tests/test_imu_io.py`

- `test_shift_imu_samples_offsets_time_without_mutating_vectors` проверяет, что сдвиг времени меняет только `t`, но не меняет значения `accel` и `omega`.
- `test_slice_imu_includes_neighbor_samples_for_integration_boundaries` проверяет, что при вырезании IMU-отрезка функция добавляет соседние сэмплы до и после интервала. Это важно для интегрирования между двумя кадрами: границы интервала не должны терять ближайшие измерения.
- `test_slice_imu_returns_empty_segment_when_no_samples_are_available` фиксирует поведение на пустом IMU-входе: функция должна вернуть пустой сегмент и индекс `0`, а не падать.
- `test_shift_imu_samples_preserves_numeric_arrays` проверяет, что после сдвига времени массивы accel/gyro остаются численно теми же через `np.testing.assert_allclose`.

`tests/test_line_detection.py`

- `test_resolve_target_color_defaults_to_blue_for_missing_or_invalid_values` проверяет дефолтный цвет: если `target_color` отсутствует или задан неверно, используется `blue`.
- `test_resolve_hsv_ranges_uses_blue_preset_by_default` проверяет конкретный HSV-пресет для blue: `[95, 100, 60]..[130, 255, 255]`.
- `test_legacy_red_hsv_ranges_apply_only_when_red_is_selected` проверяет, что старые ключи `hsv_red1_low/high` не перехватывают blue-режим. Это важно, потому что цвет линии в разных видео может быть разным, а legacy-настройки red не должны ломать текущий auto/blue сценарий.
- `test_legacy_red_hsv_ranges_are_used_for_explicit_red` проверяет обратный случай: если явно выбран `target_color: red`, старые red HSV-диапазоны продолжают работать, причем low/high нормализуются в правильном порядке.

`tests/test_offline_tape_line.py`

- `test_line_observation_confidence_is_positive_for_clean_centerline` строит искусственную маску линии и centerline без видео и проверяет, что confidence у такого наблюдения больше `0.5`.
- `test_offline_tape_line_smoothing_ignores_low_confidence_angle_outlier` строит последовательность из пяти искусственных наблюдений: четыре надежных с углом `0`, одно низкоуверенное с выбросом угла `1.5 rad`. Тест проверяет, что offline smoothing игнорирует этот выброс, valid ratio равен `0.8`, финальная траектория идет прямо по `x`, `y` остается около `0`, yaw остается около `0`.
- `test_time_based_smoothing_window_uses_frame_timestamps` проверяет, что окно сглаживания строится из секунд и реального шага timestamps, а не из жестко заданного числа кадров.
- `test_adaptive_confidence_threshold_uses_video_distribution` проверяет, что confidence threshold поднимается относительно нижней границы, если распределение confidence на записи это позволяет.
- `test_offline_tape_line_estimate_separates_raw_smoothed_and_final_outputs` проверяет, что offline-estimator возвращает отдельные `raw_traj`, `smoothed_traj`, `final_traj` и diagnostics, а низкоуверенный выброс не попадает в valid mask.
- `test_loop_canonicalization_rejects_poorly_aligned_laps` проверяет, что финальная замкнутая петля не принимается, если повторяющиеся круги плохо накладываются относительно размера траектории.

Их задача сейчас - зафиксировать критичные контракты вокруг калибровок, времени, IMU-нарезки, выбора цвета линии, adaptive confidence, time-based smoothing, offline-сглаживания `tape_line` и quality gate для замкнутой петли, чтобы эти части случайно не сломались при следующих правках.

Проверить только импорт и синтаксис без остальных проверок:

```bash
uv run python -m compileall src
```

Запустить CLI из исходников:

```bash
uv run afn-run --help
```

Локальные результаты сохраняй в `outputs/`. Эта папка предназначена для generated-файлов и не должна превращаться в источник данных.
