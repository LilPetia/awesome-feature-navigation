# awesome-feature-navigation

Python-проект для оценки **2D траектории** по:
- видео, где робот едет вдоль цветной линии на полу;
- IMU CSV с акселерометром и гироскопом.

Проект умеет:
- выделять линию по цвету в HSV;
- поддерживать пресеты цветов: `red`, `blue`, `green`, `yellow`, `white`;
- автоматически подстраивать HSV-пороги под освещение;
- интерактивно подбирать параметры на видео;
- строить траекторию `(t, x, y, yaw)` и сохранять результат в `csv` и `html`.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Быстрый Старт

Прогон по видео и IMU:

```bash
afn-run \
  --video ../resources/Left_cam.mp4 \
  --imu ../resources/imu_fixed.csv \
  --imu-time-scale 1e-9 \
  --color red \
  --auto-color \
  --config examples/config.yaml \
  --out trajectory
```

Результат:
- `trajectory.csv`
- `trajectory.html`
- `trajectory_debug.mp4`, если добавить `--save-debug`

## Как Подбирать Цвет Линии

Для ручной настройки есть интерактивный тюнер:

```bash
afn-tune --video ../resources/Left_cam.mp4 --config examples/config.yaml --color red --auto-color
```

Что делает тюнер:
- открывает видео;
- показывает исходный кадр, белую маску выделенной линии и centerline;
- позволяет двигать HSV-пороги trackbar-ами;
- сохраняет настройки в YAML по нажатию `s`.

Клавиши в тюнере:
- `q` закрыть окно;
- `s` сохранить текущие параметры;
- `r` сбросить пороги к пресету выбранного цвета;
- `1` красный;
- `2` синий;
- `3` зеленый;
- `4` желтый;
- `5` белый.

## Флаги `afn-run`

- `--video` путь до mp4;
- `--imu` путь до IMU CSV;
- `--imu-time-scale` масштаб времени IMU, например `1e-9` для наносекунд;
- `--config` путь до YAML;
- `--color` цвет линии: `red`, `blue`, `green`, `yellow`, `white`;
- `--auto-color` включить авто-подстройку HSV под сцену;
- `--out` префикс выходных файлов;
- `--save-debug` сохранить debug overlay.

Если `--imu` не передан, проект работает в demo-режиме с постоянной скоростью `forward_speed_mps`.

## Формат Конфига

Пример в [examples/config.yaml](examples/config.yaml).

Основные поля:

```yaml
target_color: red
auto_color_tune: true
hsv_ranges:
  - [0, 110, 70, 10, 255, 255]
  - [170, 110, 70, 180, 255, 255]
vision_yaw_gain: 0.08
vision_yaw_max_correction: 0.12
vision_lateral_gain: 0.002
vision_smoothing_frames: 20
centerline_smooth_window: 11
forward_speed_mps: 0.25
max_frames: 0
```

Пояснение:
- `target_color` какой пресет использовать по умолчанию;
- `auto_color_tune` динамически поджимает HSV-пороги под освещение;
- `hsv_ranges` ручные диапазоны HSV;
- `vision_smoothing_frames` сглаживание `angle_rad` и `bottom_x` по EMA.
- `centerline_smooth_window` сглаживает саму геометрию centerline внутри одного кадра, чтобы линия не была ломаной.

## Поддерживаемые Цвета

- `red`
- `blue`
- `green`
- `yellow`
- `white`

Если в одном проекте видео разные, можно не переписывать код, а просто запускать:

```bash
afn-run --video video_red.mp4 --color red --auto-color --out red_run
afn-run --video video_blue.mp4 --color blue --auto-color --out blue_run
afn-run --video video_white.mp4 --color white --auto-color --out white_run
```

## IMU CSV

Загрузчик принимает гибкие имена колонок. Нужны:
- время: `t`, `time`, `timestamp`, `sec`, `seconds`, `stamp`;
- акселерометр: `ax`, `ay`, `az` или варианты вроде `accel_x`;
- гироскоп: `gx`, `gy`, `gz` или варианты вроде `gyro_z`.

Если IMU timestamps хранятся в наносекундах:

```bash
afn-run --video path/to/video.mp4 --imu path/to/imu.csv --imu-time-scale 1e-9
```

## Что Внутри

- `awesome_feature_navigation/cli.py` — запуск основного пайплайна;
- `awesome_feature_navigation/tuner.py` — интерактивный HSV-тюнер;
- `awesome_feature_navigation/line_detection.py` — детекция цветной линии;
- `awesome_feature_navigation/trajectory.py` — fusion video + IMU;
- `awesome_feature_navigation/plotting.py` — сохранение CSV и HTML-графика.
