# Спецификация алгоритма построения траектории по видео и IMU

## 1. Назначение документа

Документ описывает текущий алгоритм проекта `awesome-feature-navigation` для построения 2D-траектории робота по:

- видео с правой камеры;
- timestamps видеокадров;
- IMU-измерениям;
- YAML-калибровкам камеры и IMU;
- видимой цветной линии на полу.

Это спецификация фактической реализации в репозитории. Она не описывает идеальный visual-inertial SLAM и не утверждает, что текущий код решает полную SLAM-задачу.

Для текущего видео линия, по которой едет робот, синяя. Поэтому проектная конфигурация использует `target_color: blue`, а явный CLI-флаг должен быть `--color blue`.

## 2. Модули реализации

| Зона ответственности | Модуль |
| --- | --- |
| CLI, объединение конфига, запуск пайплайна | `src/awesome_feature_navigation/cli.py` |
| Загрузка frame timestamps и Kalibr YAML | `src/awesome_feature_navigation/calibration.py` |
| Загрузка IMU CSV, сдвиг времени, remap осей, поворот в camera frame | `src/awesome_feature_navigation/imu_io.py` |
| IMU preintegration, GTSAM-ветка и fallback-интегратор | `src/awesome_feature_navigation/imu_preintegration.py` |
| Детекция синей линии и построение centerline | `src/awesome_feature_navigation/line_detection.py` |
| Автонастройка цвета и HSV по сэмплам видео | `src/awesome_feature_navigation/auto_config.py` |
| Оценка траектории, выбор режима, loop averaging | `src/awesome_feature_navigation/trajectory.py` |
| Сохранение CSV и HTML-графиков | `src/awesome_feature_navigation/plotting.py` |

## 3. Входные данные

### 3.1 Основные входы запуска

| Вход | Проектный путь | Назначение |
| --- | --- | --- |
| Видео | `data/Right_cam.mp4` | Видеопоток с правой камеры. |
| IMU CSV | `data/imu_data.csv` | Сэмплы IMU: timestamp, acceleration, angular velocity. |
| Основной конфиг | `configs/right_camera.yaml` | Параметры пайплайна и пути к дополнительным файлам. |
| Output prefix | `outputs/right_trajectory` | Префикс выходных CSV/HTML/debug-файлов. |

### 3.2 Входы из конфига

| Вход | Проектный путь | Назначение |
| --- | --- | --- |
| Frame timestamps | `data/timestamps2.csv` | Реальные timestamps кадров из SVO/ZED. |
| IMU calibration YAML | `data/imu-imu_calibration.yaml` | Шумы IMU и random walk из Kalibr. |
| Camera-IMU camchain YAML | `data/camchain-imucam-imu_calibration.yaml` | `T_cam_imu`, `timeshift_cam_imu`, intrinsics и distortion metadata. |

### 3.3 Типовая команда

```bash
uv run afn-run \
  --video data/Right_cam.mp4 \
  --imu data/imu_data.csv \
  --config configs/right_camera.yaml \
  --color blue \
  --auto-color \
  --out outputs/right_trajectory
```

## 4. Выходные данные

| Выход | Условие создания | Содержание |
| --- | --- | --- |
| `outputs/right_trajectory.csv` | Всегда | Финальная траектория в формате `t,x,y,yaw`. |
| `outputs/right_trajectory.html` | Всегда | Интерактивный Plotly-график траектории. Сама траектория рисуется синим (`royalblue`). |
| `outputs/right_trajectory_debug.mp4` | При `--save-debug` | Debug-видео с overlay для optical flow или line detection. |
| `outputs/right_trajectory_raw.csv/html` | При `--save-loop-debug`, если есть loop diagnostics | Raw trajectory до loop averaging. |
| `outputs/right_trajectory_laps.csv/html` | При `--save-loop-debug`, если есть loop diagnostics | Диагностика кругов, alignment и projection. |

Красный marker в HTML-графике, если он есть, обозначает только конечную точку траектории. Он не обозначает цвет линии на полу и не означает, что траектория робота красная.

## 5. Соглашения по времени и координатам

### 5.1 Единицы времени

Внутри алгоритма все время хранится в секундах.

Для текущих файлов используются масштабы:

```yaml
frame_timestamp_time_scale: 1.0e-9
imu_time_scale: 1.0e-9
```

То есть timestamps кадров и IMU timestamps, записанные в наносекундах, переводятся в секунды.

### 5.2 Приведение video time и IMU time к общей шкале

Если первый timestamp видео и первый timestamp IMU выглядят как абсолютное Unix-like время, оба потока нормализуются относительно первого видеокадра:

```text
t_frame_rel = t_frame_abs - t_frame0_abs
t_imu_rel   = t_imu_abs   - t_frame0_abs
```

После этого:

```text
t = 0
```

соответствует первому видеокадру.

### 5.3 Kalibr `timeshift_cam_imu`

В коде используется соглашение Kalibr:

```text
t_imu = t_cam + timeshift_cam_imu
```

Чтобы нарезать IMU по timestamp видеокадров, IMU timestamps переводятся в camera clock:

```text
t_cam = t_imu - timeshift_cam_imu
```

В реализации это делается так:

```text
shift_imu_samples(samples, -timeshift_cam_imu)
```

### 5.4 Переход из IMU frame в camera frame

Kalibr camchain дает матрицу:

```text
T_cam_imu
```

Текущий алгоритм использует из нее только поворот:

```text
R_cam_imu = T_cam_imu[0:3, 0:3]
```

Векторы IMU переводятся в camera frame:

```text
a_cam     = R_cam_imu * a_imu
omega_cam = R_cam_imu * omega_imu
```

Translation-компонента:

```text
T_cam_imu[0:3, 3]
```

загружается в конфиг как `imu_camera_translation_m`, но в текущей 2D-оценке позы не используется.

### 5.5 Состояние траектории

Каждая точка траектории имеет вид:

```text
TrajectoryPoint(t, x, y, yaw)
```

Где:

- `t` - время кадра в секундах;
- `x, y` - координаты в 2D-плоскости;
- `yaw` - ориентация робота в радианах;
- в режиме `generic_vio` масштаб относительный;
- в режиме `tape_line` масштаб задается параметром `forward_speed_mps`.

## 6. Общая схема пайплайна

```text
afn-run
  read YAML config
  apply CLI overrides
  merge IMU calibration YAML
  merge camera-IMU camchain YAML
  configure or auto-detect blue line HSV
  load frame timestamps
  load IMU CSV
  normalize video and IMU time bases
  apply Kalibr timeshift
  calibrate IMU samples
  estimate trajectory
  optionally run loop averaging
  save CSV and HTML outputs
```

Активный проектный режим:

```yaml
trajectory_mode: auto
target_color: blue
auto_video_config: true
loop_average: true
```

## 7. Сбор и объединение конфигурации

### 7.1 Базовый YAML

CLI сначала читает:

```text
configs/right_camera.yaml
```

Ключевые проектные параметры:

```yaml
trajectory_mode: auto
frame_timestamps: ../data/timestamps2.csv
frame_timestamp_time_scale: 1.0e-9
sync_time_base: auto

imu_calibration: ../data/imu-imu_calibration.yaml
camchain_calibration: ../data/camchain-imucam-imu_calibration.yaml
imu_time_scale: 1.0e-9
imu_gyro_scale: 0.017453292519943295

target_color: blue
hsv_ranges:
  - [95, 100, 60, 130, 255, 255]
```

Относительные пути из YAML резолвятся относительно папки самого конфига.

### 7.2 CLI overrides

Аргументы командной строки перекрывают значения из YAML.

| CLI-флаг | Ключ конфига |
| --- | --- |
| `--mode` | `trajectory_mode` |
| `--color` | `target_color` |
| `--auto-color` | `auto_color_tune` |
| `--frame-timestamps` | `frame_timestamps` |
| `--frame-time-scale` | `frame_timestamp_time_scale` |
| `--imu-time-scale` | `imu_time_scale` |
| `--imu-gyro-scale` | `imu_gyro_scale` |
| `--imu-calibration` | `imu_calibration` |
| `--camchain` | `camchain_calibration` |
| `--imu-use-translation` | `imu_use_translation` |

### 7.3 IMU calibration YAML

`load_imu_calibration_config()` переносит поля Kalibr в внутренние ключи:

| Поле Kalibr | Внутренний ключ |
| --- | --- |
| `accelerometer_noise_density` | `imu_accel_noise_density` |
| `gyroscope_noise_density` | `imu_gyro_noise_density` |
| `accelerometer_random_walk` | `imu_accel_random_walk` |
| `gyroscope_random_walk` | `imu_gyro_random_walk` |
| `update_rate` | `imu_update_rate_hz` |
| `time_offset` | `imu_time_offset_sec` |

Эти значения используются при построении параметров GTSAM preintegration, если GTSAM установлен.

### 7.4 Camera-IMU camchain YAML

`load_camchain_calibration_config()` читает первую секцию `cam*` и переносит:

| Поле camchain | Внутренний ключ |
| --- | --- |
| `T_cam_imu` | `cam_T_imu`, `imu_camera_rotation`, `imu_camera_translation_m` |
| `timeshift_cam_imu` | `imu_timeshift_cam_imu_sec` |
| `intrinsics` | `camera_intrinsics` |
| `distortion_coeffs` | `camera_distortion_coeffs` |
| `camera_model` | `camera_model` |
| `distortion_model` | `camera_distortion_model` |
| `resolution` | `camera_resolution` |
| `rostopic` | `camera_rostopic` |

Если camchain загружен, CLI включает дефолты:

```yaml
imu_apply_camchain_rotation: true
imu_apply_camchain_timeshift: true
imu_align_gravity: true
imu_yaw_axis: z
imu_yaw_only: true
imu_yaw_bias_window_sec: 1.0
```

## 8. Конфигурация синей линии

### 8.1 Статическая настройка

В текущем видео линия синяя:

```yaml
target_color: blue
hsv_ranges:
  - [95, 100, 60, 130, 255, 255]
```

Формат HSV-диапазона:

```text
[h_low, s_low, v_low, h_high, s_high, v_high]
```

### 8.2 Автонастройка по видео

Если включено:

```yaml
auto_video_config: true
```

то `apply_auto_video_config()` сэмплирует кадры видео и проверяет поддержанные цветовые пресеты:

```text
red, blue, green, yellow, white
```

Для каждого цвета:

1. Берется preset HSV.
2. На сэмплированных кадрах запускается `LineDetector.process()`.
3. Для результата считается score по площади маски, длине centerline, вертикальному span, наличию линии ближе к низу кадра и валидности угла.
4. Считается `mean_score` и `valid_ratio`.
5. Лучший цвет принимается, если проходит thresholds.
6. По найденным пикселям линии пересобираются HSV-диапазоны.

Для текущей записи ожидаемый результат - `blue`.

## 9. Загрузка и подготовка IMU

### 9.1 Чтение IMU CSV

`load_imu_csv()` ищет колонки по набору допустимых имен.

| Данные | Возможные имена колонок |
| --- | --- |
| timestamp | `t`, `time`, `timestamp`, `sec`, `seconds`, `stamp` |
| accel x/y/z | `ax`, `ay`, `az`, `accel_x`, `linear_acceleration_x` и аналоги |
| gyro x/y/z | `gx`, `gy`, `gz`, `gyro_x`, `angular_velocity_x` и аналоги |

После чтения:

1. Время умножается на `imu_time_scale`.
2. Гироскоп умножается на `imu_gyro_scale`.
3. Сэмплы сортируются по времени.
4. Возвращается `List[IMUSample]`.

### 9.2 Порядок калибровки IMU samples

`calibrate_imu_samples()` применяет операции строго в таком порядке:

1. Remap осей акселерометра через `imu_accel_axes`, если задан.
2. Remap осей гироскопа через `imu_gyro_axes`, если задан.
3. Вычитание постоянного gyro bias из `imu_gyro_bias`, если задан.
4. Поворот accel/gyro из IMU frame в camera frame через `imu_camera_rotation`.
5. Gravity alignment, если `imu_align_gravity: true`.
6. Выделение yaw-компоненты через `imu_yaw_axis`.
7. Вычитание yaw bias по начальному временному окну.
8. Зануление roll/pitch angular velocity при `imu_yaw_only: true`.

### 9.3 Gravity alignment

Если включено:

```yaml
imu_align_gravity: true
```

то средний вектор ускорения считается оценкой направления гравитации:

```text
g_est = mean(accel)
```

Затем строится поворот:

```text
R_align: g_est -> [0, 0, -1]
```

И применяется:

```text
accel = R_align * accel
omega = R_align * omega
```

Цель - уменьшить смешивание roll/pitch с yaw в 2D-задаче.

### 9.4 Yaw-only подготовка

Активная проектная конфигурация:

```yaml
imu_yaw_axis: z
imu_yaw_only: true
imu_yaw_bias_window_sec: 1.0
```

Алгоритм:

1. Выбирает ось `z`.
2. По первым `imu_yaw_bias_window_sec` секундам оценивает bias:

```text
yaw_bias = mean(omega_z in first window)
```

3. Считает yaw rate:

```text
yaw_rate = axis_sign * imu_yaw_gain * (omega_z - yaw_bias)
```

4. При `imu_yaw_only: true` записывает:

```text
omega_x = 0
omega_y = 0
omega_z = yaw_rate
```

## 10. Загрузка времени кадров

Если задан `frame_timestamps`, используется `load_frame_timestamps_csv()`.

Функция:

1. Читает CSV.
2. Ищет timestamp column.
3. Если есть frame index column, сортирует строки по индексу.
4. Умножает timestamp на `frame_timestamp_time_scale`.
5. Возвращает список времени кадров.

Для кадра `frame_i` время берется так:

```text
frame_timestamps[frame_i - 1]
```

Если CSV не задан, используется OpenCV:

```text
cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
```

Проектный путь должен использовать `timestamps2.csv`, потому что он точнее предположения о постоянном FPS.

## 11. IMU preintegration

На каждом интервале между соседними видеокадрами:

```text
t0 = previous_frame_time
t1 = current_frame_time
seg = slice_imu(samples, t0, t1)
result = preintegrate(seg)
```

`PreintegrationResult` содержит:

```text
delta_t
delta_R
delta_v
delta_p
covariance
delta_yaw
```

### 11.1 Ветка GTSAM

Если GTSAM установлен, используется `gtsam.PreintegratedImuMeasurements`.

Параметры строятся через `build_default_params()`:

```text
accelerometer covariance = accel_noise_sigma^2 * I
gyroscope covariance     = gyro_noise_sigma^2 * I
bias random walks        = accel_bias_rw_sigma, gyro_bias_rw_sigma
gravity                  = imu_gravity_mps2 or 9.81
```

Шумы берутся из Kalibr IMU YAML, если они были загружены.

### 11.2 Fallback без GTSAM

Если GTSAM недоступен, используется внутренний минимальный интегратор:

```text
dR = exp_so3(omega * dt)
p  = p + v * dt + 0.5 * (R * accel) * dt^2
v  = v + (R * accel) * dt
R  = R * dR
```

В fallback-режиме covariance возвращается нулевой. Для текущей задачи это допустимо, потому что основной стабильно используемый результат preintegration - `delta_yaw`.

## 12. Выбор режима построения траектории

Главная функция:

```text
estimate_trajectory_with_details()
```

принимает:

- путь к видео;
- подготовленные IMU samples;
- объединенный config;
- optional debug-video path;
- optional frame timestamps.

Правила выбора режима:

| `trajectory_mode` | Поведение |
| --- | --- |
| `generic_vio` | Всегда строить траекторию по optical flow и IMU yaw. |
| `tape_line` | Всегда строить траекторию по синей линии и IMU yaw. |
| `auto` при наличии IMU | Сначала попробовать `generic_vio`; если результат плохой, перейти в `tape_line`. |
| `auto` без IMU | Использовать `tape_line`. |

`generic_vio` считается пригодным, если:

1. Он вернул достаточно точек.
2. Spatial span траектории ненулевой.
3. Отношение `total_path_length / spatial_span` не превышает `auto_generic_max_length_span_ratio`.

## 13. Режим `generic_vio`

### 13.1 Назначение режима

`generic_vio` строит относительную 2D-траекторию по визуальному движению feature points и yaw из IMU.

Этот режим не дает строгий метрический масштаб. По одному monocular-видео без depth/stereo/внешней метрической привязки нельзя надежно восстановить абсолютный масштаб перемещения.

### 13.2 Подготовка кадра

Для каждого кадра:

1. Кадр уменьшается до `vio_resize_width`.
2. Кадр переводится в grayscale.
3. Применяется Gaussian blur.
4. Определяется timestamp кадра.
5. При необходимости сохраняется compact descriptor кадра для последующей оценки loop period.

### 13.3 Детекция feature points

На первом кадре или при недостатке tracked points вызывается:

```text
cv2.goodFeaturesToTrack()
```

Область около границ изображения исключается через `vio_border_margin_fraction`.

### 13.4 Optical flow tracking

Между соседними кадрами точки трекаются через Lucas-Kanade:

```text
cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts)
```

Затем применяется forward-backward check:

```text
prev -> next -> prev_back
keep if norm(prev_back - prev) <= vio_forward_backward_max_error
```

Это отбрасывает нестабильные треки.

### 13.5 Оценка визуального transform

По оставшимся точкам оценивается partial affine transform:

```text
M = cv2.estimateAffinePartial2D(prev_pts, next_pts, RANSAC)
```

Если inliers слишком мало, визуальный motion считается недоступным на этом шаге.

Матрица переводится в нормализованные координаты изображения:

```text
H_norm = normalize * H * denormalize
```

Линейная часть проецируется на чистое 2D-вращение через SVD:

```text
linear = H_norm[0:2, 0:2]
U, S, Vt = svd(linear)
R_2d = U * Vt
```

Translation в normalized coordinates ограничивается параметром `vio_max_step_norm`.

### 13.6 Слияние visual yaw и IMU yaw

На том же интервале кадров берется:

```text
dyaw_imu = preintegrate_imu(last_t, t).delta_yaw
```

Visual yaw:

```text
visual_yaw = atan2(H_norm[1, 0], H_norm[0, 0])
```

Если `dyaw_imu` конечный, углы смешиваются:

```text
adaptive_weight = clamp(vio_imu_rotation_weight + 0.2 * (0.6 - inlier_ratio), 0, 1)
fused_yaw = blend_angles(visual_yaw, dyaw_imu, adaptive_weight)
```

Когда visual tracking слабее, вес IMU становится выше.

### 13.7 Обновление pose

Шаг движения:

```text
step.R = rotation(fused_yaw)
step.t = H_norm[0:2, 2]
```

Глобальная pose обновляется так:

```text
pose = pose * inverse(step)
```

В траекторию добавляется:

```text
t   = current_frame_time
x   = pose[0, 2]
y   = pose[1, 2]
yaw = atan2(pose[1, 0], pose[0, 0])
```

### 13.8 Псевдокод `generic_vio`

```text
pose = identity_2d_transform
prev_gray = None
prev_pts = None

for frame_i, frame in video:
    gray = preprocess(frame)
    t = frame_time(frame_i)

    if first_frame:
        prev_gray = gray
        prev_pts = detect_features(gray)
        append TrajectoryPoint(t, 0, 0, 0)
        continue

    if prev_pts is missing or too small:
        prev_pts = detect_features(prev_gray)

    tracked_prev, tracked_next = optical_flow_with_fb_check(prev_gray, gray, prev_pts)
    H_norm, inliers, inlier_ratio = estimate_normalized_transform(tracked_prev, tracked_next)

    dyaw_imu = preintegrate_imu(last_t, t).delta_yaw

    if H_norm exists:
        visual_yaw = yaw_from_transform(H_norm)
        fused_yaw = blend(visual_yaw, dyaw_imu)
        step = transform(fused_yaw, H_norm.translation)
    else:
        step = transform(dyaw_imu, zero_translation)

    pose = pose * inverse(step)
    append trajectory point from pose

    prev_gray = gray
    prev_pts = current inlier points or redetected points
    last_t = t
```

## 14. Режим `tape_line`

### 14.1 Назначение режима

`tape_line` строит 2D-траекторию по видимой синей линии. IMU дает yaw delta, а видео дает направление линии и боковое смещение относительно линии.

Масштаб движения задается параметром:

```yaml
forward_speed_mps: 0.25
```

Если фактическая скорость робота отличается от этого значения, масштаб траектории также будет отличаться.

### 14.2 Сегментация синей линии

`LineDetector.process()` выполняет:

1. Resize кадра до `resize_width`.
2. Перевод BGR в HSV.
3. Выбор `target_color`, для текущего проекта - `blue`.
4. Выбор HSV ranges из явного конфига или blue preset.
5. Построение бинарной маски через `cv2.inRange()`.
6. Median blur.
7. Выделение largest connected component.
8. Удаление верхней половины изображения и нижних 10 процентов.
9. Morphological close для закрытия разрывов.

### 14.3 Построение centerline

Centerline строится через distance transform:

```text
dist = cv2.distanceTransform(mask)
for each row in ROI:
    center_x = argmax(dist[row])
```

Полученные точки сглаживаются moving average окном.

### 14.4 Наблюдение линии

Line detector возвращает:

```text
TapeObservation(
    centerline_px,
    angle_rad,
    bottom_x,
    shape_hw,
    mask,
    centerline_mask
)
```

Где:

- `centerline_px` - пиксельная центральная линия;
- `angle_rad` - оценка угла линии в изображении;
- `bottom_x` - горизонтальная позиция линии ближе к низу кадра;
- `shape_hw` - размер обработанного кадра;
- `mask` - бинарная маска синей линии;
- `centerline_mask` - маска centerline.

### 14.5 Сглаживание наблюдений

`VisionObservationSmoother` применяет exponential smoothing к:

- `angle_rad`;
- `bottom_x`.

Это уменьшает jitter между соседними кадрами.

### 14.6 Обновление состояния

Состояние:

```text
x, y, yaw
```

Для каждого интервала кадров:

1. Считается:

```text
dt = current_frame_time - previous_frame_time
```

2. Интегрируется IMU yaw:

```text
yaw = yaw + delta_yaw_imu
```

3. Если включено `imu_use_translation: true`, используется `delta_p` из IMU preintegration. В проектном режиме это выключено из-за drift.

4. Если IMU translation не используется, робот продвигается вперед:

```text
dist = forward_speed_mps * dt
x = x + dist * cos(yaw)
y = y + dist * sin(yaw)
```

5. Угол линии дает yaw correction:

```text
yaw_cmd = vision_yaw_gain * angle_rad
yaw_cmd += vision_yaw_nonlinear_gain * angle_rad * abs(angle_rad)
corr = clamp(yaw_cmd, -vision_yaw_max_correction, vision_yaw_max_correction) * dt
yaw = yaw + corr
```

6. Смещение линии от центра кадра дает lateral correction:

```text
err_px = bottom_x - image_center_x
dy_body = -vision_lateral_gain * err_px * dt
x = x - dy_body * sin(yaw)
y = y + dy_body * cos(yaw)
```

7. В траекторию добавляется `TrajectoryPoint(t, x, y, yaw)`.

### 14.7 Псевдокод `tape_line`

```text
x, y, yaw = 0, 0, 0
last_t = None

for frame_i, frame in video:
    t = frame_time(frame_i)
    obs = smooth(line_detector.process(frame))

    if first_frame:
        append TrajectoryPoint(t, x, y, yaw)
        last_t = t
        continue

    dt = max(t - last_t, epsilon)
    imu_result = preintegrate_imu(last_t, t)
    yaw += imu_result.delta_yaw

    if imu_use_translation:
        x, y = integrate_delta_p(imu_result.delta_p, yaw)
    else:
        x += forward_speed_mps * dt * cos(yaw)
        y += forward_speed_mps * dt * sin(yaw)

    if obs.angle_rad is finite:
        yaw += clipped_visual_yaw_correction(obs.angle_rad, dt)

    if obs.bottom_x is finite:
        x, y = apply_lateral_correction(x, y, yaw, obs.bottom_x, dt)

    append TrajectoryPoint(t, x, y, yaw)
    last_t = t
```

## 15. Loop averaging

### 15.1 Назначение

Если робот едет по повторяющейся петле, raw trajectory может накапливать drift. `loop_average` строит более стабильную каноническую петлю по нескольким кругам.

Активные параметры:

```yaml
loop_average: true
auto_loop_period: true
```

### 15.2 Оценка периода круга

Если `loop_period_sec` не задан, период оценивается по compact grayscale descriptors:

1. Кадр уменьшается до `32 x 24`.
2. Нормализуется mean/std.
3. Вектор L2-нормализуется.
4. Для разных lag считается similarity между descriptor sequences.
5. Выбирается lag с достаточным score и gain над baseline.
6. Lag переводится в секунды через median frame interval.

### 15.3 Нарезка на круги

Траектория режется на круги одним из способов:

1. Anchor-based boundaries - поиск повторяющегося anchor, например `bottom_left`.
2. Periodic boundaries - нарезка по известному или оцененному периоду.

Каждый круг ресэмплится по длине дуги до `loop_samples`.

### 15.4 Выравнивание кругов

Каждый круг выравнивается к шаблону:

- rigid transform;
- similarity transform, если `loop_similarity_align: true`.

Также выполняется ограниченный phase search, чтобы компенсировать несовпадение стартовой точки круга.

### 15.5 Отбрасывание выбросов

По alignment RMSE считается robust MAD-filter:

```text
robust_z = 0.67448975 * abs(rmse - median_rmse) / MAD
keep if robust_z <= loop_outlier_sigma
```

Алгоритм сохраняет минимум `min_keep` кругов.

### 15.6 Построение канонической петли

Финальная петля строится одним из способов:

1. Mean/median по выровненным кругам.
2. Fourier smoothing замкнутого пути.
3. Closed Catmull-Rom spline.
4. Или выбор representative lap, если он дает более качественную петлю.

Результат становится `final_traj`. Исходный путь сохраняется как `raw_traj`.

## 16. Ошибки и fallback-поведение

| Ситуация | Поведение |
| --- | --- |
| Видео не открывается | Runtime error. |
| В IMU CSV нет нужных колонок | Value error. |
| В timestamps CSV нет timestamp column | Value error. |
| GTSAM не установлен | Используется внутренний minimal preintegration. |
| `generic_vio` упал или не прошел quality check в `auto` | Используется fallback `tape_line`. |
| Недостаточно данных для loop averaging | `final_traj = raw_traj`. |
| Auto color detection не прошел thresholds | Остаются значения цвета и HSV из конфига. |

## 17. Ограничения текущего алгоритма

1. `generic_vio` работает в monocular relative scale.
2. Camera intrinsics и distortion из camchain читаются, но не используются для undistort или bundle adjustment.
3. Translation из `T_cam_imu` читается, но не используется в 2D-позе.
4. IMU translation по умолчанию выключена, потому что двойное интегрирование accel быстро уводит позицию.
5. `tape_line` зависит от видимости синей линии и качества HSV-сегментации.
6. В проекте нет factor graph optimization, bundle adjustment, SLAM loop closure и map reuse.
7. Loop averaging работает только для повторяющихся траекторий и не заменяет полноценный loop closure.

## 18. Проверочный чеклист

Перед сдачей изменений в алгоритме или конфиге нужно проверить:

1. `configs/right_camera.yaml` содержит `target_color: blue`.
2. Примеры запуска используют `--color blue`.
3. `uv run python -m compileall src` проходит без ошибок.
4. `afn-run` создает `right_trajectory.csv` и `right_trajectory.html`.
5. В HTML сама линия траектории синяя.
6. При `--save-debug` debug-видео показывает синюю line mask или стабильные feature tracks.
7. CLI выводит путь к frame timestamps и YAML-калибровкам.
8. Если `auto` перешел из `generic_vio` в `tape_line`, итоговый mode виден в CLI output.

## 19. Рекомендуемые инженерные улучшения

Для более строгой метрической траектории:

1. Использовать camera intrinsics и distortion coefficients для undistort кадров.
2. Использовать ZED stereo/depth для восстановления метрического масштаба.
3. Заменить lightweight `generic_vio` на ORB-SLAM3, OpenVINS или VINS-Fusion.
4. Оставить детектор синей линии как domain-specific fallback или дополнительное ограничение, а не как единственный источник навигации.
