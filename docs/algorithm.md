# Спецификация алгоритма построения траектории по видео и IMU

## 1. Назначение документа

Документ описывает текущий алгоритм проекта `awesome-feature-navigation` для построения 2D-траектории робота по:

- видео с камеры ZED, в основном по `Right_cam.mp4`;
- дополнительному видео `Left_cam.mp4` для двухкамерного усреднения той же поездки;
- timestamps видеокадров;
- IMU-измерениям;
- YAML-калибровкам камеры и IMU;
- видимой цветной линии на полу.

Это спецификация фактической реализации в репозитории. Она не описывает идеальный visual-inertial SLAM и не утверждает, что текущий код решает полную SLAM-задачу.

Для текущего видео линия, по которой едет робот, синяя. Поэтому проектная конфигурация использует `target_color: blue`, а явный CLI-флаг должен быть `--color blue`.

## 2. Модули реализации

| Зона ответственности | Модуль |
| --- | --- |
| Версия пакета и package metadata | `src/awesome_feature_navigation/__init__.py` |
| CLI, объединение конфига, запуск пайплайна | `src/awesome_feature_navigation/cli.py` |
| Загрузка frame timestamps и Kalibr YAML | `src/awesome_feature_navigation/calibration.py` |
| Загрузка IMU CSV, сдвиг времени, remap осей, поворот в camera frame | `src/awesome_feature_navigation/imu_io.py` |
| IMU preintegration, GTSAM-ветка и fallback-интегратор | `src/awesome_feature_navigation/imu_preintegration.py` |
| Детекция синей линии и построение centerline | `src/awesome_feature_navigation/line_detection.py` |
| Автонастройка цвета и HSV по сэмплам видео | `src/awesome_feature_navigation/auto_config.py` |
| Оценка траектории, выбор режима, loop averaging | `src/awesome_feature_navigation/trajectory.py` |
| Сохранение CSV и HTML-графиков | `src/awesome_feature_navigation/plotting.py` |
| Двухкамерное усреднение LEFT/RIGHT траекторий | `src/awesome_feature_navigation/fusion.py` |
| CLI для усреднения LEFT/RIGHT CSV | `src/awesome_feature_navigation/fusion_cli.py` |

### 2.1 `src/awesome_feature_navigation/__init__.py`

Пакетный файл хранит версию библиотеки:

```text
__version__ = "0.1.0"
```

Он нужен для того, чтобы пакет имел стабильную metadata-точку. В нем нет алгоритмической логики: импорт этого файла не должен открывать видео, читать конфиги, создавать output-файлы или запускать тяжелые зависимости.

### 2.2 `src/awesome_feature_navigation/cli.py`

`cli.py` - граница между пользователем и алгоритмом. Его задача не в том, чтобы оценивать траекторию самостоятельно, а в том, чтобы собрать корректное состояние запуска:

1. прочитать YAML-конфиг;
2. применить CLI overrides;
3. зарезолвить относительные пути относительно файла конфига;
4. добавить параметры из Kalibr YAML;
5. включить camera-IMU defaults после чтения camchain;
6. запустить auto-config линии;
7. загрузить timestamps и IMU;
8. привести video time и IMU time к одной шкале;
9. вызвать `estimate_trajectory_with_details()`;
10. сохранить все CSV/HTML/debug outputs.

Теоретически это слой orchestration. Он не должен содержать CV-математику и не должен напрямую знать, как устроены Procrustes, Lucas-Kanade или IMU preintegration. Это снижает coupling: можно менять алгоритм в `trajectory.py`, не переписывая CLI.

Ключевые helper-функции:

- `_load_cfg()` читает YAML и возвращает dict; пустой конфиг разрешен.
- `_resolve_path()` делает относительные пути устойчивыми: если путь из YAML существует рядом с YAML, используется он.
- `_parse_lap_bounds()` поддерживает секунды и формат `M:S`/`H:M:S`.
- `_normalize_time_base()` решает, нужно ли вычитать первый video timestamp из кадров и IMU.
- `_merge_calibration_cfg()` добавляет в основной dict значения из calibration loaders.

Главная тонкость `cli.py` - time base. Видео и IMU могут быть записаны в абсолютных наносекундах. Если их не привести к одной шкале, IMU-нарезка между кадрами будет пустой или сдвинутой, а yaw-интеграция станет бессмысленной.

### 2.3 `src/awesome_feature_navigation/calibration.py`

`calibration.py` отвечает за файлы, которые описывают измерительную систему, но не являются самой траекторией:

- frame timestamps CSV;
- Kalibr IMU YAML;
- Kalibr camera-IMU camchain YAML.

`load_frame_timestamps_csv()` ищет timestamp column, опционально сортирует строки по frame index и масштабирует время. Сортировка важна теоретически: последовательность кадров является временным рядом, поэтому перестановка строк в CSV не должна менять физический порядок кадров. Если timestamp не найден, функция падает явно, потому что молчаливый fallback на FPS дал бы скрытый сдвиг IMU.

`load_imu_calibration_config()` переносит параметры шума Kalibr в внутренние ключи. Эти параметры нужны для GTSAM preintegration covariance. Даже если сейчас основной рабочий режим использует fallback-интегратор, хранение этих параметров делает код расширяемым.

`load_camchain_calibration_config()` читает `T_cam_imu`, `timeshift_cam_imu`, intrinsics, distortion metadata и resolution. Из `T_cam_imu` используются:

```text
R_cam_imu = T_cam_imu[0:3, 0:3]
t_cam_imu = T_cam_imu[0:3, 3]
```

Поворот нужен уже сейчас для перевода accel/gyro в camera frame. Translation сохраняется как metadata: она потребуется, если добавлять полноценную stereo/VIO-модель с extrinsics.

### 2.4 `src/awesome_feature_navigation/imu_io.py`

`imu_io.py` превращает CSV и calibration config в поток `IMUSample(t, accel, omega)`, пригодный для интегрирования.

Основные операции:

- `_pick_col()` ищет колонки по точному имени и по нормализованному substring. Это поддерживает разные ROS/ZED/Kalibr naming styles.
- `_parse_axis_spec()` понимает `x`, `-x`, `+z` и возвращает индекс оси и знак.
- `_apply_axis_map()` переставляет и отражает оси accel/gyro.
- `_rotation_matrix_from_cfg()` принимает 3x3, 4x4, flat 9 или flat 16 matrix.
- `_rotation_from_a_to_b()` строит минимальный 3D-поворот между двумя векторами через cross product и Rodrigues-like формулу.
- `load_imu_csv()` читает CSV, масштабирует время и gyro, сортирует по `t`.
- `calibrate_imu_samples()` применяет axis remap, bias, camera rotation, gravity alignment и yaw-only reduction.
- `shift_imu_samples()` переносит IMU timestamps на другую временную шкалу.
- `slice_imu()` вырезает IMU-сегмент между двумя timestamp кадрами и добавляет соседние измерения по краям.

Теоретический смысл `slice_imu()` важен: интеграл по интервалу `[t0, t1]` не должен зависеть только от сэмплов строго внутри интервала. Если первый IMU-сэмпл после `t0`, то нужно добавить предыдущий сэмпл, иначе первый кусок интеграла теряется.

Gravity alignment используется как pragmatic 2D-приближение. Средний accel на старте/записи считается направлением гравитации, после чего строится поворот к `[0, 0, -1]`. Это не заменяет полноценную ориентационную оценку, но уменьшает смешивание roll/pitch с yaw для плоского движения.

### 2.5 `src/awesome_feature_navigation/imu_preintegration.py`

`imu_preintegration.py` реализует интегрирование IMU между двумя видеокадрами. Есть две ветки:

1. GTSAM, если пакет установлен.
2. Minimal fallback, если GTSAM недоступен.

`IMUSample` хранит один сэмпл:

```text
t
accel[3]
omega[3]
```

`PreintegrationResult` хранит интеграл на интервале:

```text
delta_t
delta_R
delta_v
delta_p
covariance
delta_yaw
```

GTSAM-ветка создает `PreintegrationParams`, задает covariance accel/gyro/bias, затем вызывает `integrateMeasurement()` для каждого положительного `dt`. Это математически ближе к стандартной visual-inertial preintegration: измерения IMU сворачиваются в один relative motion factor между двумя состояниями.

Fallback-ветка нужна, чтобы проект работал без тяжелой зависимости. Она использует экспоненту SO(3):

```text
R_next = R * exp(omega * dt)
v_next = v + R * accel * dt
p_next = p + v * dt + 0.5 * R * accel * dt^2
```

Для текущего `tape_line` основная ценность preintegration - `delta_yaw`. `delta_p` доступен, но по умолчанию не используется из-за квадратичного дрейфа double integration.

### 2.6 `src/awesome_feature_navigation/line_detection.py`

`line_detection.py` отвечает за извлечение геометрического наблюдения цветной линии из одного кадра.

`COLOR_PRESETS` задает базовые HSV-диапазоны для:

```text
red, blue, green, yellow, white
```

`resolve_target_color()` защищает от неправильного цвета и возвращает `blue` по умолчанию. `resolve_hsv_ranges()` выбирает явные `hsv_ranges`, legacy red keys или preset.

`LineDetector.process()` выполняет pipeline:

1. resize кадра;
2. BGR -> HSV;
3. optional HSV auto-tune внутри ROI;
4. binary mask через `cv2.inRange`;
5. median blur;
6. largest connected component;
7. удаление верхней половины и нижней полосы кадра;
8. morphological close;
9. distance transform;
10. centerline по argmax distance в каждой строке ROI;
11. smoothing centerline;
12. расчет угла и bottom-x.

Теория distance transform здесь простая: для каждой точки маски расстояние до ближайшего фона максимально около геометрической середины ленты. Поэтому `argmax` по строке дает центральную линию без явного skeletonization.

Угол линии может считаться глобально через PCA (`_fit_direction`) или локально по нижнему сегменту (`_fit_local_direction`). Локальный угол важен для управления роботом: нижняя часть кадра ближе к роботу, значит она лучше отражает ближайшее направление движения.

`bottom_x` - оценка горизонтального положения линии около нижней части ROI. Она используется как lateral error: если линия не по центру кадра, trajectory получает боковую correction.

### 2.7 `src/awesome_feature_navigation/auto_config.py`

`auto_config.py` уменьшает ручной подбор HSV под конкретное видео. Он не знает правильную траекторию и не использует координаты ответа. Он только выбирает цвет и HSV-диапазоны по качеству line detection на сэмплах кадров.

Алгоритм:

1. `_sample_frames()` берет равномерные кадры по frame count или stride.
2. Для каждого candidate color запускается `LineDetector`.
3. `_observation_score()` оценивает наблюдение по длине centerline, vertical span, bottom hit, площади mask и валидности угла.
4. `_score_color()` считает mean score и valid ratio.
5. `infer_line_config_from_video()` выбирает цвет с максимальным rank.
6. `_collect_line_hsv_pixels()` собирает HSV-пиксели внутри надежных line masks.
7. `_build_ranges_from_pixels()` пересобирает HSV ranges по percentiles.
8. `apply_auto_video_config()` вписывает найденный цвет/ranges в копию конфига.

Почему используются percentiles, а не min/max: отдельные пиксели могут быть бликами, шумом или краями ленты. Percentile-диапазон устойчивее к выбросам и лучше переносится между кадрами.

### 2.8 `src/awesome_feature_navigation/trajectory.py`

`trajectory.py` - основной алгоритмический модуль. Он содержит:

- dataclasses результата (`TrajectoryPoint`, `TrajectoryEstimateResult`, diagnostics);
- pure geometry helpers;
- extraction и canonicalization кругов;
- offline smoothing для line observations;
- `tape_line` estimator;
- `generic_vio` estimator;
- auto-dispatch между режимами;
- output transforms.

Геометрическая часть использует:

- rigid Procrustes alignment через SVD;
- optional similarity alignment со scale;
- resampling polyline по длине дуги;
- robust MAD-filter;
- Fourier smoothing closed path;
- closed Catmull-Rom spline;
- projection точек на замкнутый путь;
- self-intersection checks.

`tape_line` часть берет `TapeObservation` из `line_detection.py`, считает confidence, сглаживает angle/bottom-x по времени, интегрирует yaw из IMU и движение вперед через `forward_speed_mps`, затем применяет lateral correction. Если найден надежный период круга, включаются variable speed correction, soft loop closure и representative-lap final.

`generic_vio` часть использует feature tracking:

```text
goodFeaturesToTrack -> Lucas-Kanade optical flow -> forward-backward check -> estimateAffinePartial2D
```

Из affine transform извлекаются относительные yaw и translation в normalized image coordinates. Yaw смешивается с IMU yaw. Масштаб остается относительным, потому что monocular-видео без depth/stereo не задает абсолютную длину перемещения.

`estimate_trajectory_with_details()` - главная функция модуля. Она выбирает режим, запускает estimator, проверяет пригодность результата и возвращает полный `TrajectoryEstimateResult`, включая raw/smoothed/final trajectory и diagnostics.

### 2.9 `src/awesome_feature_navigation/plotting.py`

`plotting.py` не меняет алгоритм траектории. Он сериализует результаты:

- `save_trajectory_csv()` пишет `t,x,y,yaw`.
- `save_trajectory_plot()` строит интерактивный Plotly HTML или static image.
- `save_tape_diagnostics_csv/plot()` сохраняет confidence, angles, bottom-x, speed и IMU yaw deltas.
- `save_loop_debug_csv/plot()` сохраняет raw/aligned/projected laps, representative loop и spline controls.

Важная инженерная деталь: Plotly HTML export получает `toImageButtonOptions` с большим `width`, `height` и `scale`, чтобы скачиваемые PNG из браузера были качественными. Для static image используется `write_image`; если `kaleido` не установлен, пользователь получает понятное предупреждение.

### 2.10 `src/awesome_feature_navigation/fusion.py`

`fusion.py` реализует двухкамерное усреднение уже построенных 2D-траекторий. Он не открывает `Left_cam.mp4` и `Right_cam.mp4` напрямую: эти видео сначала независимо обрабатываются обычным `afn-run`, а `fusion.py` получает на вход два CSV с колонками `t,x,y,yaw`.

Основные публичные сущности:

- `load_trajectory_csv()` читает CSV, проверяет наличие колонок `t,x,y,yaw`, отбрасывает неявные ошибки через `ValueError` и возвращает список `TrajectoryPoint`.
- `fuse_trajectories()` принимает LEFT и RIGHT trajectory lists, ресэмплирует обе петли к одному числу точек, выравнивает LEFT к RIGHT и возвращает `TrajectoryFusionResult`.
- `TrajectoryFusionDiagnostics` хранит `sample_count`, `phase_shift`, `alignment_rmse`, `reverse_used`, `scale`.

Алгоритм `fuse_trajectories()`:

1. Проверяет, что обе траектории содержат минимум две точки.
2. Ресэмплирует LEFT и RIGHT как замкнутые polyline через длину дуги.
3. Берет RIGHT как reference frame.
4. Выравнивает LEFT к RIGHT через `_align_lap_to_template(..., allow_scale=False)`.
5. Если `allow_reverse=True`, отдельно пробует LEFT в обратном порядке.
6. Выбирает вариант с меньшим alignment RMSE.
7. Строит fused координаты:

```text
F_i = 0.5 * (R_i + L_aligned_i)
```

8. По fused координатам пересчитывает yaw из локальной производной пути.
9. Возвращает fused trajectory, aligned LEFT trajectory, resampled RIGHT trajectory и diagnostics.

Почему RIGHT используется как reference: это не означает, что RIGHT считается "правильным ответом". Reference задает только систему координат, стартовую фазу и направление вывода. После rigid alignment вклад в координаты идет симметрично: `0.5 * RIGHT + 0.5 * LEFT_aligned`.

Почему `allow_scale=False`: обе траектории должны быть в одной метрической шкале, заданной `forward_speed_mps` и одинаковым loop extraction. Автоматическая scale-подгонка могла бы скрыть ошибку скорости или неверное выделение круга.

### 2.11 `src/awesome_feature_navigation/fusion_cli.py`

`fusion_cli.py` - пользовательская обертка над `fusion.py`. Команда зарегистрирована в `pyproject.toml` как:

```text
afn-fuse = "awesome_feature_navigation.fusion_cli:main"
```

Она принимает:

- `--left` - CSV траектории, построенной по `Left_cam.mp4`;
- `--right` - CSV траектории, построенной по `Right_cam.mp4`;
- `--out` - output prefix;
- `--samples` - число точек для общего ресэмплинга;
- `--phase-search-fraction` - ширина поиска фазового сдвига;
- `--no-reverse-search` - отключение проверки обратного направления LEFT;
- `--save-aligned-debug` - сохранение aligned LEFT и resampled RIGHT для диагностики.

Основные outputs:

```text
outputs/fused_trajectory.csv
outputs/fused_trajectory.html
outputs/fused_trajectory_left_aligned.csv/html
outputs/fused_trajectory_right_resampled.csv/html
```

Debug outputs нужны, чтобы визуально проверить, что LEFT действительно легла на RIGHT перед усреднением. Если `alignment_rmse` высокий, усредненную траекторию нельзя считать надежной: проблема может быть в другом направлении обхода, плохом круге, разных конфигах или неверном входном CSV.

## 3. Входные данные

### 3.1 Основные входы запуска

| Вход | Проектный путь | Назначение |
| --- | --- | --- |
| Видео | `data/Right_cam.mp4` | Видеопоток с правой камеры. |
| Видео левой камеры | `data/Left_cam.mp4` | Видеопоток с левой камеры той же поездки; используется для независимой оценки и последующего усреднения с RIGHT. |
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
| `outputs/right_trajectory_raw.csv/html` | При `--save-loop-debug` | Raw trajectory без offline-сглаживания наблюдений. |
| `outputs/right_trajectory_smoothed.csv/html` | При `--save-loop-debug` | Trajectory после confidence-weighted offline smoothing и variable-speed correction. |
| `outputs/right_trajectory_diagnostics.csv/html` | При `--save-loop-debug` для `tape_line` | Confidence, raw/smoothed angle, estimated speed и IMU yaw delta по времени. |
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
  optionally run soft loop closure
  build closed representative lap if loop period is reliable
  save CSV and HTML outputs
```

Активный проектный режим:

```yaml
trajectory_mode: auto
auto_prefer_tape_line: true
target_color: blue
auto_video_config: true
offline_tape_smoothing: true
offline_adaptive_confidence: true
offline_variable_speed: true
offline_soft_loop_closure: true
loop_average: true
loop_strategy: representative_lap
loop_similarity_align: false
loop_fourier_harmonics: 12
loop_min_kept_laps: 2
loop_max_alignment_rmse_ratio: 0.80
loop_max_projection_rmse_ratio: 0.70
manual_lap_bounds_sec: [11.0, 53.0, 104.0, 159.48]
manual_lap_directions: [forward, forward, reverse]
loop_average_direction: any
loop_normalize_reverse_laps: true
loop_start_anchor: manual_start
trajectory_output_flip_x: true
max_frames: 0
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
auto_prefer_tape_line: true
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
offline_tape_smoothing: true
offline_line_smoothing_sec: 0.75
offline_adaptive_confidence: true
offline_line_min_confidence: 0.15
offline_variable_speed: true
offline_soft_loop_closure: true
loop_average: true
loop_strategy: representative_lap
loop_similarity_align: false
loop_fourier_harmonics: 12
loop_min_kept_laps: 2
loop_max_alignment_rmse_ratio: 0.80
loop_max_projection_rmse_ratio: 0.70
manual_lap_bounds_sec: [11.0, 53.0, 104.0, 159.48]
manual_lap_directions: [forward, forward, reverse]
loop_average_direction: any
loop_normalize_reverse_laps: true
loop_start_anchor: manual_start
trajectory_output_flip_x: true
line_angle_mode: bottom_segment
max_frames: 0
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
| `auto` при наличии IMU и `auto_prefer_tape_line: true` | Сначала попробовать `tape_line`; если линия недостаточно надежна, перейти в `generic_vio`. |
| `auto` при наличии IMU и `auto_prefer_tape_line: false` | Сначала попробовать `generic_vio`; если результат плохой, перейти в `tape_line`. |
| `auto` без IMU | Использовать `tape_line`. |

`tape_line` считается пригодным, если:

1. Он вернул достаточно точек.
2. Доля кадров с надежным наблюдением линии не ниже `auto_tape_min_valid_ratio`.
3. Spatial span траектории ненулевой.

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

### 14.5 Офлайн-сглаживание наблюдений

Так как задача офлайн, `tape_line` не обязан принимать решение сразу на текущем кадре. Режим сначала собирает по всей записи `TapeFrameObservation`:

```text
t
dt
TapeObservation
line confidence
IMU delta_yaw
IMU delta_p
```

Для каждого наблюдения считается confidence по:

- количеству точек centerline;
- вертикальному span линии;
- попаданию линии в нижнюю часть рабочей области;
- площади mask;
- валидности `angle_rad`.

Затем `angle_rad` и `bottom_x` сглаживаются симметричным centered weighted average по всей временной последовательности:

- `angle_rad`;
- `bottom_x`.

Вес кадра равен confidence. Порог confidence может быть адаптивным:

```yaml
offline_adaptive_confidence: true
offline_line_min_confidence: 0.15
```

`offline_line_min_confidence` в этом режиме является нижней границей, а фактический threshold поднимается по распределению confidence на всем видео. Это уменьшает ручной подбор под конкретную запись: хорошее видео получает более строгий отбор, плохое не теряет все кадры.

Размер окна сглаживания задается временем, а не числом кадров:

```yaml
offline_line_smoothing_sec: 0.75
```

Реальное число кадров вычисляется по median frame interval. Поэтому один и тот же конфиг одинаково интерпретируется на видео с разным FPS. Это не использует будущие данные в онлайн-смысле, потому что весь pipeline офлайн и видео уже полностью доступно.

### 14.6 Обновление состояния

Состояние:

```text
x, y, yaw
```

После офлайн-сглаживания для каждого интервала кадров:

1. Считается:

```text
dt = current_frame_time - previous_frame_time
```

2. Интегрируется IMU yaw:

```text
yaw = yaw + delta_yaw_imu
```

3. Если включено `imu_use_translation: true`, используется `delta_p` из IMU preintegration. В проектном режиме это выключено из-за drift.

4. Если IMU translation не используется, робот продвигается вперед. Базовый масштаб задает `forward_speed_mps`, но в offline-режиме может быть включена переменная скорость:

```text
dist = forward_speed_mps * speed_scale[i] * dt
x = x + dist * cos(yaw)
y = y + dist * sin(yaw)
```

`speed_scale[i]` оценивается по всей записи как гладкая последовательность, близкая к `1.0`. Если найден надежный период петли, speed scales дополнительно подбираются из least-squares условия мягкого замыкания кругов. Если периода нет или замыкание выглядит ненадежным, `speed_scale[i] = 1.0`.

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

Скорость при этом не вычисляется из воздуха. `forward_speed_mps` остается метрическим масштабом. Новая variable-speed часть меняет относительное распределение скорости по времени, но не создает абсолютный масштаб без внешнего источника.

### 14.7 Псевдокод `tape_line`

```text
records = []
last_t = None

for frame_i, frame in video:
    t = frame_time(frame_i)
    obs = line_detector.process(frame)
    confidence = line_observation_confidence(obs)

    if first_frame:
        records.append(TapeFrameObservation(t, 0, obs, confidence, 0, zero_delta_p))
        last_t = t
        continue

    dt = max(t - last_t, epsilon)
    imu_result = preintegrate_imu(last_t, t)
    records.append(TapeFrameObservation(t, dt, obs, confidence, imu_result.delta_yaw, imu_result.delta_p))
    last_t = t

threshold = adaptive_confidence_threshold(records.confidence)
weights = records.confidence where confidence >= threshold else 0
window_frames = smoothing_seconds_to_frames(offline_line_smoothing_sec, frame_times)

smooth_angle = centered_weighted_angle_average(records.angle_rad, weights, window_frames)
smooth_bottom_x = centered_weighted_average(records.bottom_x, weights, window_frames)

x, y, yaw = 0, 0, 0
append TrajectoryPoint(records[0].t, x, y, yaw)

for i in range(1, len(records)):
    yaw_pred = yaw + records[i].delta_yaw
    yaw_next = yaw_pred + visual_yaw_correction(smooth_angle[i], records[i].confidence)
    yaw_move = midpoint_angle(yaw, yaw_next)

    if imu_use_translation:
        x, y = integrate_delta_p(records[i].delta_p, yaw_move)
    else:
        x += forward_speed_mps * speed_scale[i] * records[i].dt * cos(yaw_move)
        y += forward_speed_mps * speed_scale[i] * records[i].dt * sin(yaw_move)

    x, y = apply_lateral_correction(x, y, yaw_next, smooth_bottom_x[i], records[i].confidence)
    yaw = yaw_next
    append TrajectoryPoint(records[i].t, x, y, yaw)

raw_traj = integrate_with_raw_observations(records)
smoothed_traj = integrate_with_smoothed_observations(records, speed_scale)
final_traj = apply_soft_loop_closure(smoothed_traj)
```

## 15. Soft loop closure и representative-lap финал

### 15.1 Назначение

Если робот едет по повторяющейся петле, raw trajectory может накапливать drift. В проектном режиме строятся две разные сущности:

- `smoothed_traj` - полный многокруговой путь после offline smoothing и variable speed;
- `final_traj` - один замкнутый representative lap, если период круга найден надежно.

Перед построением representative lap может применяться мягкое loop closure к `smoothed_traj`:

- период оценивается по visual descriptors;
- находятся повторяющиеся boundaries круга;
- если ошибка замыкания мала относительно длины круга, она распределяется по кругу частично;
- если ошибка слишком большая, замыкание не применяется.

Активные параметры:

```yaml
auto_loop_period: true
offline_soft_loop_closure: true
loop_average: true
loop_strategy: representative_lap
loop_similarity_align: false
loop_fourier_harmonics: 12
```

`loop_strategy: representative_lap` означает, что финальная петля берется из лучшего наблюдаемого круга, а не как средняя synthetic-кривая. `loop_similarity_align: false` запрещает масштабное растяжение кругов при alignment. `loop_fourier_harmonics: 12` задает умеренное Fourier-сглаживание: оно убирает высокочастотный шум линии, но не превращает петлю в низкочастотный эллипс.

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

1. Manual boundaries - явные `manual_lap_bounds_sec` из конфига.
2. Anchor-based boundaries - поиск повторяющегося anchor, например `bottom_left`.
3. Periodic boundaries - нарезка по известному или оцененному периоду.

Каждый круг ресэмплится по длине дуги до `loop_samples`.

Для текущего `Right_cam.mp4` в конфиге явно задано, что первые два интервала являются прямыми кругами, а третий проезд после разворота является обратным:

```yaml
manual_lap_bounds_sec: [11.0, 53.0, 104.0, 159.48]
manual_lap_directions: [forward, forward, reverse]
loop_average_direction: any
loop_normalize_reverse_laps: true
loop_start_anchor: manual_start
```

Это означает, что `raw_traj` и `smoothed_traj` содержат весь проезд, включая обратное направление. Для `final_traj` reverse-круг разворачивается по порядку точек и участвует в representative lap вместе с forward-кругами, не ломая общее направление канонической петли.

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

### 15.6 Quality gate повторяемости

После построения candidate loop алгоритм дополнительно проверяет, можно ли считать движение повторяющимся:

```text
alignment_ratio  = median(kept alignment RMSE) / loop_span
projection_ratio = median(kept projection RMSE) / loop_span
```

Финальная замкнутая петля принимается только если:

```yaml
kept_laps >= loop_min_kept_laps
alignment_ratio <= loop_max_alignment_rmse_ratio
projection_ratio <= loop_max_projection_rmse_ratio
```

Если проверка не проходит, canonical loop не используется. Тогда `final_traj` остается текущей offline-траекторией, а не искусственно замкнутой петлей.

### 15.7 Построение финальной петли

Финальная петля может строиться одним из способов:

1. Mean/median по выровненным кругам.
2. Fourier smoothing замкнутого пути.
3. Closed Catmull-Rom spline.
4. Или выбор representative lap, если он дает более качественную петлю.

В обычном проектном режиме выбран `representative_lap`:

```text
raw_traj      = интеграция raw-наблюдений
smoothed_traj = offline smoothing + variable speed
final_traj    = closed representative lap from smoothed_traj
```

### 15.8 Усреднение LEFT и RIGHT траекторий

ZED дает два синхронных видеопотока: `Left_cam.mp4` и `Right_cam.mp4`. Это не две разные поездки, а одна и та же физическая траектория робота, наблюдаемая двумя оптическими центрами. В проекте это реализовано отдельной командой `afn-fuse`, которая усредняет уже построенные CSV-траектории:

1. Построить траекторию по `Left_cam.mp4`.
2. Построить траекторию по `Right_cam.mp4`.
3. Передать два CSV в `afn-fuse`.
4. Привести обе траектории к одному числу sample points.
5. Проверить прямое и обратное направление LEFT.
6. Выровнять phase, потому что стартовая точка canonical loop может немного отличаться.
7. Выровнять LEFT к RIGHT через rigid Procrustes transform.
8. Усреднить соответствующие точки.
9. Сохранить fused trajectory как итоговую двухкамерную оценку.

Команды:

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

Усреднять нужно не кадры и не пиксельные наблюдения, а уже построенные 2D-траектории. Причина: левая и правая камеры имеют разные optical centers, поэтому одна и та же линия на полу проецируется в разные пиксельные координаты. Даже если робот едет по одной линии, `bottom_x`, локальный угол centerline и optical flow в LEFT и RIGHT не обязаны совпадать покадрово. После построения 2D-траектории эти различия становятся ошибками оценки в общей плоскости движения, и их уже можно уменьшать геометрическим выравниванием.

Базовая математическая модель fusion:

```text
R = right representative loop, shape (N, 2)
L = left representative loop,  shape (N, 2)

L_aligned = rigid_align_with_phase_search(L, R)
F = 0.5 * (R + L_aligned)
```

В реализации RIGHT задает reference frame для вывода, а не "правильный ответ". LEFT поворачивается и переносится в систему RIGHT, после чего координаты усредняются с одинаковыми весами.

Если одна камера дает более надежное наблюдение, вместо обычного среднего можно использовать weighted average:

```text
F_i = (w_R_i * R_i + w_L_i * L_aligned_i) / (w_R_i + w_L_i)
```

Где веса могут зависеть от diagnostics: valid ratio, confidence, alignment RMSE, projection RMSE или доли кадров, в которых линия была надежно видна.

Почему нужен rigid Procrustes, а не простое среднее координат:

- каждая монокулярная оценка может иметь небольшой поворот глобальной системы координат;
- стартовая точка canonical loop может быть сдвинута по фазе;
- output transform может отличаться между LEFT и RIGHT;
- reverse-lap normalization должен привести обход к одному направлению;
- без выравнивания среднее двух одинаковых окружностей со сдвигом фазы может искусственно уменьшить петлю или исказить форму.

Почему scale лучше не подгонять автоматически:

- в текущем `tape_line` метрический масштаб задается `forward_speed_mps`;
- если разрешить similarity scale без ограничения, можно скрыть ошибку скорости или loop extraction;
- поэтому базовый вариант fusion должен использовать rigid transform без масштаба, а scale менять только явно и обоснованно.

Ожидаемый практический эффект:

- случайные ошибки HSV/centerline одной камеры частично компенсируются другой;
- локальные ошибки видимости линии меньше влияют на итог;
- representative loop становится стабильнее;
- residual LEFT -> RIGHT после alignment становится диагностикой качества двухкамерной оценки.

## 16. Ошибки и fallback-поведение

| Ситуация | Поведение |
| --- | --- |
| Видео не открывается | Runtime error. |
| В IMU CSV нет нужных колонок | Value error. |
| В timestamps CSV нет timestamp column | Value error. |
| GTSAM не установлен | Используется внутренний minimal preintegration. |
| `tape_line` не прошел quality check в `auto_prefer_tape_line` | Используется fallback `generic_vio`. |
| `generic_vio` упал или не прошел quality check в `auto` | Используется fallback `tape_line`. |
| Недостаточно данных для soft loop closure | `smoothed_traj` не корректируется мягким замыканием. |
| Недостаточно данных для representative-lap финала | Замкнутая петля не строится, сохраняется текущий `final_traj`. |
| Auto color detection не прошел thresholds | Остаются значения цвета и HSV из конфига. |

## 17. Ограничения текущего алгоритма

1. `generic_vio` работает в monocular relative scale.
2. Camera intrinsics и distortion из camchain читаются, но не используются для undistort или bundle adjustment.
3. Translation из `T_cam_imu` читается, но не используется в 2D-позе.
4. IMU translation по умолчанию выключена, потому что двойное интегрирование accel быстро уводит позицию.
5. `tape_line` зависит от видимости синей линии и качества HSV-сегментации.
6. В проекте нет factor graph optimization, bundle adjustment, SLAM loop closure и map reuse.
7. Loop averaging работает только для повторяющихся траекторий и не заменяет полноценный loop closure.

## 18. Тестовая спецификация

Тесты являются частью спецификации поведения. Они не проверяют визуальное сходство с заранее нарисованной траекторией. Вместо этого они фиксируют инварианты: корректность парсинга данных, устойчивость к пустым входам, математические свойства helper-функций, отказ от плохих loop candidates и стабильность форматов output.

Текущий обязательный порог:

```text
pytest coverage >= 95%
```

Порог задан в `Makefile` и в GitHub Actions через `--cov-fail-under=95`. Badge в README строится из `badges/coverage.json`, который обновляет GitHub workflow после успешного push.

### 18.1 `tests/test_calibration.py`

Проверяет `calibration.py`.

Что покрывается:

- перенос Kalibr IMU полей `accelerometer_noise_density`, `gyroscope_noise_density`, random walk, update rate и time offset во внутренний config;
- чтение `T_cam_imu` из camchain, разделение матрицы на rotation и translation;
- сохранение `timeshift_cam_imu`, intrinsics, distortion, camera model, distortion model, resolution и rostopic;
- сортировка frame timestamps по frame index;
- масштабирование timestamps через `time_scale`;
- ошибки на CSV без timestamp column, пустом CSV и non-finite timestamp;
- поведение на YAML, где ожидаемый mapping отсутствует.

Теоретический смысл: timestamps и extrinsics являются основой синхронизации. Если timestamps не отсортированы или `timeshift_cam_imu` потерян, IMU будет интегрироваться на неправильных интервалах. Поэтому тесты проверяют не только happy path, но и отказ от невалидных входов.

### 18.2 `tests/test_imu_io.py` и `tests/test_imu_more.py`

Проверяют `imu_io.py` и `imu_preintegration.py`.

Что покрывается:

- поиск IMU колонок по разным naming styles;
- сортировка IMU samples по времени;
- `imu_time_scale` и `imu_gyro_scale`;
- ошибка при отсутствии timestamp/accel/gyro колонок;
- axis remap через `x`, `-x`, `+z`;
- ошибка на axis map длиной не 3;
- чтение rotation matrix из 3x3, 4x4, flat 9 и flat 16;
- поворот вектора `a -> b`, включая нулевые, одинаковые, противоположные и обычные направления;
- gravity alignment к `[0, 0, -1]`;
- yaw-only calibration, yaw bias window и non-yaw-only ветка;
- `shift_imu_samples()` без мутации accel/gyro;
- `slice_imu()` с добавлением соседних сэмплов у границ интегрирования;
- fallback SO(3) preintegration;
- накопление preintegration при `reset=False`;
- пропуск неположительного `dt`;
- fake-GTSAM ветка без установки настоящего `gtsam`;
- `MinimalRot3`, `_skew`, `_so3_exp`, `rot3_yaw`.

Теоретический смысл: IMU - шумный временной сигнал. Тесты фиксируют, что мы не теряем порядок времени, не путаем оси, не ломаем bias correction и корректно обрабатываем интервалы между кадрами. Fake-GTSAM тест нужен, чтобы проверить код стандартной preintegration-ветки в среде, где зависимость может быть не установлена.

### 18.3 `tests/test_line_detection.py` и `tests/test_line_detection_more.py`

Проверяют `line_detection.py`.

Что покрывается:

- fallback target color к `blue`;
- blue HSV preset;
- legacy red HSV ranges только при `target_color: red`;
- нормализация low/high HSV и clipping в допустимые границы;
- PCA/global angle;
- local bottom-segment angle;
- смешивание углов через unit vectors;
- degenerate cases: пустые точки, одна точка, нулевой вектор, противоположные углы;
- `bottom_x` через mean/median;
- сглаживание centerline и rendering single-point/polyline centerline;
- auto-tune HSV для white и цветных линий;
- обработка синтетической синей линии;
- fallback на предыдущую centerline при плохом следующем кадре.

Теоретический смысл: line detection должен быть устойчив к типичным CV-краевым случаям. Угол линии не должен превращаться в NaN там, где есть достаточная геометрия, а HSV-настройки не должны выходить за физические границы OpenCV HSV.

### 18.4 `tests/test_auto_config.py`

Проверяет `auto_config.py`.

Что покрывается:

- выбор сэмплов по `CAP_PROP_FRAME_COUNT`;
- fallback sampling по stride, если frame count неизвестен;
- ошибка на неоткрываемом видео;
- preset HSV ranges для поддержанных цветов;
- scoring наблюдения по маске, centerline, площади, bottom hit и валидности угла;
- penalty для слишком маленькой, слишком большой и слишком широкой mask;
- выбор лучшего candidate color по `mean_score + 1.5 * valid_ratio`;
- empty frames -> zero score;
- сбор HSV-пикселей только из надежных line masks;
- пропуск слабых наблюдений и shape mismatch;
- downsample больших masks;
- percentile-based HSV ranges для blue/green/yellow/red/white;
- special red split на две hue-группы;
- расширение слишком узкого hue диапазона;
- отключение `auto_video_config` и `auto_detect_line`;
- сохранение исходного конфига при неуспешной автонастройке.

Теоретический смысл: auto-config не должен подгонять траекторию. Он может смотреть только на изображение линии и качество segmentation. Поэтому тесты проверяют именно локальные свойства маски/HSV, а не форму финального пути.

### 18.5 `tests/test_offline_tape_line.py`

Проверяет offline-режим `tape_line` в `trajectory.py`.

Что покрывается:

- положительный confidence для чистой centerline;
- игнорирование low-confidence angle outlier;
- time-based smoothing window по timestamps, а не по жесткому числу кадров;
- adaptive confidence threshold по распределению confidence;
- разделение `raw_traj`, `smoothed_traj`, `final_traj`;
- diagnostics mask для валидных/невалидных кадров;
- отказ от loop canonicalization при плохо согласованных кругах;
- выбор только forward laps;
- включение reverse laps через `loop_normalize_reverse_laps`;
- распределение single closing jump по петле;
- output transform, например flip X.

Теоретический смысл: offline smoothing имеет доступ ко всей записи, поэтому может использовать centered weighted average и adaptive threshold. Но он обязан отбрасывать слабые кадры и не должен строить искусственную замкнутую петлю, если круги геометрически плохо повторяются.

### 18.6 `tests/test_trajectory_helpers.py`

Проверяет чистую математику `trajectory.py` без настоящего видео.

Что покрывается:

- `_clamp`, `_cfg_float`, `_cfg_bool_any`;
- IMU preintegration params builder;
- angle blending с NaN и противоположными направлениями;
- frame time fallback;
- rigid Procrustes alignment и reflection correction;
- similarity alignment и scale limits;
- RMSE;
- anchor scoring и anchor choice;
- robust MAD keep mask;
- open/closed polyline resampling по длине дуги;
- manual/periodic lap extraction;
- нормализация lap directions;
- reverse-lap normalization;
- Fourier smoothing;
- canonical loop build;
- representative lap selection;
- projection points to closed path;
- conversion loop -> trajectory;
- segment intersection и self-intersection count;
- confidence, smoothing, adaptive threshold;
- tape motion terms;
- trajectory integration with speed scales;
- loop boundary indices;
- soft loop closure;
- output transforms для trajectory и loop debug;
- synthetic feature descriptors, normalized similarity transform и mode usability checks.

Теоретический смысл: это regression layer для геометрии. Если сломать SVD alignment, resampling или robust filtering, итоговая траектория может визуально стать правдоподобной, но математически потерять повторяемость. Эти тесты ловят такие ошибки на малых контролируемых данных.

### 18.7 `tests/test_trajectory_edge_cases.py`

Проверяет редкие и отказные ветки `trajectory.py`.

Что покрывается:

- empty/degenerate anchor scores;
- resampling нулевых и повторяющихся точек;
- alignment при коротких/плохих данных;
- manual laps с почти одинаковыми boundaries;
- invalid lap direction fallback;
- closed path edge cases;
- spline/canonical loop edge cases;
- плохие `manual_lap_bounds_sec`;
- rejection по `loop_min_kept_laps`, alignment RMSE и projection RMSE;
- speed scale solver при пустом и вырожденном входе;
- empty offline tape-line estimate;
- failure branches optical flow и transform estimation.

Теоретический смысл: эти тесты защищают от скрытых падений на реальных данных, где видео может закончиться раньше, IMU может отсутствовать, frame tracking может вернуть `None`, а loop segmentation может дать слишком короткий круг.

### 18.8 `tests/test_trajectory_video_paths.py`

Проверяет ветки, которые обычно требуют OpenCV video IO, но делает это через fake capture/writer.

Что покрывается:

- `_write_tape_line_debug_video()` создает overlay frames и пишет их в fake writer;
- `tape_line` estimator проходит по синтетическому видео с простой линией;
- `generic_vio` estimator проходит по синтетическому видео с feature motion;
- `estimate_trajectory_with_details()` dispatch-логика выбирает режим и fallback.

Теоретический смысл: эти тесты проверяют glue code между OpenCV, estimator и result object. Fake objects позволяют тестировать это быстро и воспроизводимо, без хранения больших `.mp4` в репозитории.

### 18.9 `tests/test_cli_helpers.py`

Проверяет `cli.py`.

Что покрывается:

- parsing `--lap-bounds` в секундах и clock notation;
- bool normalization для YAML/CLI значений;
- определение absolute timestamps;
- path resolution относительно config path;
- time-base normalization для video и IMU;
- merge calibration config;
- полный `cli.main()` с monkeypatch dependencies;
- wiring `--mode`, `--color`, `--auto-color`, IMU scales, calibration paths, frame timestamps, `--save-debug`, `--save-loop-debug`;
- сохранение final/raw/smoothed/diagnostics/laps outputs;
- печать mode, auto-line summary, time sync и calibration paths.

Теоретический смысл: CLI-тест гарантирует, что алгоритм запускается с теми параметрами, которые пользователь реально указал. Это особенно важно для `manual_lap_bounds_sec`, calibration YAML и time sync.

### 18.10 `tests/test_fusion.py`

Проверяет `fusion.py` и `fusion_cli.py`.

Что покрывается:

- чтение trajectory CSV с колонками `t,x,y,yaw`;
- ошибка при отсутствующих колонках;
- ошибка при `nan`/`inf` координатах;
- validation на слишком короткие траектории и слишком малое `sample_count`;
- ресэмплинг LEFT и RIGHT к общему числу точек;
- phase alignment между двумя петлями;
- автоматический выбор reverse-направления LEFT, если левая камера дала ту же петлю в обратном порядке;
- rigid alignment LEFT к RIGHT без scale-подгонки;
- сохранение fused trajectory через `afn-fuse`;
- сохранение `left_aligned` и `right_resampled` debug outputs.

Теоретический смысл: двухкамерное усреднение должно уменьшать ошибку между двумя независимыми оценками одной поездки, а не подгонять траекторию под заранее нарисованный ответ. Поэтому тест строит синтетическую геометрию, где LEFT отличается от RIGHT только фазой, поворотом, переносом и направлением обхода. После alignment fused trajectory должна лечь на RIGHT reference frame с малым RMSE.

### 18.11 `tests/test_plotting.py`

Проверяет `plotting.py`.

Что покрывается:

- CSV header и строки trajectory output;
- Plotly HTML export config;
- `toImageButtonOptions` с высоким `width`, `height`, `scale`;
- обработка отсутствующего `kaleido` для PNG/JPG export;
- diagnostics CSV/plot;
- loop debug CSV/plot для kept и rejected laps;
- warnings на пустой trajectory, пустые diagnostics и пустой loop debug.

Теоретический смысл: plotting - это часть пользовательского контракта. Даже если алгоритм вернул правильные точки, плохой export или низкое качество PNG делает результат неудобным для анализа и защиты проекта.

## 19. Проверочный чеклист

Перед сдачей изменений в алгоритме или конфиге нужно проверить:

1. `configs/right_camera.yaml` содержит `target_color: blue`.
2. Примеры запуска используют `--color blue`.
3. `uv run python -m compileall src` проходит без ошибок.
4. `afn-run` создает `right_trajectory.csv` и `right_trajectory.html`.
5. В HTML сама линия траектории синяя.
6. При `--save-debug` debug-видео показывает синюю line mask или стабильные feature tracks.
7. CLI выводит путь к frame timestamps и YAML-калибровкам.
8. Если `auto` перешел из `generic_vio` в `tape_line`, итоговый mode виден в CLI output.
