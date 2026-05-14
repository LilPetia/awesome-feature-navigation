# Новый calibrated-пайплайн построения траектории

## Что добавлено

Теперь `afn-run` может использовать три файла калибровки/синхронизации:

- `timestamps2.csv` - реальные timestamps кадров из SVO/ZED;
- `imu-imu_calibration.yaml` - шумы IMU для preintegration;
- `camchain-imucam-imu_calibration.yaml` - `T_cam_imu` и `timeshift_cam_imu` из Kalibr.

Типовой запуск в этой `uv`-ветке:

```bash
uv run afn-run \
  --video data/Right_cam.mp4 \
  --imu data/imu_data.csv \
  --config configs/right_camera.yaml \
  --color red \
  --auto-color \
  --out outputs/right_trajectory
```

Лучше использовать `data/imu_data.csv`, а не заранее обнуленный `imu_fixed.csv`, если рядом есть `timestamps2.csv`: в исходном IMU CSV timestamps абсолютные, поэтому их можно синхронизировать с абсолютными timestamps кадров.

## Как работает алгоритм

1. Загружается видео.

2. Если задан `timestamps2.csv`, время каждого кадра берется из CSV, а не из `cv2.CAP_PROP_POS_MSEC`. Это важно, потому что реальное видео имеет не идеально постоянный FPS.

3. Загружается IMU CSV. Время переводится через `imu_time_scale`, гироскоп переводится через `imu_gyro_scale`.

4. Если timestamps видео и IMU выглядят как абсолютные Unix-like timestamps, оба потока приводятся к общей шкале: `t = 0` соответствует первому видеокадру.

5. Если задан camchain, применяется Kalibr time shift. В коде используется соглашение Kalibr:

```text
t_imu = t_cam + timeshift_cam_imu
```

Поэтому IMU timestamps переводятся в camera clock как:

```text
t_cam = t_imu - timeshift_cam_imu
```

6. Если задан `T_cam_imu`, векторы акселерометра и гироскопа поворачиваются из IMU frame в camera frame через верхний левый блок `3x3` этой матрицы.

7. Затем IMU можно дополнительно стабилизировать под 2D-задачу:

```yaml
imu_align_gravity: true
imu_yaw_axis: z
imu_yaw_only: true
imu_yaw_bias_window_sec: 1.0
```

Это оставляет для 2D-траектории в основном yaw-компоненту, а не смешанные roll/pitch/yaw из сырой системы координат IMU.

8. Шумы из `imu-imu_calibration.yaml` передаются в `build_default_params()` для GTSAM preintegration. Если GTSAM не установлен, проект использует внутренний минимальный интегратор; в нем шумы не участвуют в ковариации, но остальные шаги синхронизации и поворота IMU все равно работают.

9. Дальше выбирается режим траектории:

- `generic_vio` - трекает визуальные feature points между кадрами, оценивает 2D similarity motion, а yaw смешивает с IMU yaw;
- `tape_line` - использует цветную линию как дорожку, оценивает направление/смещение линии и корректирует движение робота;
- `auto` - сначала пробует `generic_vio`, если результат выглядит плохо, откатывается на `tape_line`.

10. Если включен `loop_average`, алгоритм ищет повторяющиеся круги, выравнивает несколько кругов между собой и строит более стабильную финальную петлю.

## Нормальный ли это алгоритм

Для этого проекта алгоритм нормальный как практичный lightweight-вариант: он работает с обычным MP4, CSV IMU и цветной линией, не требует ROS runtime и тяжелого SLAM-стека.

Но это не полноценный state-of-the-art visual-inertial SLAM. Слабые места:

- monocular/generic video дает относительный масштаб, если не использовать stereo/depth;
- `tape_line` зависит от видимости цветной линии;
- IMU translation через двойное интегрирование быстро уплывает, поэтому по умолчанию лучше не включать `--imu-use-translation`;
- текущий `generic_vio` оценивает 2D motion по optical flow, а не решает полноценную bundle adjustment / factor graph задачу.

Если нужна максимально правильная траектория именно по видео + IMU, лучше брать готовый VIO/SLAM:

- ORB-SLAM3 для mono/stereo/RGB-D + IMU;
- OpenVINS для visual-inertial odometry;
- VINS-Fusion для camera/IMU fusion.

Переписать такой алгоритм с нуля внутри этого репозитория было бы хуже, чем подключить готовый движок. Ближайший сильный следующий шаг для нашего проекта - использовать ZED stereo/depth и `camera_calibration2.yaml`, чтобы получить метрический масштаб без угадывания скорости.
