# awesome-feature-navigation

Репозиторий для оценки 2D-траектории по видео с цветной линией и данным IMU.

Основной пакет лежит в `awesome-feature-navigation-solution/`.

Быстрый старт:

```bash
cd awesome-feature-navigation-solution
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

afn-tune --video ../resources/Left_cam.mp4 --config examples/config.yaml --color red --auto-color
afn-run --video ../resources/Left_cam.mp4 --imu ../resources/imu_fixed.csv --imu-time-scale 1e-9 --color red --auto-color --config examples/config.yaml --out trajectory
```

Подробная инструкция:
- [awesome-feature-navigation-solution/README.md](awesome-feature-navigation-solution/README.md)
